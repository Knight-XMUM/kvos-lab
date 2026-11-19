"""Dual-plane KV block & eviction primitives (distilled from Infra/nano-vllm/kvos).

Unifies OS virtual-memory cache concepts with Apple Silicon residency behavior:
  - Live plane (`pinned` / `wired`): blocks actively owned by running requests.
    Must never be evicted mid-flight.
  - Context plane (`evictable` / `prefix`): cached prefix blocks retained after
    request completion. Subject to eviction when physical memory or wired limits
    tighten.
  - Allocator order (`Fifo` vs `LowestId`): determines how quickly fragmented
    allocations touch all Metal buffers in a pool (see Experiment 1).
  - Eviction policy (`LRU`, `CLOCK`): selects context-plane victims under memory
    pressure without disturbing live requests.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass


class FifoAllocator:
    """Standard vLLM-style FIFO free queue (cycles through all block IDs)."""

    def __init__(self, num_blocks: int, reserve_zero: bool = True) -> None:
        start = 1 if reserve_zero else 0
        self._free: deque[int] = deque(range(start, num_blocks))

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("Out of KV blocks")
        return self._free.popleft()

    def free(self, block_id: int) -> None:
        self._free.append(block_id)

    def __len__(self) -> int:
        return len(self._free)


class LowestIdAllocator:
    """Min-heap allocator that always reuses the lowest free block ID.

    On Apple Silicon, keeping block IDs compact prevents touching higher Metal
    buffer chunks and keeps `mincore` physical residency bounded.
    """

    def __init__(self, num_blocks: int, reserve_zero: bool = True) -> None:
        start = 1 if reserve_zero else 0
        self._heap: list[int] = list(range(start, num_blocks))
        heapq.heapify(self._heap)

    def allocate(self) -> int:
        if not self._heap:
            raise RuntimeError("Out of KV blocks")
        return heapq.heappop(self._heap)

    def free(self, block_id: int) -> None:
        heapq.heappush(self._heap, block_id)

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class DualPlaneStats:
    live_blocks: int
    context_blocks: int
    evictions: int


class DualPlaneKVTracker:
    """Tracks live (pinned) vs context (evictable) KV blocks with LRU or CLOCK."""

    def __init__(self, policy: str = "lru") -> None:
        if policy not in {"lru", "clock"}:
            raise ValueError(f"Unsupported eviction policy: {policy}")
        self.policy = policy
        self._live: set[int] = set()
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._clock_ring: list[int] = []
        self._clock_ref: dict[int, bool] = {}
        self._clock_hand: int = 0
        self.evictions: int = 0

    def pin_live(self, block_id: int) -> None:
        """Mark block as live (in-flight request); remove from evictable context."""
        self._live.add(block_id)
        if self.policy == "lru":
            self._lru.pop(block_id, None)
        else:
            self._remove_from_clock(block_id)

    def release_to_context(self, block_id: int) -> None:
        """Move a finished block from live plane to evictable context plane."""
        self._live.discard(block_id)
        if self.policy == "lru":
            self._lru[block_id] = None
            self._lru.move_to_end(block_id)
        else:
            if block_id not in self._clock_ref:
                self._clock_ring.append(block_id)
            self._clock_ref[block_id] = True

    def touch_context(self, block_id: int) -> None:
        """Record a prefix-cache hit on a context block."""
        if block_id in self._live:
            return
        if self.policy == "lru" and block_id in self._lru:
            self._lru.move_to_end(block_id)
        elif self.policy == "clock" and block_id in self._clock_ref:
            self._clock_ref[block_id] = True

    def evict_one(self) -> int | None:
        """Evict one block from the context plane (never touches live blocks)."""
        if self.policy == "lru":
            if not self._lru:
                return None
            bid, _ = self._lru.popitem(last=False)
            assert bid not in self._live
            self.evictions += 1
            return bid

        if not self._clock_ring:
            return None
        while True:
            if self._clock_hand >= len(self._clock_ring):
                self._clock_hand = 0
            bid = self._clock_ring[self._clock_hand]
            if self._clock_ref.get(bid, False):
                self._clock_ref[bid] = False
                self._clock_hand = (self._clock_hand + 1) % len(self._clock_ring)
            else:
                self._clock_ring.pop(self._clock_hand)
                self._clock_ref.pop(bid, None)
                if self._clock_ring:
                    self._clock_hand %= len(self._clock_ring)
                else:
                    self._clock_hand = 0
                assert bid not in self._live
                self.evictions += 1
                return bid

    def _remove_from_clock(self, block_id: int) -> None:
        if block_id not in self._clock_ref:
            return
        idx = self._clock_ring.index(block_id)
        self._clock_ring.pop(idx)
        self._clock_ref.pop(block_id, None)
        if not self._clock_ring:
            self._clock_hand = 0
        elif idx < self._clock_hand:
            self._clock_hand -= 1
        elif self._clock_hand >= len(self._clock_ring):
            self._clock_hand = 0

    def stats(self) -> DualPlaneStats:
        ctx_len = len(self._lru) if self.policy == "lru" else len(self._clock_ring)
        return DualPlaneStats(
            live_blocks=len(self._live),
            context_blocks=ctx_len,
            evictions=self.evictions,
        )
