"""Adaptive Unified Memory KV Governor for Apple Silicon.

Bridges XNU physical memory signals (`vm_stat` wired/compressed/pageouts) and
MLX allocator telemetry (`active` vs `cache`) with dual-plane KV block policy
(`kvos.policy.DualPlaneKVTracker`).

State machine:
  - GREEN  : Plenty of headroom. Keep `wired_limit` enabled and retain prefix cache.
  - YELLOW : Approaching unified-memory budget or detecting "Ghost Wired Pages"
             (cached MLX buffers locked in wired RAM). Evict cold context-plane
             blocks and trigger `mx.clear_cache()`.
  - RED    : High wired utilization or active swap/pageouts. Drop `wired_limit`
             (unwire) and aggressively evict context-plane blocks to prevent
             macOS compressor/swap cliff, while NEVER evicting live in-flight blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

from kvos.policy import DualPlaneKVTracker
from kvos.probe import mlx_memory_metrics, vm_stat_metrics


class PressureLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class GovernorThresholds:
    """Memory thresholds tuned for 16–24 GiB Apple Silicon unified memory."""

    total_ram_gib: float = 24.0
    yellow_wired_ratio: float = 0.60
    red_wired_ratio: float = 0.72
    yellow_compressed_gib: float = 4.0
    red_compressed_gib: float = 7.0
    ghost_wired_trigger_gib: float = 1.0
    pageout_delta_red_gib: float = 0.05
    yellow_evict_batch: int = 8
    red_evict_fraction: float = 0.50


@dataclass
class GovernorDecision:
    level: PressureLevel
    reasons: list[str]
    evicted_blocks: list[int]
    clear_cache_called: bool
    unwired: bool
    wired_gib: float
    compressed_gib: float
    mlx_cache_gib: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


class KVGovernor:
    """Closed-loop Apple Silicon KV cache governor with pluggable engine hooks."""

    def __init__(
        self,
        tracker: DualPlaneKVTracker,
        thresholds: GovernorThresholds | None = None,
        *,
        on_evict_block: Callable[[int], None] | None = None,
        on_clear_cache: Callable[[], None] | None = None,
        on_set_wired_limit: Callable[[int], None] | None = None,
    ) -> None:
        self.tracker = tracker
        self.thresholds = thresholds or GovernorThresholds()
        self._on_evict_block = on_evict_block
        self._on_clear_cache = on_clear_cache
        self._on_set_wired_limit = on_set_wired_limit
        self._last_pageouts_gib: float | None = None
        self.history: list[GovernorDecision] = []

    def evaluate(
        self,
        *,
        vm: dict[str, float] | None = None,
        mlx_m: dict[str, float] | None = None,
    ) -> GovernorDecision:
        """Sample unified memory telemetry, classify pressure, and apply guardrails."""
        vm = vm if vm is not None else vm_stat_metrics()
        mlx_m = mlx_m if mlx_m is not None else mlx_memory_metrics()

        wired_gib = vm.get("wired_gib", 0.0)
        compressed_gib = vm.get("compressed_gib", 0.0)
        pageouts_gib = vm.get("pageouts_gib", 0.0)
        mlx_cache_gib = mlx_m.get("mlx_cache_gib", 0.0)

        pageout_delta = 0.0
        if self._last_pageouts_gib is not None:
            pageout_delta = max(0.0, pageouts_gib - self._last_pageouts_gib)
        self._last_pageouts_gib = pageouts_gib

        t = self.thresholds
        wired_ratio = wired_gib / max(t.total_ram_gib, 1e-6)
        reasons: list[str] = []
        level = PressureLevel.GREEN

        # Check RED triggers first
        if wired_ratio >= t.red_wired_ratio:
            level = PressureLevel.RED
            reasons.append(f"wired_ratio={wired_ratio:.2f} >= {t.red_wired_ratio:.2f}")
        if compressed_gib >= t.red_compressed_gib:
            level = PressureLevel.RED
            reasons.append(f"compressed={compressed_gib:.2f}GiB >= {t.red_compressed_gib:.2f}GiB")
        if pageout_delta >= t.pageout_delta_red_gib:
            level = PressureLevel.RED
            reasons.append(f"pageouts_delta=+{pageout_delta:.3f}GiB")

        # Check YELLOW triggers if not already RED
        if level == PressureLevel.GREEN:
            if wired_ratio >= t.yellow_wired_ratio:
                level = PressureLevel.YELLOW
                reasons.append(f"wired_ratio={wired_ratio:.2f} >= {t.yellow_wired_ratio:.2f}")
            if compressed_gib >= t.yellow_compressed_gib:
                level = PressureLevel.YELLOW
                reasons.append(
                    f"compressed={compressed_gib:.2f}GiB >= {t.yellow_compressed_gib:.2f}GiB"
                )
            if mlx_cache_gib >= t.ghost_wired_trigger_gib:
                level = PressureLevel.YELLOW
                reasons.append(
                    f"mlx_cache={mlx_cache_gib:.2f}GiB >= {t.ghost_wired_trigger_gib:.2f}GiB (ghost-wired risk)"
                )

        evicted: list[int] = []
        clear_cache_called = False
        unwired = False

        if level == PressureLevel.YELLOW:
            for _ in range(t.yellow_evict_batch):
                bid = self.tracker.evict_one()
                if bid is None:
                    break
                evicted.append(bid)
                if self._on_evict_block is not None:
                    self._on_evict_block(bid)
            if self._on_clear_cache is not None:
                self._on_clear_cache()
                clear_cache_called = True

        elif level == PressureLevel.RED:
            ctx_count = self.tracker.stats().context_blocks
            target = max(t.yellow_evict_batch, int(ctx_count * t.red_evict_fraction))
            for _ in range(target):
                bid = self.tracker.evict_one()
                if bid is None:
                    break
                evicted.append(bid)
                if self._on_evict_block is not None:
                    self._on_evict_block(bid)
            if self._on_set_wired_limit is not None:
                self._on_set_wired_limit(0)
                unwired = True
            if self._on_clear_cache is not None:
                self._on_clear_cache()
                clear_cache_called = True

        decision = GovernorDecision(
            level=level,
            reasons=reasons or ["within safe unified-memory headroom"],
            evicted_blocks=evicted,
            clear_cache_called=clear_cache_called,
            unwired=unwired,
            wired_gib=round(wired_gib, 2),
            compressed_gib=round(compressed_gib, 2),
            mlx_cache_gib=round(mlx_cache_gib, 2),
        )
        self.history.append(decision)
        return decision
