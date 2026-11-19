"""Tests for kvos.policy (dual-plane live/context tracking & allocator ordering)."""

from __future__ import annotations

import unittest

from kvos.policy import DualPlaneKVTracker, FifoAllocator, LowestIdAllocator


class TestKvosPolicy(unittest.TestCase):
    def test_fifo_vs_lowest_allocator_footprint(self) -> None:
        fifo = FifoAllocator(num_blocks=10)
        lowest = LowestIdAllocator(num_blocks=10)

        a_f = [fifo.allocate() for _ in range(3)]
        a_l = [lowest.allocate() for _ in range(3)]
        self.assertEqual(a_f, [1, 2, 3])
        self.assertEqual(a_l, [1, 2, 3])

        for b in a_f:
            fifo.free(b)
        for b in a_l:
            lowest.free(b)

        self.assertEqual([fifo.allocate() for _ in range(3)], [4, 5, 6])
        self.assertEqual([lowest.allocate() for _ in range(3)], [1, 2, 3])

    def test_dual_plane_lru_never_evicts_live(self) -> None:
        t = DualPlaneKVTracker(policy="lru")
        t.pin_live(1)
        t.pin_live(2)
        t.release_to_context(2)
        t.pin_live(3)
        t.release_to_context(3)
        t.touch_context(2)

        self.assertEqual(t.evict_one(), 3)
        self.assertEqual(t.evict_one(), 2)
        self.assertIsNone(t.evict_one())
        self.assertEqual(t.stats().live_blocks, 1)

    def test_dual_plane_clock_second_chance(self) -> None:
        t = DualPlaneKVTracker(policy="clock")
        for bid in (10, 20, 30):
            t.pin_live(bid)
            t.release_to_context(bid)

        self.assertEqual(t.evict_one(), 10)
        self.assertEqual(t.stats().context_blocks, 2)


if __name__ == "__main__":
    unittest.main()
