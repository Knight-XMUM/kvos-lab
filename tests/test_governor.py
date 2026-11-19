"""Tests for kvos.governor and extended CLI subcommands (doctor, sim)."""

from __future__ import annotations

import unittest

from kvos.cli import main as cli_main
from kvos.governor import GovernorThresholds, KVGovernor, PressureLevel
from kvos.policy import DualPlaneKVTracker


class TestKvosGovernor(unittest.TestCase):
    def test_green_state_no_eviction(self) -> None:
        tracker = DualPlaneKVTracker(policy="lru")
        for bid in (1, 2, 3):
            tracker.pin_live(bid)
            tracker.release_to_context(bid)

        gov = KVGovernor(tracker, GovernorThresholds(total_ram_gib=24.0))
        d = gov.evaluate(
            vm={"wired_gib": 6.0, "compressed_gib": 1.0, "pageouts_gib": 0.0},
            mlx_m={"mlx_active_gib": 4.0, "mlx_cache_gib": 0.2},
        )
        self.assertEqual(d.level, PressureLevel.GREEN)
        self.assertEqual(d.evicted_blocks, [])
        self.assertFalse(d.clear_cache_called)
        self.assertFalse(d.unwired)

    def test_yellow_ghost_wired_triggers_evict_and_clear_cache(self) -> None:
        tracker = DualPlaneKVTracker(policy="lru")
        tracker.pin_live(1)  # live block must be protected
        for bid in (2, 3, 4):
            tracker.pin_live(bid)
            tracker.release_to_context(bid)

        cleared = []
        evicted_cb = []
        gov = KVGovernor(
            tracker,
            GovernorThresholds(total_ram_gib=24.0, ghost_wired_trigger_gib=1.0, yellow_evict_batch=2),
            on_evict_block=evicted_cb.append,
            on_clear_cache=lambda: cleared.append(True),
        )
        d = gov.evaluate(
            vm={"wired_gib": 8.0, "compressed_gib": 1.0, "pageouts_gib": 0.0},
            mlx_m={"mlx_active_gib": 0.0, "mlx_cache_gib": 2.23},
        )
        self.assertEqual(d.level, PressureLevel.YELLOW)
        self.assertEqual(d.evicted_blocks, [2, 3])
        self.assertEqual(evicted_cb, [2, 3])
        self.assertTrue(d.clear_cache_called)
        self.assertEqual(len(cleared), 1)
        self.assertFalse(d.unwired)
        self.assertEqual(tracker.stats().live_blocks, 1)

    def test_red_state_unwires_and_preserves_live_blocks(self) -> None:
        tracker = DualPlaneKVTracker(policy="clock")
        tracker.pin_live(99)  # active request block
        for bid in range(10, 20):
            tracker.pin_live(bid)
            tracker.release_to_context(bid)

        wired_limits: list[int] = []
        cleared: list[bool] = []
        gov = KVGovernor(
            tracker,
            GovernorThresholds(total_ram_gib=24.0, red_wired_ratio=0.72),
            on_clear_cache=lambda: cleared.append(True),
            on_set_wired_limit=wired_limits.append,
        )
        d = gov.evaluate(
            vm={"wired_gib": 18.0, "compressed_gib": 8.0, "pageouts_gib": 1.0},
            mlx_m={"mlx_active_gib": 10.0, "mlx_cache_gib": 3.0},
        )
        self.assertEqual(d.level, PressureLevel.RED)
        self.assertTrue(d.unwired)
        self.assertEqual(wired_limits, [0])
        self.assertTrue(d.clear_cache_called)
        self.assertNotIn(99, d.evicted_blocks)
        self.assertEqual(tracker.stats().live_blocks, 1)

    def test_cli_doctor_and_sim(self) -> None:
        self.assertEqual(cli_main(["doctor", "--json"]), 0)
        self.assertEqual(cli_main(["sim", "--blocks", "128", "--requests", "30", "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
