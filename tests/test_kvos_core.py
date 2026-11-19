"""Unit tests for kvos core modules (runs with Python stdlib + numpy, no GPU required)."""

from __future__ import annotations

import ctypes
import unittest

from kvos.budget import compute_kv_budget
from kvos.cli import main as cli_main
from kvos.probe import PAGE_SIZE, MemoryProbe, mincore_bytes


class TestKvosCore(unittest.TestCase):
    def test_mincore_bytes_on_touched_buffer(self) -> None:
        nbytes = PAGE_SIZE * 4
        buf = (ctypes.c_char * nbytes)()
        # Touch first and last page so XNU faults them in
        buf[0] = b"a"
        buf[nbytes - 1] = b"z"
        addr = ctypes.addressof(buf)
        resident = mincore_bytes(addr, nbytes)
        self.assertGreaterEqual(resident, PAGE_SIZE)
        self.assertLessEqual(resident, nbytes)

    def test_budget_24gb_qwen_7b_q4(self) -> None:
        plan = compute_kv_budget(
            total_ram_gib=24.0,
            model_weights_gib=4.5,
            num_layers=28,
            n_kv_heads=4,
            head_dim=128,
            kv_dtype_bytes=2,
            os_headroom_gib=6.0,
            activation_scratch_gib=1.5,
        )
        self.assertAlmostEqual(plan.safe_kv_budget_gib, 12.0, places=2)
        self.assertEqual(plan.bytes_per_token_slot, 2 * 28 * 4 * 128 * 2)
        self.assertGreater(plan.max_safe_kv_tokens, 200_000)
        self.assertEqual(plan.wired_risk_level, "LOW")

    def test_budget_tight_memory_marks_high_risk(self) -> None:
        plan = compute_kv_budget(
            total_ram_gib=16.0,
            model_weights_gib=10.5,
            num_layers=32,
            n_kv_heads=8,
            head_dim=128,
            os_headroom_gib=5.0,
            activation_scratch_gib=1.5,
        )
        self.assertEqual(plan.safe_kv_budget_gib, 0.0)
        self.assertEqual(plan.wired_risk_level, "HIGH")

    def test_memory_probe_format_table(self) -> None:
        probe = MemoryProbe(logical_pool_gib=2.0)
        snap = probe.record("init (lazy)", buffers_evaluated=False)
        self.assertEqual(snap.mincore_gib, 0.0)
        table = probe.format_table()
        self.assertIn("Ghost Lock", table)
        self.assertIn("init (lazy)", table)

    def test_cli_budget_runs(self) -> None:
        rc = cli_main(["budget", "--ram-gib", "24", "--weights-gib", "4.5", "--json"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
