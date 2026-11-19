"""kvos: Unified Memory & KV Residency Observatory for Apple Silicon (MLX / Metal)."""

from kvos.apple import AppleSiliconInfo, get_apple_silicon_info
from kvos.bench import BenchPoint, export_bench_results, format_bench_table, run_residency_sweep
from kvos.budget import KVBudgetPlan, compute_kv_budget
from kvos.governor import GovernorDecision, GovernorThresholds, KVGovernor, PressureLevel
from kvos.policy import DualPlaneKVTracker, DualPlaneStats, FifoAllocator, LowestIdAllocator
from kvos.probe import MemoryProbe, ProbeSnapshot, mincore_bytes, mincore_gib, vm_stat_metrics

__all__ = [
    "AppleSiliconInfo",
    "BenchPoint",
    "DualPlaneKVTracker",
    "DualPlaneStats",
    "FifoAllocator",
    "GovernorDecision",
    "GovernorThresholds",
    "KVBudgetPlan",
    "KVGovernor",
    "LowestIdAllocator",
    "MemoryProbe",
    "PressureLevel",
    "ProbeSnapshot",
    "compute_kv_budget",
    "export_bench_results",
    "format_bench_table",
    "get_apple_silicon_info",
    "mincore_bytes",
    "mincore_gib",
    "run_residency_sweep",
    "vm_stat_metrics",
]
