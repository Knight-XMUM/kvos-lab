"""Unified Memory wired-cliff & residency sweep benchmark for Apple Silicon.

Sweeps KV-shaped MLX buffer pools across target GiB steps under both `nowired`
and `wired` modes, measuring:
  - `mincore` 16 KiB physical residency (GiB)
  - MLX logical active & cache memory deltas (GiB)
  - XNU `vm_stat` wired / compressed / pageouts deltas (GiB)
  - Ghost wired lock-in after releasing the pool without `mx.clear_cache()`
  - Probe matmul latency (ms) to spot compression/swap slowdown cliffs

Exports results directly to CSV/JSON for reproducible plotting and README tables.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kvos.probe import GIB, MemoryProbe


@dataclass
class BenchPoint:
    mode: str
    target_gib: float
    mincore_gib: float
    mlx_active_delta_gib: float
    mlx_cache_delta_gib: float
    wired_delta_gib: float
    compressed_delta_gib: float
    ghost_wired_after_release_gib: float
    probe_matmul_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_residency_sweep(
    steps_gib: list[float],
    modes: tuple[str, ...] = ("nowired", "wired"),
    *,
    num_layers: int = 28,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    max_safe_gib: float = 8.0,
) -> list[BenchPoint]:
    """Run an MLX KV-pool residency & wired-cliff sweep on Apple Silicon."""
    import mlx.core as mx

    for g in steps_gib:
        if g > max_safe_gib:
            raise ValueError(
                f"Step {g:.2f} GiB exceeds safety cap {max_safe_gib:.2f} GiB for 24 GiB Macs."
            )

    bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * 2  # K+V fp16
    a = mx.ones((1024, 1024), dtype=mx.float16)
    b = mx.ones((1024, 1024), dtype=mx.float16)
    mx.eval(a, b)

    results: list[BenchPoint] = []
    for mode in modes:
        for target_gib in steps_gib:
            mx.clear_cache()
            if mode == "wired":
                limit = int(mx.device_info().get("max_recommended_working_set_size", 0))
                mx.set_wired_limit(limit)
            else:
                mx.set_wired_limit(0)

            time.sleep(0.3)
            probe = MemoryProbe(logical_pool_gib=target_gib)
            slots = max(1, int(target_gib * GIB) // bytes_per_slot)
            bufs = [
                mx.zeros((slots, n_kv_heads, head_dim), dtype=mx.float16)
                for _ in range(2 * num_layers)
            ]
            mx.eval(*bufs)
            time.sleep(0.3)
            snap_eval = probe.record("eval", bufs, buffers_evaluated=True)

            t0 = time.perf_counter()
            mx.eval(a @ b)
            matmul_ms = (time.perf_counter() - t0) * 1000.0

            # Drop Python references without calling mx.clear_cache() to measure ghost wired lock-in
            del bufs
            time.sleep(0.3)
            snap_release = probe.record("released_no_clear_cache", (), buffers_evaluated=False)

            results.append(
                BenchPoint(
                    mode=mode,
                    target_gib=round(target_gib, 2),
                    mincore_gib=round(snap_eval.mincore_gib, 2),
                    mlx_active_delta_gib=round(snap_eval.mlx_active_delta_gib, 2),
                    mlx_cache_delta_gib=round(snap_eval.mlx_cache_delta_gib, 2),
                    wired_delta_gib=round(snap_eval.wired_delta_gib, 2),
                    compressed_delta_gib=round(snap_eval.compressed_delta_gib, 2),
                    ghost_wired_after_release_gib=round(snap_release.ghost_wired_gib, 2),
                    probe_matmul_ms=round(matmul_ms, 2),
                )
            )
            mx.clear_cache()
            mx.set_wired_limit(0)

    return results


def export_bench_results(points: list[BenchPoint], out_path: Path) -> None:
    """Save benchmark points to CSV or JSON based on file suffix."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [p.to_dict() for p in points]
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_bench_table(points: list[BenchPoint]) -> str:
    header = (
        f"{'Mode':<8} | {'Target':>6} | {'mincore':>7} | {'OS Wired':>8} | "
        f"{'Comp':>6} | {'Ghost Lock':>10} | {'Matmul':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for p in points:
        lines.append(
            f"{p.mode:<8} | {p.target_gib:>5.2f}G | {p.mincore_gib:>6.2f}G | "
            f"{p.wired_delta_gib:>+7.2f}G | {p.compressed_delta_gib:>+5.2f}G | "
            f"{p.ghost_wired_after_release_gib:>9.2f}G | {p.probe_matmul_ms:>6.2f}ms"
        )
    return "\n".join(lines)
