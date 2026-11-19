"""Experiment 2: SGLang MlxAttentionKVPool Physical Residency & Wired Lock-in.

Demonstrates two Apple Silicon-specific KV cache behaviors in SGLang's MLX backend
(`python/sglang/srt/hardware_backend/mlx/kv_cache/mlx_kv_pool.py`):
  1. Lazy Initialization Trap: `MlxAttentionKVPool.__init__` and `clear()` allocate
     per-layer K/V arrays with `mx.zeros(...)` without calling `mx.eval(...)`.
     Immediately after `__init__`, physical residency (`mincore`) and wired rise are 0 GiB.
  2. Wired Memory Lock-in ("Ghost Wired Pages"): `MlxModelRunner` sets
     `mx.set_wired_limit(max_recommended_working_set_size)` by default. When `pool.clear()`
     re-assigns `self.k_buffers` / `self.v_buffers` to new lazy `mx.zeros(...)` arrays:
     - Without `mx.eval(*pool.all_buffers())`, the previous physical buffers move into
       MLX's metal buffer cache and remain wired down (`wired_delta` stays at ~2.23 GiB
       for a 2.00 GiB pool).
     - Calling `clear() + mx.clear_cache()` or evaluating cleanly under controlled wired
       limits is required to release physical wired pages back to macOS.

Usage (inside an environment with `mlx` and `sglang` in PYTHONPATH):
  PYTHONPATH=/Users/yurongguang/Develop/oss-contrib/sglang/python:. \\
    /Users/yurongguang/Develop/oss-contrib/vllm-metal/.venv-vllm-metal/bin/python \\
    experiments/sglang_kv_pool_residency.py --pool-gib 2.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvos.probe import GIB, MemoryProbe  # noqa: E402
from sglang.srt.hardware_backend.mlx.kv_cache.mlx_kv_pool import MlxAttentionKVPool  # noqa: E402


def run_case(mode: str, pool_gib: float, layers: int, kv_heads: int, head_dim: int) -> str:
    mx.clear_cache()
    if mode == "wired":
        limit = int(mx.metal.device_info().get("max_recommended_working_set_size", 0))
        mx.set_wired_limit(limit)
    else:
        mx.set_wired_limit(0)

    time.sleep(0.3)
    probe = MemoryProbe(logical_pool_gib=pool_gib)

    bytes_per_slot = 2 * layers * kv_heads * head_dim * 2  # K+V, fp16
    pool_size = int(pool_gib * GIB) // bytes_per_slot

    pool = MlxAttentionKVPool(
        pool_size=pool_size,
        num_layers=layers,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=mx.float16,
    )
    probe.record("1. after __init__ (lazy)", pool.all_buffers(), buffers_evaluated=False)

    mx.eval(*pool.all_buffers())
    time.sleep(0.4)
    probe.record("2. after mx.eval(pool)", pool.all_buffers(), buffers_evaluated=True)

    mx.eval(mx.ones((1024, 1024)) @ mx.ones((1024, 1024)))
    time.sleep(0.4)
    probe.record("3. after unrelated op", pool.all_buffers(), buffers_evaluated=True)

    pool.clear()
    time.sleep(0.4)
    probe.record("4. after clear() (lazy)", pool.all_buffers(), buffers_evaluated=False)

    mx.eval(*pool.all_buffers())
    time.sleep(0.4)
    probe.record("5. after clear()+eval", pool.all_buffers(), buffers_evaluated=True)

    pool.clear()
    mx.clear_cache()
    time.sleep(0.4)
    probe.record("6. clear()+clear_cache()", pool.all_buffers(), buffers_evaluated=False)

    del pool
    mx.clear_cache()
    mx.set_wired_limit(0)

    title = (
        f"=== SGLang MlxAttentionKVPool | mode={mode} | target={pool_gib:.2f} GiB "
        f"(slots={pool_size}, buffers={2 * layers}) ==="
    )
    return f"{title}\n{probe.format_table()}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-gib", type=float, default=2.0)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()
    if args.pool_gib > 8.0:
        sys.exit("Refusing pool > 8 GiB on 24 GB MacBook Air.")

    for mode in ("nowired", "wired"):
        print(run_case(mode, args.pool_gib, args.layers, args.kv_heads, args.head_dim), flush=True)


if __name__ == "__main__":
    main()
