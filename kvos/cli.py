"""Command-line interface for kvos-lab."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from kvos.apple import get_apple_silicon_info
from kvos.bench import export_bench_results, format_bench_table, run_residency_sweep
from kvos.budget import compute_kv_budget
from kvos.governor import GovernorThresholds, KVGovernor
from kvos.plot import render_bench_svg
from kvos.policy import DualPlaneKVTracker, FifoAllocator, LowestIdAllocator


def _cmd_status(args: argparse.Namespace) -> int:
    info = get_apple_silicon_info()
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
        return 0

    print("=== kvos: Apple Silicon Unified Memory Status ===")
    print(f"Machine / Chip : {info.model} ({info.chip})")
    print(f"Total RAM      : {info.total_ram_gib:.2f} GiB")
    print(f"MLX Rec. Limit : {info.mlx_recommended_working_set_gib or 'N/A'} GiB")
    print(
        f"IOGPU Wired Cap: "
        f"{info.iogpu_wired_limit_mb if info.iogpu_wired_limit_mb is not None else 'default (unset)'}"
    )
    print(
        f"XNU vm_stat    : wired={info.vm_wired_gib:.2f} GiB, "
        f"active={info.vm_active_gib:.2f} GiB, "
        f"compressed={info.vm_compressed_gib:.2f} GiB, "
        f"pageouts={info.vm_pageouts_gib:.2f} GiB"
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Audit current Mac unified-memory readiness for local MLX LLM serving."""
    info = get_apple_silicon_info()
    tracker = DualPlaneKVTracker(policy="lru")
    gov = KVGovernor(tracker, GovernorThresholds(total_ram_gib=max(info.total_ram_gib, 8.0)))
    decision = gov.evaluate()

    cache_env = os.environ.get("SGLANG_MLX_CACHE_LIMIT_GB")
    checks = [
        ("Chip / RAM", f"{info.chip} / {info.total_ram_gib:.2f} GiB"),
        ("MLX Available", "yes" if info.mlx_recommended_working_set_gib is not None else "no (install mlx)"),
        ("Pressure State", f"{decision.level.value} ({'; '.join(decision.reasons)})"),
        (
            "SGLANG_MLX_CACHE_LIMIT_GB",
            cache_env if cache_env is not None else "unset (recommend setting 2.0-4.0 to cap ghost wired pages)",
        ),
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "apple_silicon": info.to_dict(),
                    "governor": decision.to_dict(),
                    "sglang_mlx_cache_limit_gb": cache_env,
                },
                indent=2,
            )
        )
        return 0

    print("=== kvos doctor: Apple Silicon LLM Serving Readiness ===")
    for k, v in checks:
        print(f"{k:<26}: {v}")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    info = get_apple_silicon_info()
    ram_gib = args.ram_gib if args.ram_gib is not None else info.total_ram_gib
    if ram_gib <= 0:
        ram_gib = 24.0

    plan = compute_kv_budget(
        total_ram_gib=ram_gib,
        model_weights_gib=args.weights_gib,
        num_layers=args.layers,
        n_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_dtype_bytes=args.dtype_bytes,
        activation_scratch_gib=args.scratch_gib,
        current_wired_gib=info.vm_wired_gib if args.ram_gib is None else None,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    print("=== kvos: Apple Silicon KV Cache Budget ===")
    print(f"Total RAM          : {plan.total_ram_gib:.2f} GiB")
    print(f"OS + App Headroom  : {plan.os_and_app_headroom_gib:.2f} GiB")
    print(f"Model Weights      : {plan.model_weights_gib:.2f} GiB")
    print(f"Metal Scratch      : {plan.activation_scratch_gib:.2f} GiB")
    print(f"Safe KV Budget     : {plan.safe_kv_budget_gib:.2f} GiB")
    print(f"Bytes / Token Slot : {plan.bytes_per_token_slot:,} B")
    print(f"Max Safe KV Tokens : {plan.max_safe_kv_tokens:,}")
    print(f"Wired Risk Level   : {plan.wired_risk_level}")
    print(f"Suggested Env      : SGLANG_MLX_CACHE_LIMIT_GB={plan.recommended_mlx_cache_limit_gb}")
    for note in plan.notes:
        print(f"Note               : {note}")
    return 0


def _cmd_sim(args: argparse.Namespace) -> int:
    """Quickly compare FIFO vs LowestId peak buffer span (blocks touched)."""
    import random

    rng = random.Random(args.seed)
    n_blocks = args.blocks
    out: dict[str, dict[str, int]] = {}
    for name, cls in (("fifo", FifoAllocator), ("lowest", LowestIdAllocator)):
        alloc = cls(num_blocks=n_blocks)
        touched: set[int] = set()
        for _ in range(args.requests):
            need = rng.randint(4, max(4, args.max_req_blocks))
            if len(alloc) < need:
                continue
            ids = [alloc.allocate() for _ in range(need)]
            touched.update(ids)
            for bid in ids:
                alloc.free(bid)
        out[name] = {
            "blocks_touched": len(touched),
            "max_block_id": max(touched) if touched else 0,
            "total_blocks": n_blocks,
        }
        rng.seed(args.seed)

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"=== kvos sim: FIFO vs LowestId Footprint ({args.requests} requests, {n_blocks} blocks) ===")
    for name, stats in out.items():
        pct = 100.0 * stats["blocks_touched"] / max(1, n_blocks)
        print(
            f"{name:<8} | touched={stats['blocks_touched']:>4}/{n_blocks} ({pct:5.1f}%) "
            f"| highest_block_id={stats['max_block_id']}"
        )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    steps = [float(x.strip()) for x in args.steps.split(",") if x.strip()]
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    points = run_residency_sweep(
        steps_gib=steps,
        modes=modes,
        num_layers=args.layers,
        n_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        max_safe_gib=args.max_safe_gib,
    )
    print(format_bench_table(points))
    if args.out:
        out_path = Path(args.out)
        export_bench_results(points, out_path)
        print(f"\nSaved {len(points)} rows to {out_path}")
        if args.plot and out_path.suffix.lower() == ".csv":
            svg_path = render_bench_svg(out_path, Path(args.plot))
            print(f"Rendered SVG chart to {svg_path}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    svg_path = render_bench_svg(Path(args.input), Path(args.out), title=args.title)
    print(f"Rendered SVG chart to {svg_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvos",
        description="Unified Memory & KV Residency Observatory for Apple Silicon",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Inspect Apple Silicon RAM, wired limit, and XNU vm_stat")
    p_status.add_argument("--json", action="store_true", help="Output JSON")
    p_status.set_defaults(func=_cmd_status)

    p_doctor = sub.add_parser("doctor", help="Audit Mac unified-memory readiness & pressure state")
    p_doctor.add_argument("--json", action="store_true", help="Output JSON")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_budget = sub.add_parser("budget", help="Compute safe KV token budget for unified memory")
    p_budget.add_argument("--ram-gib", type=float, default=None, help="Override detected RAM (GiB)")
    p_budget.add_argument("--weights-gib", type=float, required=True, help="Quantized model weights size (GiB)")
    p_budget.add_argument("--layers", type=int, default=28, help="Number of attention layers")
    p_budget.add_argument("--kv-heads", type=int, default=8, help="Number of KV heads per layer")
    p_budget.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    p_budget.add_argument("--dtype-bytes", type=int, default=2, help="KV dtype size in bytes (2=fp16/bf16)")
    p_budget.add_argument("--scratch-gib", type=float, default=1.5, help="Activation/logits scratch headroom (GiB)")
    p_budget.add_argument("--json", action="store_true", help="Output JSON")
    p_budget.set_defaults(func=_cmd_budget)

    p_sim = sub.add_parser("sim", help="Compare FIFO vs LowestId KV block footprint")
    p_sim.add_argument("--blocks", type=int, default=512, help="Total KV blocks in pool")
    p_sim.add_argument("--requests", type=int, default=80, help="Number of simulated requests")
    p_sim.add_argument("--max-req-blocks", type=int, default=32, help="Max blocks per request")
    p_sim.add_argument("--seed", type=int, default=0, help="Random seed")
    p_sim.add_argument("--json", action="store_true", help="Output JSON")
    p_sim.set_defaults(func=_cmd_sim)

    p_bench = sub.add_parser("bench", help="Run MLX unified-memory residency & wired-cliff sweep")
    p_bench.add_argument("--steps", type=str, default="1.0,2.0,4.0,6.0,8.0", help="Comma-separated pool sizes (GiB)")
    p_bench.add_argument("--modes", type=str, default="nowired,wired", help="Comma-separated modes")
    p_bench.add_argument("--layers", type=int, default=28, help="Number of layers")
    p_bench.add_argument("--kv-heads", type=int, default=8, help="Number of KV heads")
    p_bench.add_argument("--head-dim", type=int, default=128, help="Head dim")
    p_bench.add_argument("--max-safe-gib", type=float, default=8.0, help="Safety cap (GiB)")
    p_bench.add_argument("--out", type=str, default="", help="Optional CSV/JSON output path")
    p_bench.add_argument("--plot", type=str, default="", help="Optional SVG output path (requires --out .csv)")
    p_bench.set_defaults(func=_cmd_bench)

    p_plot = sub.add_parser("plot", help="Render an SVG chart from a kvos bench CSV file")
    p_plot.add_argument("--input", "-i", type=str, required=True, help="Input CSV path")
    p_plot.add_argument("--out", "-o", type=str, required=True, help="Output SVG path")
    p_plot.add_argument(
        "--title",
        type=str,
        default="Apple Silicon Unified Memory KV Residency & Wired Lock-in (M4 Air 24GB)",
        help="Chart title",
    )
    p_plot.set_defaults(func=_cmd_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
