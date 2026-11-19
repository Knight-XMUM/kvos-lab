"""Zero-dependency SVG chart generator for kvos bench CSV results.

Generates a clean 2-panel SVG suitable for embedding directly in GitHub READMEs
and postgraduate defense slides without requiring matplotlib.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_bench_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "mode": r["mode"],
                    "target_gib": float(r["target_gib"]),
                    "mincore_gib": float(r["mincore_gib"]),
                    "wired_delta_gib": float(r["wired_delta_gib"]),
                    "compressed_delta_gib": float(r["compressed_delta_gib"]),
                    "pageouts_delta_gib": float(r.get("pageouts_delta_gib", 0.0)),
                    "ghost_wired_after_release_gib": float(r["ghost_wired_after_release_gib"]),
                    "probe_matmul_ms": float(r["probe_matmul_ms"]),
                }
            )
    return rows


def _scale(val: float, vmin: float, vmax: float, p0: float, p1: float) -> float:
    if vmax <= vmin:
        return (p0 + p1) / 2.0
    return p0 + (val - vmin) / (vmax - vmin) * (p1 - p0)


def render_bench_svg(
    csv_path: Path,
    out_svg_path: Path,
    *,
    title: str = "Apple Silicon Unified Memory KV Residency & Wired Lock-in (M4 Air 24GB)",
) -> Path:
    """Render a 2-panel SVG chart from a `kvos bench` CSV file."""
    rows = load_bench_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    for lst in by_mode.values():
        lst.sort(key=lambda x: x["target_gib"])

    xs = sorted({r["target_gib"] for r in rows})
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmin = max(0.0, xmin - 1.0)
        xmax = xmax + 1.0

    # Panel 1: Ghost Wired Lock-in (GiB) vs Target Pool (GiB)
    y1_max = max(1.0, max(r["ghost_wired_after_release_gib"] for r in rows) * 1.15, xmax * 1.05)
    # Panel 2: XNU Compressor Delta (GiB) vs Target Pool (GiB)
    y2_vals = [r["compressed_delta_gib"] for r in rows]
    y2_min = min(0.0, min(y2_vals))
    y2_max = max(0.8, max(y2_vals) * 1.25)

    W, H = 960, 420
    colors = {
        "wired": "#E11D48",    # rose-600
        "nowired": "#2563EB",  # blue-600
    }

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        "<style>",
        "  text { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', Helvetica, Arial, sans-serif; }",
        "  .title { font-size: 16px; font-weight: 700; fill: #0F172A; }",
        "  .subtitle { font-size: 13px; font-weight: 600; fill: #1E293B; }",
        "  .axis { font-size: 11px; fill: #475569; }",
        "  .val { font-size: 10px; font-weight: 600; }",
        "  .grid { stroke: #E2E8F0; stroke-width: 1; }",
        "  .frame { stroke: #94A3B8; stroke-width: 1.2; fill: none; }",
        "</style>",
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="#FFFFFF" stroke="#E2E8F0"/>',
        f'<text x="28" y="34" class="title">{title}</text>',
    ]

    # Legend at top-right
    lx = 690
    for idx, mode in enumerate(("wired", "nowired")):
        c = colors[mode]
        x0 = lx + idx * 120
        svg.append(f'<line x1="{x0}" y1="28" x2="{x0 + 20}" y2="28" stroke="{c}" stroke-width="3"/>')
        svg.append(f'<circle cx="{x0 + 10}" cy="28" r="4" fill="{c}"/>')
        svg.append(f'<text x="{x0 + 26}" y="32" class="axis" font-weight="600">{mode}</text>')

    def draw_panel(
        px: float,
        py: float,
        pw: float,
        ph: float,
        panel_title: str,
        y_key: str,
        y_unit: str,
        ymin: float,
        ymax: float,
        show_ideal_diag: bool = False,
    ) -> None:
        svg.append(f'<text x="{px}" y="{py - 12}" class="subtitle">{panel_title}</text>')
        # Horizontal gridlines (4 intervals)
        for i in range(5):
            yv = ymin + (ymax - ymin) * i / 4.0
            yy = _scale(yv, ymin, ymax, py + ph, py)
            svg.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + pw}" y2="{yy:.1f}" class="grid"/>')
            svg.append(
                f'<text x="{px - 8}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{yv:.1f}{y_unit}</text>'
            )

        # Vertical ticks
        for xv in xs:
            xx = _scale(xv, xmin, xmax, px, px + pw)
            svg.append(f'<line x1="{xx:.1f}" y1="{py}" x2="{xx:.1f}" y2="{py + ph}" class="grid"/>')
            svg.append(
                f'<text x="{xx:.1f}" y="{py + ph + 18}" text-anchor="middle" class="axis">{xv:g}G</text>'
            )

        svg.append(
            f'<text x="{px + pw / 2:.1f}" y="{py + ph + 38}" text-anchor="middle" class="axis">Target KV Pool Size (GiB)</text>'
        )

        # Optional 1:1 diagonal reference line (y = x)
        if show_ideal_diag:
            x1 = _scale(xmin, xmin, xmax, px, px + pw)
            y1 = _scale(xmin, ymin, ymax, py + ph, py)
            x2 = _scale(xmax, xmin, xmax, px, px + pw)
            y2 = _scale(xmax, ymin, ymax, py + ph, py)
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#94A3B8" stroke-dasharray="5,4" stroke-width="1.2"/>'
            )

        svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" class="frame"/>')

        # Data series
        for mode, pts in by_mode.items():
            c = colors.get(mode, "#0D9488")
            coords = [
                (
                    _scale(p["target_gib"], xmin, xmax, px, px + pw),
                    _scale(p[y_key], ymin, ymax, py + ph, py),
                    p[y_key],
                )
                for p in pts
            ]
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
            svg.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{poly}"/>')
            dy = -10 if mode == "wired" else 16
            for x, y, v in coords:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c}" stroke="#FFFFFF" stroke-width="1.5"/>')
                svg.append(
                    f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" class="val" fill="{c}">{v:.2f}</text>'
                )

    draw_panel(
        px=68,
        py=82,
        pw=380,
        ph=270,
        panel_title="1. Ghost Wired Lock-in After Release (without clear_cache)",
        y_key="ghost_wired_after_release_gib",
        y_unit="G",
        ymin=0.0,
        ymax=y1_max,
        show_ideal_diag=True,
    )
    draw_panel(
        px=540,
        py=82,
        pw=380,
        ph=270,
        panel_title="2. XNU Memory Compressor Delta (GiB)",
        y_key="compressed_delta_gib",
        y_unit="G",
        ymin=y2_min,
        ymax=y2_max,
        show_ideal_diag=False,
    )

    svg.append("</svg>")
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    out_svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return out_svg_path
