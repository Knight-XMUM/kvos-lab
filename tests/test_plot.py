"""Tests for kvos.plot SVG generation from bench CSV output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kvos.cli import main as cli_main
from kvos.plot import render_bench_svg

ROOT = Path(__file__).resolve().parent.parent


class TestKvosPlot(unittest.TestCase):
    def test_render_svg_from_existing_m4_air_csv(self) -> None:
        csv_path = ROOT / "results" / "bench-m4-air.csv"
        self.assertTrue(csv_path.exists())
        with tempfile.TemporaryDirectory() as tmp:
            out_svg = Path(tmp) / "chart.svg"
            render_bench_svg(csv_path, out_svg)
            content = out_svg.read_text(encoding="utf-8")
            self.assertIn("<svg", content)
            self.assertIn("</svg>", content)
            self.assertIn("Ghost Wired Lock-in", content)
            self.assertIn("wired", content)
            self.assertIn("nowired", content)

    def test_cli_plot_command(self) -> None:
        csv_path = ROOT / "results" / "bench-m4-air.csv"
        with tempfile.TemporaryDirectory() as tmp:
            out_svg = Path(tmp) / "cli_chart.svg"
            rc = cli_main(["plot", "--input", str(csv_path), "--out", str(out_svg)])
            self.assertEqual(rc, 0)
            self.assertTrue(out_svg.exists())


if __name__ == "__main__":
    unittest.main()
