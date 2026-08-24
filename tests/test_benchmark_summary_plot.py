import subprocess
import sys
from pathlib import Path

import pytest


def test_benchmark_summary_plot_is_generated_without_type3_fonts(tmp_path):
    pytest.importorskip("matplotlib")

    root = Path(__file__).resolve().parents[1]
    figures_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_benchmark_summary.py",
            "--results-dir",
            "paper_evidence",
            "--figures-dir",
            str(figures_dir),
            "--formats",
            "pdf",
            "png",
        ],
        cwd=root,
        check=True,
    )

    pdf_path = figures_dir / "smoothing_benchmark_summary.pdf"
    assert pdf_path.exists()
    assert b"/Subtype /Type3" not in pdf_path.read_bytes()
    assert (figures_dir / "smoothing_benchmark_summary.png").exists()
