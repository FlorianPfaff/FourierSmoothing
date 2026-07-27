import csv
import subprocess
import sys

import pytest


def test_runtime_accuracy_column_plot_uses_connected_curves(tmp_path):
    pytest.importorskip("matplotlib")

    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()

    with (results_dir / "smoothing_evaluation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "parameter",
                "runtime_s_mean",
                "mean_error_rad_mean",
                "l1_error_mean",
            ],
        )
        writer.writeheader()
        for method, parameter, runtime, mean_error, l1_error in (
            ("FIGFAN", 9, 0.001, 0.03, 0.02),
            ("FIGFAN", 17, 0.002, 0.02, 0.01),
            ("FIGFDN", 9, 0.001, 0.025, 0.015),
            ("FIGFDN", 17, 0.002, 0.02, 0.01),
            ("PF", 50, 0.01, 0.08, 0.10),
            ("PF", 100, 0.02, 0.06, 0.08),
            ("PWC", 9, 0.0008, 0.05, 0.08),
            ("PWC", 17, 0.0015, 0.04, 0.05),
        ):
            writer.writerow(
                {
                    "method": method,
                    "parameter": parameter,
                    "runtime_s_mean": runtime,
                    "mean_error_rad_mean": mean_error,
                    "l1_error_mean": l1_error,
                }
            )

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_runtime_accuracy_column.py",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
            "--formats",
            "pdf",
            "png",
        ],
        check=True,
    )

    pdf_path = figures_dir / "smoothing_runtime_accuracy_column.pdf"
    assert pdf_path.exists()
    assert b"/Subtype /Type3" not in pdf_path.read_bytes()
    assert (figures_dir / "smoothing_runtime_accuracy_column.png").exists()
