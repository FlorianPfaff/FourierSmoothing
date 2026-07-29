import csv
import subprocess
import sys
from pathlib import Path

import pytest


def test_runtime_accuracy_column_plot_uses_runtime_iqr_and_writes_pgf_data(tmp_path):
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
                "runtime_s_median",
                "runtime_s_q25",
                "runtime_s_q75",
                "mean_error_rad_mean",
                "mean_error_rad_q25",
                "mean_error_rad_q75",
                "l1_error_mean",
                "l1_error_q25",
                "l1_error_q75",
            ],
        )
        writer.writeheader()
        for method, parameter, mean_runtime, median_runtime, q25, q75, mean_error, l1_error in (
            ("FIGFAN", 9, 0.0010, 0.0009, 0.0008, 0.0011, 0.03, 0.02),
            ("FIGFAN", 17, 0.0020, 0.0018, 0.0016, 0.0022, 0.02, 0.01),
            ("FIGFDN", 9, 0.0010, 0.0009, 0.0008, 0.0011, 0.025, 0.015),
            ("FIGFDN", 17, 0.0020, 0.0018, 0.0016, 0.0022, 0.02, 0.01),
            ("PF", 50, 0.0100, 0.0080, 0.0060, 0.0120, 0.08, 0.10),
            ("PF", 100, 0.0200, 0.0160, 0.0130, 0.0240, 0.06, 0.08),
            ("PWC", 9, 0.0008, 0.0007, 0.0006, 0.0009, 0.05, 0.08),
            ("PWC", 17, 0.0015, 0.0013, 0.0011, 0.0016, 0.04, 0.05),
        ):
            writer.writerow(
                {
                    "method": method,
                    "parameter": parameter,
                    "runtime_s_mean": mean_runtime,
                    "runtime_s_median": median_runtime,
                    "runtime_s_q25": q25,
                    "runtime_s_q75": q75,
                    "mean_error_rad_mean": mean_error,
                    "mean_error_rad_q25": 0.9 * mean_error,
                    "mean_error_rad_q75": 1.1 * mean_error,
                    "l1_error_mean": l1_error,
                    "l1_error_q25": 0.9 * l1_error,
                    "l1_error_q75": 1.1 * l1_error,
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
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    pdf_path = figures_dir / "smoothing_runtime_accuracy_column.pdf"
    assert pdf_path.exists()
    assert b"/Subtype /Type3" not in pdf_path.read_bytes()
    assert (figures_dir / "smoothing_runtime_accuracy_column.png").exists()

    data_dir = figures_dir / "data"
    expected_header = (
        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error mean_error_q25 "
        "mean_error_q75 mean_error_err_low mean_error_err_high l1_error l1_error_q25 "
        "l1_error_q75 l1_error_err_low l1_error_err_high\n"
    )
    expected_files = {
        "figfan": (
            "9 1 0.9 0.8 1.1 0.1 0.2 0.03 0.027 0.033 0.003 0.003 0.02 0.018 0.022 0.002 0.002\n"
            "17 2 1.8 1.6 2.2 0.2 0.4 0.02 0.018 0.022 0.002 0.002 0.01 0.009 0.011 0.001 0.001\n"
        ),
        "figfdn": (
            "9 1 0.9 0.8 1.1 0.1 0.2 0.025 0.0225 0.0275 0.0025 0.0025 0.015 0.0135 0.0165 0.0015 0.0015\n"
            "17 2 1.8 1.6 2.2 0.2 0.4 0.02 0.018 0.022 0.002 0.002 0.01 0.009 0.011 0.001 0.001\n"
        ),
        "pf": (
            "50 10 8 6 12 2 4 0.08 0.072 0.088 0.008 0.008 0.1 0.09 0.11 0.01 0.01\n"
            "100 20 16 13 24 3 8 0.06 0.054 0.066 0.006 0.006 0.08 0.072 0.088 0.008 0.008\n"
        ),
        "pwc": (
            "9 0.8 0.7 0.6 0.9 0.1 0.2 0.05 0.045 0.055 0.005 0.005 0.08 0.072 0.088 0.008 0.008\n"
            "17 1.5 1.3 1.1 1.6 0.2 0.3 0.04 0.036 0.044 0.004 0.004 0.05 0.045 0.055 0.005 0.005\n"
        ),
    }
    for slug, rows in expected_files.items():
        content = (data_dir / f"smoothing_evaluation_{slug}.dat").read_text(encoding="utf-8")
        assert content == expected_header + rows
