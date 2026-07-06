import csv
import subprocess
import sys

import pytest


def test_plot_paper_results_script_generates_figures(tmp_path):
    pytest.importorskip("matplotlib")

    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()

    with (results_dir / "identity_torus_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "grid_size",
                "repetition",
                "runtime_s",
                "max_abs_difference_to_grid",
                "max_normalization_error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "grid",
                "grid_size": 9,
                "repetition": 0,
                "runtime_s": 0.01,
                "max_abs_difference_to_grid": 0.0,
                "max_normalization_error": 0.0,
            }
        )
        writer.writerow(
            {
                "method": "fourier_identity_grid",
                "grid_size": 9,
                "repetition": 0,
                "runtime_s": 0.02,
                "max_abs_difference_to_grid": 1e-12,
                "max_normalization_error": 0.0,
            }
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_paper_results.py",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
            "--formats",
            "png",
        ],
        check=True,
    )

    assert (figures_dir / "identity_torus_runtime.png").exists()
    assert (figures_dir / "identity_torus_difference.png").exists()
