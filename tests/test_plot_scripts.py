import csv
import subprocess
import sys

import pytest


def test_plot_paper_results_script_generates_figures(tmp_path):
    pytest.importorskip("matplotlib")

    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()

    metrics = [
        "runtime_s",
        "mean_error_rad",
        "max_mean_error_rad",
        "l1_error",
        "max_l1_error",
        "min_evaluated_density",
        "max_normalization_error",
    ]
    with (results_dir / "smoothing_evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "parameter",
                "n_repetitions",
                *(f"{metric}_mean" for metric in metrics),
                *(f"{metric}_std" for metric in metrics),
            ],
        )
        writer.writeheader()
        for method, parameter, mean_error, l1_error, runtime in (
            ("FIGFAN", 9, 0.03, 0.02, 0.01),
            ("FIGFDN", 9, 0.02, 0.01, 0.01),
            ("PF", 50, 0.08, 0.10, 0.04),
            ("PWC", 9, 0.05, 0.08, 0.02),
        ):
            writer.writerow(
                {
                    "method": method,
                    "parameter": parameter,
                    "n_repetitions": 1,
                    "runtime_s_mean": runtime,
                    "mean_error_rad_mean": mean_error,
                    "max_mean_error_rad_mean": 2 * mean_error,
                    "l1_error_mean": l1_error,
                    "max_l1_error_mean": 2 * l1_error,
                    "min_evaluated_density_mean": 0.0,
                    "max_normalization_error_mean": 0.0,
                    "runtime_s_std": 0.0,
                    "mean_error_rad_std": 0.0,
                    "max_mean_error_rad_std": 0.0,
                    "l1_error_std": 0.0,
                    "max_l1_error_std": 0.0,
                    "min_evaluated_density_std": 0.0,
                    "max_normalization_error_std": 0.0,
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

    assert (figures_dir / "smoothing_mean_error_by_parameter.png").exists()
    assert (figures_dir / "smoothing_l1_error_by_parameter.png").exists()
    assert (figures_dir / "smoothing_runtime_by_parameter.png").exists()
    assert (figures_dir / "smoothing_mean_error_by_runtime.png").exists()
    assert (figures_dir / "smoothing_l1_error_by_runtime.png").exists()


def test_plot_smoothing_hero_script_generates_figure(tmp_path):
    pytest.importorskip("matplotlib")

    figures_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_smoothing_hero.py",
            "--figures-dir",
            str(figures_dir),
            "--formats",
            "png",
            "--grid-size",
            "31",
            "--time-steps",
            "4",
        ],
        check=True,
    )

    assert (figures_dir / "smoothing_space_time.png").exists()
