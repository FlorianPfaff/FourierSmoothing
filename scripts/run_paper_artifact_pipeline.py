#!/usr/bin/env python
"""Generate paper result CSVs, figures, and LaTeX tables in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fourier_smoothing import (
    run_identity_torus_benchmark,
    run_truncation_negativity_diagnostic,
    write_benchmark_csv,
    write_negativity_csv,
)
from fourier_smoothing.particle_experiments import run_particle_baseline_benchmark, write_particle_baseline_csv
from fourier_smoothing.tables import write_latex_tables


SMOKE_CONFIG = {
    "grid_sizes": [9, 15],
    "identity_repetitions": 1,
    "identity_time_steps": 3,
    "k_max_values": [1, 3],
    "sharpness_values": [5.0, 9.0],
    "evaluation_grid_size": 65,
    "negativity_time_steps": 3,
    "n_particles_values": [50],
    "n_trajectories": 20,
    "particle_repetitions": 1,
    "particle_grid_size": 41,
    "particle_time_steps": 3,
    "main_figf_grid_sizes": [9, 17],
    "main_pwc_grid_sizes": [9, 17],
    "main_pf_particle_counts": [50],
    "main_repetitions": 1,
    "main_time_steps": 3,
    "main_l1_reference_grid_size": 65,
    "main_mean_reference_particles": 500,
    "main_particle_chunk_size": 500,
    "main_pwc_quadrature_points": 4,
}

PAPER_CONFIG = {
    "grid_sizes": [15, 31, 63, 127, 255],
    "identity_repetitions": 10,
    "identity_time_steps": 4,
    "k_max_values": [1, 2, 3, 5, 8, 12, 16, 24],
    "sharpness_values": [2.0, 5.0, 9.0, 13.0],
    "evaluation_grid_size": 257,
    "negativity_time_steps": 4,
    "n_particles_values": [100, 300, 1000, 3000],
    "n_trajectories": 300,
    "particle_repetitions": 10,
    "particle_grid_size": 257,
    "particle_time_steps": 4,
    "main_figf_grid_sizes": [15, 31, 63, 127, 255, 511, 1023, 2047, 4095],
    "main_pwc_grid_sizes": [15, 31, 63, 127, 255, 511, 1023, 2047, 4095],
    "main_pf_particle_counts": [100, 300, 1000, 3000, 10000],
    "main_repetitions": 5,
    "main_time_steps": 5,
    "main_l1_reference_grid_size": 8193,
    "main_mean_reference_particles": 100_000,
    "main_particle_chunk_size": 100_000,
    "main_pwc_quadrature_points": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--output-root", type=Path, default=Path("generated_paper_artifacts"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--include-smoothing-evaluation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Generate the FIGFAN/FIGFDN/PWC/PF main evaluation. By default it is enabled for the smoke profile "
            "and disabled for the paper profile because paper runtimes should be measured on the designated server."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SMOKE_CONFIG if args.profile == "smoke" else PAPER_CONFIG
    include_smoothing_evaluation = (
        args.include_smoothing_evaluation
        if args.include_smoothing_evaluation is not None
        else args.profile == "smoke"
    )

    results_dir = args.results_dir or args.output_root / "results"
    figures_dir = args.figures_dir or args.output_root / "figures"
    tables_dir = args.tables_dir or args.output_root / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing results to {results_dir}")
    identity_rows = run_identity_torus_benchmark(
        config["grid_sizes"],
        repetitions=config["identity_repetitions"],
        time_steps=config["identity_time_steps"],
    )
    write_benchmark_csv(identity_rows, results_dir / "identity_torus_benchmark.csv")

    negativity_rows = run_truncation_negativity_diagnostic(
        config["k_max_values"],
        sharpness_values=config["sharpness_values"],
        evaluation_grid_size=config["evaluation_grid_size"],
        time_steps=config["negativity_time_steps"],
    )
    write_negativity_csv(negativity_rows, results_dir / "truncation_negativity_diagnostic.csv")

    particle_rows = run_particle_baseline_benchmark(
        config["n_particles_values"],
        n_trajectories=config["n_trajectories"],
        repetitions=config["particle_repetitions"],
        grid_size=config["particle_grid_size"],
        time_steps=config["particle_time_steps"],
    )
    write_particle_baseline_csv(particle_rows, results_dir / "particle_smoother_baseline.csv")

    if include_smoothing_evaluation:
        print("Generating main FIGFAN/FIGFDN/PWC/PF smoothing evaluation")
        _run_main_smoothing_evaluation(config, results_dir)

    if not args.skip_plots:
        print(f"Writing figures to {figures_dir}")
        subprocess.run(
            [
                sys.executable,
                "scripts/plot_paper_results.py",
                "--results-dir",
                str(results_dir),
                "--figures-dir",
                str(figures_dir),
            ],
            check=True,
        )

    print(f"Writing tables to {tables_dir}")
    table_paths = write_latex_tables(results_dir, tables_dir)
    for path in table_paths:
        print(path)


def _run_main_smoothing_evaluation(config: dict[str, object], results_dir: Path) -> None:
    command = [
        sys.executable,
        "scripts/run_smoothing_evaluation.py",
        "--output-dir",
        str(results_dir),
        "--figf-grid-sizes",
        *[str(value) for value in config["main_figf_grid_sizes"]],
        "--pwc-grid-sizes",
        *[str(value) for value in config["main_pwc_grid_sizes"]],
        "--pf-particle-counts",
        *[str(value) for value in config["main_pf_particle_counts"]],
        "--repetitions",
        str(config["main_repetitions"]),
        "--time-steps",
        str(config["main_time_steps"]),
        "--l1-reference-grid-size",
        str(config["main_l1_reference_grid_size"]),
        "--mean-reference-particles",
        str(config["main_mean_reference_particles"]),
        "--particle-chunk-size",
        str(config["main_particle_chunk_size"]),
        "--pwc-quadrature-points",
        str(config["main_pwc_quadrature_points"]),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
