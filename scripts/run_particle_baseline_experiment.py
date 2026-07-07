#!/usr/bin/env python
"""Generate CSV results for the one-dimensional torus particle-smoother baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fourier_smoothing.particle_experiments import run_particle_baseline_benchmark, write_particle_baseline_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated CSV files. Use the paper repo's results/ directory for paper artifacts.",
    )
    parser.add_argument("--n-particles-values", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=257)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--noise-concentration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--filename", default="particle_smoother_baseline.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_particle_baseline_benchmark(
        args.n_particles_values,
        n_trajectories=args.n_trajectories,
        repetitions=args.repetitions,
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        noise_concentration=args.noise_concentration,
        seed=args.seed,
    )
    output_path = write_particle_baseline_csv(rows, args.output_dir / args.filename)
    print(output_path)


if __name__ == "__main__":
    main()
