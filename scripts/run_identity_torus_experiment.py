#!/usr/bin/env python
"""Generate CSV results for the one-dimensional identity-torus benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from fourier_smoothing import run_identity_torus_benchmark, write_benchmark_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated CSV files. Use the paper repo's results/ directory for paper artifacts.",
    )
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[15, 31, 63, 127])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--noise-concentration", type=float, default=3.0)
    parser.add_argument("--filename", default="identity_torus_benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_identity_torus_benchmark(
        args.grid_sizes,
        repetitions=args.repetitions,
        time_steps=args.time_steps,
        noise_concentration=args.noise_concentration,
    )
    output_path = write_benchmark_csv(rows, args.output_dir / args.filename)
    print(output_path)


if __name__ == "__main__":
    main()
