#!/usr/bin/env python
"""Generate CSV results for the FIGFAN/FIGFDN/PWC smoothing benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from fourier_smoothing import run_figf_pwc_benchmark, write_figf_pwc_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated CSV files. Use the paper repo's results/ directory for paper artifacts.",
    )
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[15, 31, 63, 127, 255])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--reference-grid-size", type=int, default=2049)
    parser.add_argument("--likelihood-sharpness", type=float, default=5.0)
    parser.add_argument("--noise-concentration", type=float, default=4.0)
    parser.add_argument("--pwc-quadrature-points", type=int, default=8)
    parser.add_argument("--filename", default="figf_pwc_benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_figf_pwc_benchmark(
        args.grid_sizes,
        repetitions=args.repetitions,
        time_steps=args.time_steps,
        reference_grid_size=args.reference_grid_size,
        likelihood_sharpness=args.likelihood_sharpness,
        noise_concentration=args.noise_concentration,
        pwc_quadrature_points=args.pwc_quadrature_points,
    )
    output_path = write_figf_pwc_csv(rows, args.output_dir / args.filename)
    print(output_path)


if __name__ == "__main__":
    main()
