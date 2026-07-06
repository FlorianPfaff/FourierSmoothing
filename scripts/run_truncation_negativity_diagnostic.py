#!/usr/bin/env python
"""Generate CSV diagnostics for truncated identity-Fourier negativity."""

from __future__ import annotations

import argparse
from pathlib import Path

from fourier_smoothing import run_truncation_negativity_diagnostic, write_negativity_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated CSV files. Use the paper repo's results/ directory for paper artifacts.",
    )
    parser.add_argument("--k-max-values", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 16])
    parser.add_argument("--sharpness-values", type=float, nargs="+", default=[2.0, 5.0, 9.0, 13.0])
    parser.add_argument("--evaluation-grid-size", type=int, default=257)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--noise-concentration", type=float, default=4.0)
    parser.add_argument("--filename", default="truncation_negativity_diagnostic.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_truncation_negativity_diagnostic(
        args.k_max_values,
        sharpness_values=args.sharpness_values,
        evaluation_grid_size=args.evaluation_grid_size,
        time_steps=args.time_steps,
        noise_concentration=args.noise_concentration,
    )
    output_path = write_negativity_csv(rows, args.output_dir / args.filename)
    print(output_path)


if __name__ == "__main__":
    main()
