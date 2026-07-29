#!/usr/bin/env python3
"""Validate weighted adjoint identities for additive and dense transitions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fourier_smoothing import (
    DenseGridTransition,
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    normalize_grid_density,
    torus_additive_transition_density_matrix,
    torus_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def weighted_inner(left, right, cell_volume):
    return float(np.sum(left * right) * cell_volume)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    grid_shape = (7, 9)
    x, y = torus_grid(grid_shape)
    volume = cell_volume_for_grid(grid_shape)
    noise = normalize_grid_density(np.exp(1.1 * np.cos(x) + 0.7 * np.cos(y)), volume)
    density = normalize_grid_density(0.5 + rng.random(grid_shape), volume)
    message = 0.5 + rng.random(grid_shape)
    fft_adjoint = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
    matrix = torus_additive_transition_density_matrix(noise, cell_volume=volume)
    dense = DenseGridTransition.for_grid_shape(matrix, grid_shape, cell_volume=volume)
    forward = dense.forward_predict(density, 0)
    rows.append(
        {
            "case": "2d_additive_weighted_adjoint",
            "absolute_error": abs(weighted_inner(forward, message, volume) - weighted_inner(density, fft_adjoint(message, 0), volume)),
        }
    )
    rows.append(
        {
            "case": "2d_additive_fft_vs_dense_adjoint",
            "absolute_error": float(np.max(np.abs(fft_adjoint(message, 0) - dense(message, 0)))),
        }
    )

    grid_shape = (13,)
    volume = cell_volume_for_grid(grid_shape)
    raw_matrix = 0.1 + rng.random((13, 13))
    transition = DenseGridTransition.for_grid_shape(raw_matrix, grid_shape, cell_volume=volume)
    density = normalize_grid_density(0.5 + rng.random(grid_shape), volume)
    message = 0.5 + rng.random(grid_shape)
    rows.append(
        {
            "case": "1d_nonadditive_weighted_adjoint",
            "absolute_error": abs(
                weighted_inner(transition.forward_predict(density, 0), message, volume)
                - weighted_inner(density, transition(message, 0), volume)
            ),
        }
    )

    output = args.output_dir / "adjoint_validation.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "absolute_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
