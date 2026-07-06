"""Small reproducible experiments for the Fourier smoothing paper.

The routines in this module are intentionally lightweight and write plain CSV
files. Code stays in this repository; generated CSV files can be written to the
paper repository's ``results/`` directory.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .smoother import (
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    fourier_identity_smoother,
    fourier_to_grid,
    grid_backward_information_smoother,
    grid_to_fourier,
    normalize_grid_density,
    torus_grid,
)


@dataclass(frozen=True)
class BenchmarkRow:
    """One CSV-ready benchmark row."""

    method: str
    grid_size: int
    repetition: int
    runtime_s: float
    max_abs_difference_to_grid: float
    max_normalization_error: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "grid_size": self.grid_size,
            "repetition": self.repetition,
            "runtime_s": self.runtime_s,
            "max_abs_difference_to_grid": self.max_abs_difference_to_grid,
            "max_normalization_error": self.max_normalization_error,
        }


CSV_COLUMNS = [
    "method",
    "grid_size",
    "repetition",
    "runtime_s",
    "max_abs_difference_to_grid",
    "max_normalization_error",
]


def make_identity_likelihoods(grid_shape: Sequence[int], time_steps: int) -> NDArray[np.float64]:
    """Create deterministic positive likelihood functions on an equidistant torus grid."""

    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    axes = torus_grid(grid_shape)
    combined = np.zeros(grid_shape, dtype=float)
    likelihoods = []
    for t in range(time_steps):
        values = np.ones(grid_shape, dtype=float)
        for dim, axis in enumerate(axes):
            phase = 0.37 * (t + 1) * (dim + 1)
            values *= 1.15 + 0.22 * np.cos(axis - phase)
            combined = combined + 0.07 * np.sin((dim + 1) * axis + phase)
        values = values + combined
        likelihoods.append(np.maximum(values, 1.0e-12))
    return np.stack(likelihoods, axis=0)


def filtered_from_likelihoods(likelihoods: NDArray[np.float64], cell_volume: float) -> NDArray[np.float64]:
    """Construct a simple forward-filtered sequence for an identity-transition test case."""

    filtered = []
    cumulative = np.ones_like(likelihoods[0], dtype=float)
    for likelihood in likelihoods:
        cumulative = cumulative * likelihood
        filtered.append(normalize_grid_density(cumulative, cell_volume))
    return np.stack(filtered, axis=0)


def make_von_mises_like_noise(grid_shape: Sequence[int], concentration: float) -> NDArray[np.float64]:
    """Return a separable wrapped von-Mises-like density on ``T^d``."""

    axes = torus_grid(grid_shape)
    values = np.ones(tuple(grid_shape), dtype=float)
    for axis in axes:
        values *= np.exp(concentration * np.cos(axis))
    return normalize_grid_density(values, cell_volume_for_grid(grid_shape))


def run_identity_torus_benchmark(
    grid_sizes: Iterable[int] = (15, 31, 63),
    *,
    repetitions: int = 3,
    time_steps: int = 4,
    noise_concentration: float = 3.0,
    fourier_multiplication: str = "truncated_convolution",
) -> list[BenchmarkRow]:
    """Benchmark grid and Fourier smoothers for a 1-D additive identity model.

    The grid row is a discretized reference on the same equidistant grid. The
    Fourier row uses ``fourier_multiplication``; the default is the aliasing-free
    coefficient convolution followed by truncation. Use ``"grid"`` to reproduce
    the grid-transform multiplication exactly.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")

    rows: list[BenchmarkRow] = []
    for grid_size in grid_sizes:
        if int(grid_size) <= 0:
            raise ValueError("grid sizes must be positive.")
        grid_shape = (int(grid_size),)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_identity_likelihoods(grid_shape, time_steps)
        filtered = filtered_from_likelihoods(likelihoods, cell_volume)
        noise = make_von_mises_like_noise(grid_shape, noise_concentration)
        transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
        likelihood_coeffs = np.stack([grid_to_fourier(values) for values in likelihoods], axis=0)
        filtered_coeffs = np.stack([grid_to_fourier(values) for values in filtered], axis=0)
        noise_coeffs = grid_to_fourier(noise)

        for repetition in range(repetitions):
            start = time.perf_counter()
            grid_result = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)
            grid_runtime = time.perf_counter() - start
            grid_norm_error = _max_grid_normalization_error(grid_result.smoothed, cell_volume)
            rows.append(
                BenchmarkRow(
                    method="grid",
                    grid_size=grid_size,
                    repetition=repetition,
                    runtime_s=grid_runtime,
                    max_abs_difference_to_grid=0.0,
                    max_normalization_error=grid_norm_error,
                )
            )

            start = time.perf_counter()
            fourier_result = fourier_identity_smoother(
                filtered_coeffs,
                likelihood_coeffs,
                noise_coeffs,
                multiplication=fourier_multiplication,
            )
            fourier_runtime = time.perf_counter() - start
            fourier_grid = np.stack([fourier_to_grid(coeffs) for coeffs in fourier_result.smoothed_coefficients], axis=0)
            fourier_norm_error = _max_grid_normalization_error(fourier_grid, cell_volume)
            rows.append(
                BenchmarkRow(
                    method=f"fourier_identity_{fourier_multiplication}",
                    grid_size=grid_size,
                    repetition=repetition,
                    runtime_s=fourier_runtime,
                    max_abs_difference_to_grid=float(np.max(np.abs(fourier_grid - grid_result.smoothed))),
                    max_normalization_error=fourier_norm_error,
                )
            )
    return rows


def write_benchmark_csv(rows: Sequence[BenchmarkRow], output_path: str | Path) -> Path:
    """Write benchmark rows to ``output_path`` and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def _max_grid_normalization_error(values: NDArray[np.float64], cell_volume: float) -> float:
    integrals = np.sum(values, axis=tuple(range(1, values.ndim))) * cell_volume
    return float(np.max(np.abs(integrals - 1.0)))
