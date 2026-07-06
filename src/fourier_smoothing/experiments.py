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
    truncate_fourier_coefficients,
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


@dataclass(frozen=True)
class NegativityDiagnosticRow:
    """One CSV-ready row for truncated Fourier negativity diagnostics."""

    k_max: int
    n_coefficients: int
    sharpness: float
    time_steps: int
    min_value: float
    negative_mass: float
    max_negative_undershoot: float
    max_normalization_error: float
    l1_error_to_dense_grid: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "k_max": self.k_max,
            "n_coefficients": self.n_coefficients,
            "sharpness": self.sharpness,
            "time_steps": self.time_steps,
            "min_value": self.min_value,
            "negative_mass": self.negative_mass,
            "max_negative_undershoot": self.max_negative_undershoot,
            "max_normalization_error": self.max_normalization_error,
            "l1_error_to_dense_grid": self.l1_error_to_dense_grid,
        }


BENCHMARK_CSV_COLUMNS = [
    "method",
    "grid_size",
    "repetition",
    "runtime_s",
    "max_abs_difference_to_grid",
    "max_normalization_error",
]

NEGATIVITY_CSV_COLUMNS = [
    "k_max",
    "n_coefficients",
    "sharpness",
    "time_steps",
    "min_value",
    "negative_mass",
    "max_negative_undershoot",
    "max_normalization_error",
    "l1_error_to_dense_grid",
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


def make_sharp_multimodal_likelihoods(
    grid_shape: Sequence[int],
    time_steps: int,
    *,
    sharpness: float,
) -> NDArray[np.float64]:
    """Create sharp positive 1-D likelihoods that stress low-order truncations."""

    shape = tuple(int(n) for n in grid_shape)
    if len(shape) != 1:
        raise ValueError("make_sharp_multimodal_likelihoods currently supports only 1-D grids.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    (x,) = torus_grid(shape)
    cell_volume = cell_volume_for_grid(shape)
    likelihoods = []
    for t in range(time_steps):
        phase = 0.43 * (t + 1)
        values = (
            0.04
            + np.exp(sharpness * np.cos(x - phase))
            + 0.65 * np.exp(0.85 * sharpness * np.cos(x - phase - 2.35))
            + 0.35 * np.exp(0.55 * sharpness * np.cos(2.0 * x + phase))
        )
        likelihoods.append(normalize_grid_density(values, cell_volume))
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


def run_truncation_negativity_diagnostic(
    k_max_values: Iterable[int] = (1, 2, 3, 5, 8, 12),
    *,
    sharpness_values: Iterable[float] = (2.0, 5.0, 9.0),
    evaluation_grid_size: int = 257,
    time_steps: int = 4,
    noise_concentration: float = 4.0,
) -> list[NegativityDiagnosticRow]:
    """Quantify negative undershoot caused by truncated identity coefficients.

    A dense grid smoother is used as a reference. The Fourier smoother receives
    truncated coefficients and is evaluated back on the dense grid. Each row
    reports worst-case values over all smoothed time steps for one ``k_max`` and
    likelihood sharpness.
    """

    if evaluation_grid_size <= 0:
        raise ValueError("evaluation_grid_size must be positive.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")

    grid_shape = (int(evaluation_grid_size),)
    cell_volume = cell_volume_for_grid(grid_shape)
    rows: list[NegativityDiagnosticRow] = []

    for sharpness in sharpness_values:
        likelihoods = make_sharp_multimodal_likelihoods(grid_shape, time_steps, sharpness=float(sharpness))
        filtered = filtered_from_likelihoods(likelihoods, cell_volume)
        noise = make_von_mises_like_noise(grid_shape, noise_concentration)
        transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
        dense_reference = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)

        dense_likelihood_coeffs = [grid_to_fourier(values) for values in likelihoods]
        dense_filtered_coeffs = [grid_to_fourier(values) for values in filtered]
        dense_noise_coeffs = grid_to_fourier(noise)

        for k_max in k_max_values:
            k_max = int(k_max)
            if k_max < 0:
                raise ValueError("k_max values must be nonnegative.")
            coeff_shape = (2 * k_max + 1,)
            likelihood_coeffs = np.stack(
                [truncate_fourier_coefficients(coeffs, coeff_shape) for coeffs in dense_likelihood_coeffs],
                axis=0,
            )
            filtered_coeffs = np.stack(
                [truncate_fourier_coefficients(coeffs, coeff_shape) for coeffs in dense_filtered_coeffs],
                axis=0,
            )
            noise_coeffs = truncate_fourier_coefficients(dense_noise_coeffs, coeff_shape)
            result = fourier_identity_smoother(
                filtered_coeffs,
                likelihood_coeffs,
                noise_coeffs,
                multiplication="truncated_convolution",
            )
            dense_values = np.stack(
                [fourier_to_grid(coeffs, grid_shape=grid_shape) for coeffs in result.smoothed_coefficients],
                axis=0,
            )
            negative_part = np.maximum(-dense_values, 0.0)
            rows.append(
                NegativityDiagnosticRow(
                    k_max=k_max,
                    n_coefficients=coeff_shape[0],
                    sharpness=float(sharpness),
                    time_steps=time_steps,
                    min_value=float(np.min(dense_values)),
                    negative_mass=float(np.max(np.sum(negative_part, axis=1) * cell_volume)),
                    max_negative_undershoot=float(np.max(negative_part)),
                    max_normalization_error=_max_grid_normalization_error(dense_values, cell_volume),
                    l1_error_to_dense_grid=float(
                        np.max(np.sum(np.abs(dense_values - dense_reference.smoothed), axis=1) * cell_volume)
                    ),
                )
            )
    return rows


def write_benchmark_csv(rows: Sequence[BenchmarkRow], output_path: str | Path) -> Path:
    """Write benchmark rows to ``output_path`` and return the path."""

    return _write_csv(rows, output_path, BENCHMARK_CSV_COLUMNS)


def write_negativity_csv(rows: Sequence[NegativityDiagnosticRow], output_path: str | Path) -> Path:
    """Write truncation-negativity diagnostic rows to ``output_path``."""

    return _write_csv(rows, output_path, NEGATIVITY_CSV_COLUMNS)


def _write_csv(rows, output_path: str | Path, fieldnames: Sequence[str]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def _max_grid_normalization_error(values: NDArray[np.float64], cell_volume: float) -> float:
    integrals = np.sum(values, axis=tuple(range(1, values.ndim))) * cell_volume
    return float(np.max(np.abs(integrals - 1.0)))
