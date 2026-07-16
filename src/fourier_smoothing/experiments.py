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

from .particle import (
    bootstrap_von_mises_particle_filter_1d,
    ffbsi_von_mises_particle_smoother_1d,
)
from .smoother import (
    DenseGridTransition,
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


@dataclass(frozen=True)
class FIGFPWCBenchmarkRow:
    """One CSV-ready row for the FIGF/PWC smoothing comparison."""

    method: str
    grid_size: int
    repetition: int
    filter_runtime_s: float
    smoother_runtime_s: float
    evaluation_runtime_s: float
    runtime_s: float
    mean_l1_error_to_reference: float
    max_l1_error_to_reference: float
    min_evaluated_density: float
    max_normalization_error: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "grid_size": self.grid_size,
            "repetition": self.repetition,
            "filter_runtime_s": self.filter_runtime_s,
            "smoother_runtime_s": self.smoother_runtime_s,
            "evaluation_runtime_s": self.evaluation_runtime_s,
            "runtime_s": self.runtime_s,
            "mean_l1_error_to_reference": self.mean_l1_error_to_reference,
            "max_l1_error_to_reference": self.max_l1_error_to_reference,
            "min_evaluated_density": self.min_evaluated_density,
            "max_normalization_error": self.max_normalization_error,
        }


@dataclass(frozen=True)
class SmoothingEvaluationRow:
    """One raw evaluation row for the smoother comparison."""

    method: str
    parameter: int
    repetition: int
    runtime_s: float
    mean_error_rad: float
    max_mean_error_rad: float
    l1_error: float
    max_l1_error: float
    min_evaluated_density: float
    max_normalization_error: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "parameter": self.parameter,
            "repetition": self.repetition,
            "runtime_s": self.runtime_s,
            "mean_error_rad": self.mean_error_rad,
            "max_mean_error_rad": self.max_mean_error_rad,
            "l1_error": self.l1_error,
            "max_l1_error": self.max_l1_error,
            "min_evaluated_density": self.min_evaluated_density,
            "max_normalization_error": self.max_normalization_error,
        }


@dataclass(frozen=True)
class SmoothingRuntimeRow:
    """One timing-only row for the smoother comparison."""

    method: str
    parameter: int
    repetition: int
    runtime_s: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "parameter": self.parameter,
            "repetition": self.repetition,
            "runtime_s": self.runtime_s,
        }


@dataclass(frozen=True)
class SmoothingGainRow:
    """One state-truth error pair from the smoothing-gain simulation."""

    trial: int
    time_step: int
    filter_error_rad: float
    smoother_error_rad: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "trial": self.trial,
            "time_step": self.time_step,
            "filter_error_rad": self.filter_error_rad,
            "smoother_error_rad": self.smoother_error_rad,
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

FIGF_PWC_CSV_COLUMNS = [
    "method",
    "grid_size",
    "repetition",
    "filter_runtime_s",
    "smoother_runtime_s",
    "evaluation_runtime_s",
    "runtime_s",
    "mean_l1_error_to_reference",
    "max_l1_error_to_reference",
    "min_evaluated_density",
    "max_normalization_error",
]

SMOOTHING_EVALUATION_CSV_COLUMNS = [
    "method",
    "parameter",
    "repetition",
    "runtime_s",
    "mean_error_rad",
    "max_mean_error_rad",
    "l1_error",
    "max_l1_error",
    "min_evaluated_density",
    "max_normalization_error",
]

SMOOTHING_GAIN_CSV_COLUMNS = [
    "trial",
    "time_step",
    "filter_error_rad",
    "smoother_error_rad",
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
    grid_offset: float = 0.0,
) -> NDArray[np.float64]:
    """Create sharp positive 1-D likelihoods that stress low-order truncations."""

    shape = tuple(int(n) for n in grid_shape)
    if len(shape) != 1:
        raise ValueError("make_sharp_multimodal_likelihoods currently supports only 1-D grids.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")
    if not np.isfinite(grid_offset):
        raise ValueError("grid_offset must be finite.")

    x = 2.0 * np.pi * (np.arange(shape[0], dtype=float) + grid_offset) / shape[0]
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
    repetitions: int = 5,
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


def run_figf_pwc_benchmark(
    grid_sizes: Iterable[int] = (15, 31, 63, 127, 255),
    *,
    repetitions: int = 5,
    time_steps: int = 5,
    reference_grid_size: int = 2049,
    likelihood_sharpness: float = 5.0,
    noise_concentration: float = 4.0,
    pwc_quadrature_points: int = 8,
) -> list[FIGFPWCBenchmarkRow]:
    """Compare FIGFAN, FIGFDN, and PWC smoothed-density reconstructions.

    A fine-grid FIGF smoother provides the numerical reference. For each coarse
    grid size, the FIGF recursion is run once and evaluated through the AN and
    DN interpolation layers. The PWC baseline uses a dense cell-averaged
    transition matrix, i.e. the classical histogram/HMM forward-backward
    smoother.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    if reference_grid_size <= 0:
        raise ValueError("reference_grid_size must be positive.")
    if pwc_quadrature_points < 1:
        raise ValueError("pwc_quadrature_points must be at least one.")

    reference_shape = (int(reference_grid_size),)
    reference_cell_volume = cell_volume_for_grid(reference_shape)
    reference_likelihoods = make_sharp_multimodal_likelihoods(
        reference_shape,
        time_steps,
        sharpness=likelihood_sharpness,
    )
    reference_noise = make_von_mises_like_noise(reference_shape, noise_concentration)
    reference_filtered = _run_figf_forward_filter(reference_likelihoods, reference_noise, reference_cell_volume)
    reference_transition = TorusAdditiveGridTransition.for_grid_shape(reference_noise, reference_shape)
    reference_smoothed = grid_backward_information_smoother(
        reference_filtered,
        reference_likelihoods,
        reference_transition,
        cell_volume=reference_cell_volume,
    ).smoothed

    rows: list[FIGFPWCBenchmarkRow] = []
    for grid_size in grid_sizes:
        grid_size = int(grid_size)
        if grid_size <= 0:
            raise ValueError("grid sizes must be positive.")
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_sharp_multimodal_likelihoods(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
        )
        figf_noise = make_von_mises_like_noise(grid_shape, noise_concentration)
        figf_transition = TorusAdditiveGridTransition.for_grid_shape(figf_noise, grid_shape)

        pwc_transition_matrix = make_pwc_additive_transition_density_matrix_1d(
            grid_size,
            noise_concentration,
            quadrature_points=pwc_quadrature_points,
        )
        pwc_transition = DenseGridTransition.for_grid_shape(
            pwc_transition_matrix,
            grid_shape,
            cell_volume=cell_volume,
        )

        for repetition in range(repetitions):
            figf_filter_start = time.perf_counter()
            figf_filtered = _run_figf_forward_filter(likelihoods, figf_noise, cell_volume)
            figf_filter_runtime = time.perf_counter() - figf_filter_start

            figf_smoother_start = time.perf_counter()
            figf_result = grid_backward_information_smoother(
                figf_filtered,
                likelihoods,
                figf_transition,
                cell_volume=cell_volume,
            )
            figf_smoother_runtime = time.perf_counter() - figf_smoother_start

            for method, evaluator in (
                ("FIGFAN", _evaluate_figfan_1d),
                ("FIGFDN", _evaluate_figfdn_1d),
            ):
                evaluation_start = time.perf_counter()
                evaluated = evaluator(figf_result.smoothed, reference_grid_size)
                evaluation_runtime = time.perf_counter() - evaluation_start
                rows.append(
                    _make_figf_pwc_row(
                        method=method,
                        grid_size=grid_size,
                        repetition=repetition,
                        filter_runtime_s=figf_filter_runtime,
                        smoother_runtime_s=figf_smoother_runtime,
                        evaluation_runtime_s=evaluation_runtime,
                        evaluated=evaluated,
                        reference=reference_smoothed,
                        reference_cell_volume=reference_cell_volume,
                    )
                )

            pwc_filter_start = time.perf_counter()
            pwc_filtered = _run_dense_transition_forward_filter(likelihoods, pwc_transition, cell_volume)
            pwc_filter_runtime = time.perf_counter() - pwc_filter_start

            pwc_smoother_start = time.perf_counter()
            pwc_result = grid_backward_information_smoother(
                pwc_filtered,
                likelihoods,
                pwc_transition,
                cell_volume=cell_volume,
            )
            pwc_smoother_runtime = time.perf_counter() - pwc_smoother_start

            evaluation_start = time.perf_counter()
            pwc_evaluated = _evaluate_pwc_1d(pwc_result.smoothed, reference_grid_size)
            evaluation_runtime = time.perf_counter() - evaluation_start
            rows.append(
                _make_figf_pwc_row(
                    method="PWC",
                    grid_size=grid_size,
                    repetition=repetition,
                    filter_runtime_s=pwc_filter_runtime,
                    smoother_runtime_s=pwc_smoother_runtime,
                    evaluation_runtime_s=evaluation_runtime,
                    evaluated=pwc_evaluated,
                    reference=reference_smoothed,
                    reference_cell_volume=reference_cell_volume,
                )
            )
    return rows


def run_smoothing_evaluation(
    figf_grid_sizes: Iterable[int] = (15, 31, 63, 127, 255, 511, 1023, 2047, 4095),
    pwc_grid_sizes: Iterable[int] = (15, 31, 63, 127, 255, 511, 1023, 2047, 4095),
    pf_particle_counts: Iterable[int] = (100, 300, 1000, 3000, 10000),
    *,
    repetitions: int = 3,
    time_steps: int = 9,
    likelihood_sharpness: float = 5.0,
    noise_concentration: float = 4.0,
    l1_reference_grid_size: int = 65_535,
    mean_reference_particles: int = 1_000_000,
    mean_reference_repetitions: int = 3,
    particle_kde_bandwidth_scale: float = 1.0,
    pwc_quadrature_points: int = 8,
    seed: int = 1,
) -> list[SmoothingEvaluationRow]:
    """Evaluate FIGFAN, FIGFDN, PWC, and PF smoothers.

    Mean errors are measured against an aggregate of independent high-sample
    bootstrap-PF/FFBSi references. L1 errors are measured against a
    high-resolution PWC reference density. PF densities are reconstructed with
    a wrapped-normal kernel density estimator whose bandwidth is proportional
    to ``n_particles**(-1/5)``.
    """

    figf_grid_sizes = _positive_int_tuple(figf_grid_sizes, "figf_grid_sizes")
    pwc_grid_sizes = _positive_int_tuple(pwc_grid_sizes, "pwc_grid_sizes")
    pf_particle_counts = _positive_int_tuple(pf_particle_counts, "pf_particle_counts")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    if l1_reference_grid_size <= 0:
        raise ValueError("l1_reference_grid_size must be positive.")
    if mean_reference_particles <= 0:
        raise ValueError("mean_reference_particles must be positive.")
    if mean_reference_repetitions < 1:
        raise ValueError("mean_reference_repetitions must be at least one.")
    if particle_kde_bandwidth_scale <= 0.0:
        raise ValueError("particle_kde_bandwidth_scale must be positive.")

    reference_shape = (int(l1_reference_grid_size),)
    reference_cell_volume = cell_volume_for_grid(reference_shape)
    reference_likelihoods = make_sharp_multimodal_likelihoods(
        reference_shape,
        time_steps,
        sharpness=likelihood_sharpness,
    )
    reference_pwc_likelihoods = make_sharp_multimodal_likelihoods(
        reference_shape,
        time_steps,
        sharpness=likelihood_sharpness,
        grid_offset=0.5,
    )
    reference_pwc_density = _run_pwc_smoother_1d(
        reference_pwc_likelihoods,
        noise_concentration,
        quadrature_points=pwc_quadrature_points,
    )
    reference_seed_sequence = np.random.SeedSequence(seed)
    reference_moments = []
    for child_seed in reference_seed_sequence.spawn(mean_reference_repetitions):
        reference_run_seed = int(child_seed.generate_state(1)[0])
        _, reference_smoother = _run_von_mises_ffbsi_1d(
            reference_likelihoods,
            noise_concentration,
            mean_reference_particles,
            seed=reference_run_seed,
        )
        reference_moments.append(np.mean(np.exp(1j * reference_smoother.trajectories), axis=0))
    mean_reference = np.mod(np.angle(np.mean(reference_moments, axis=0)), 2.0 * np.pi)

    rows: list[SmoothingEvaluationRow] = []
    seed_sequence = np.random.SeedSequence(seed + 1000)
    n_pf_runs = len(pf_particle_counts) * repetitions
    pf_seeds = iter(seed_sequence.spawn(n_pf_runs))

    for grid_size in figf_grid_sizes:
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_sharp_multimodal_likelihoods(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
        )
        noise = make_von_mises_like_noise(grid_shape, noise_concentration)
        transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
        figfan_template = None
        figfdn_template = None
        for repetition in range(repetitions):
            start = time.perf_counter()
            filtered = _run_figf_forward_filter(likelihoods, noise, cell_volume)
            result = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)
            runtime = time.perf_counter() - start
            if figfan_template is None or figfdn_template is None:
                figfan_density = _evaluate_figfan_1d(result.smoothed, l1_reference_grid_size)
                figfdn_density = _evaluate_figfdn_1d(result.smoothed, l1_reference_grid_size)
                figfan_template = _make_smoothing_evaluation_row(
                    method="FIGFAN",
                    parameter=grid_size,
                    repetition=0,
                    runtime_s=runtime,
                    evaluated=figfan_density,
                    mean_reference=mean_reference,
                    l1_reference=reference_pwc_density,
                    reference_cell_volume=reference_cell_volume,
                )
                figfdn_template = _make_smoothing_evaluation_row(
                    method="FIGFDN",
                    parameter=grid_size,
                    repetition=0,
                    runtime_s=runtime,
                    evaluated=figfdn_density,
                    mean_reference=mean_reference,
                    l1_reference=reference_pwc_density,
                    reference_cell_volume=reference_cell_volume,
                )
            rows.append(_evaluation_row_with_timing(figfan_template, repetition, runtime))
            rows.append(_evaluation_row_with_timing(figfdn_template, repetition, runtime))

    for grid_size in pwc_grid_sizes:
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_sharp_multimodal_likelihoods(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
            grid_offset=0.5,
        )
        kernel = make_pwc_additive_transition_kernel_1d(
            grid_size,
            noise_concentration,
            quadrature_points=pwc_quadrature_points,
        )
        pwc_template = None
        for repetition in range(repetitions):
            start = time.perf_counter()
            filtered = _run_pwc_forward_filter(likelihoods, kernel, cell_volume)
            smoothed = grid_backward_information_smoother(
                filtered,
                likelihoods,
                lambda message, _t: _pwc_backward_predict_fft(message, kernel, cell_volume),
                cell_volume=cell_volume,
            ).smoothed
            runtime = time.perf_counter() - start
            if pwc_template is None:
                evaluated = _evaluate_pwc_1d(smoothed, l1_reference_grid_size)
                pwc_template = _make_smoothing_evaluation_row(
                    method="PWC",
                    parameter=grid_size,
                    repetition=0,
                    runtime_s=runtime,
                    evaluated=evaluated,
                    mean_reference=mean_reference,
                    l1_reference=reference_pwc_density,
                    reference_cell_volume=reference_cell_volume,
                    means=_pwc_circular_means_1d(smoothed),
                )
            rows.append(_evaluation_row_with_timing(pwc_template, repetition, runtime))

    for n_particles in pf_particle_counts:
        for repetition in range(repetitions):
            run_seed = int(next(pf_seeds).generate_state(1)[0])
            start = time.perf_counter()
            _, particle_smoother = _run_von_mises_ffbsi_1d(
                reference_likelihoods,
                noise_concentration,
                n_particles,
                seed=run_seed,
            )
            runtime = time.perf_counter() - start
            particle_density = _particle_trajectories_to_wrapped_normal_kde_1d(
                particle_smoother.trajectories,
                l1_reference_grid_size,
                bandwidth_scale=particle_kde_bandwidth_scale,
            )
            rows.append(
                _make_smoothing_evaluation_row(
                    method="PF",
                    parameter=n_particles,
                    repetition=repetition,
                    runtime_s=runtime,
                    evaluated=particle_density,
                    mean_reference=mean_reference,
                    l1_reference=reference_pwc_density,
                    reference_cell_volume=reference_cell_volume,
                    means=particle_smoother.mean_directions,
                )
            )

    return rows


def run_smoothing_runtime_evaluation(
    figf_grid_sizes: Iterable[int] = (15, 31, 63, 127, 255, 511, 1023, 2047, 4095),
    pwc_grid_sizes: Iterable[int] = (15, 31, 63, 127, 255, 511, 1023, 2047, 4095),
    pf_particle_counts: Iterable[int] = (100, 300, 1000, 3000, 10000),
    *,
    repetitions: int = 30,
    time_steps: int = 9,
    likelihood_sharpness: float = 5.0,
    noise_concentration: float = 4.0,
    particle_likelihood_grid_size: int = 65_535,
    pwc_quadrature_points: int = 8,
    seed: int = 1,
) -> list[SmoothingRuntimeRow]:
    """Time only the forward filters and backward smoothers.

    Reference generation, interpolation, densification, and particle KDE work
    are deliberately absent. The returned keys match :func:`run_smoothing_evaluation`
    so timings from a controlled host can be paired with errors computed elsewhere.
    """

    figf_grid_sizes = _positive_int_tuple(figf_grid_sizes, "figf_grid_sizes")
    pwc_grid_sizes = _positive_int_tuple(pwc_grid_sizes, "pwc_grid_sizes")
    pf_particle_counts = _positive_int_tuple(pf_particle_counts, "pf_particle_counts")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    if particle_likelihood_grid_size < 1:
        raise ValueError("particle_likelihood_grid_size must be positive.")
    if pwc_quadrature_points < 1:
        raise ValueError("pwc_quadrature_points must be at least one.")

    rows: list[SmoothingRuntimeRow] = []
    for grid_size in figf_grid_sizes:
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_sharp_multimodal_likelihoods(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
        )
        noise = make_von_mises_like_noise(grid_shape, noise_concentration)
        transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
        for repetition in range(repetitions):
            start = time.perf_counter()
            filtered = _run_figf_forward_filter(likelihoods, noise, cell_volume)
            grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)
            runtime = time.perf_counter() - start
            rows.append(SmoothingRuntimeRow("FIGFAN", grid_size, repetition, runtime))
            rows.append(SmoothingRuntimeRow("FIGFDN", grid_size, repetition, runtime))

    for grid_size in pwc_grid_sizes:
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_sharp_multimodal_likelihoods(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
            grid_offset=0.5,
        )
        kernel = make_pwc_additive_transition_kernel_1d(
            grid_size,
            noise_concentration,
            quadrature_points=pwc_quadrature_points,
        )
        for repetition in range(repetitions):
            start = time.perf_counter()
            filtered = _run_pwc_forward_filter(likelihoods, kernel, cell_volume)
            grid_backward_information_smoother(
                filtered,
                likelihoods,
                lambda message, _t: _pwc_backward_predict_fft(message, kernel, cell_volume),
                cell_volume=cell_volume,
            )
            runtime = time.perf_counter() - start
            rows.append(SmoothingRuntimeRow("PWC", grid_size, repetition, runtime))

    particle_likelihoods = make_sharp_multimodal_likelihoods(
        (int(particle_likelihood_grid_size),),
        time_steps,
        sharpness=likelihood_sharpness,
    )
    seed_sequence = np.random.SeedSequence(seed + 1000)
    pf_seeds = iter(seed_sequence.spawn(len(pf_particle_counts) * repetitions))
    for n_particles in pf_particle_counts:
        for repetition in range(repetitions):
            run_seed = int(next(pf_seeds).generate_state(1)[0])
            start = time.perf_counter()
            _run_von_mises_ffbsi_1d(
                particle_likelihoods,
                noise_concentration,
                n_particles,
                seed=run_seed,
            )
            runtime = time.perf_counter() - start
            rows.append(SmoothingRuntimeRow("PF", n_particles, repetition, runtime))
    return rows


def run_smoothing_gain_evaluation(
    *,
    n_trials: int = 500,
    grid_size: int = 1023,
    time_steps: int = 20,
    prior_concentration: float = 1.0,
    noise_concentration: float = 8.0,
    likelihood_concentration: float = 12.0,
    outlier_probability: float = 0.3,
    outlier_offset: float = 2.35,
    seed: int = 21,
) -> list[SmoothingGainRow]:
    """Measure filtering and smoothing mean errors against simulated states.

    Measurements follow a two-component circular-noise model. The second
    component is shifted by ``outlier_offset``, producing an ambiguous
    likelihood while remaining a proper generative model. The returned raw
    rows include the final time step; summaries should exclude it because
    filtering and fixed-interval smoothing coincide there.
    """

    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least three.")
    if time_steps < 2:
        raise ValueError("time_steps must be at least two.")
    if min(prior_concentration, noise_concentration, likelihood_concentration) <= 0.0:
        raise ValueError("all concentration parameters must be positive.")
    if not 0.0 <= outlier_probability <= 1.0:
        raise ValueError("outlier_probability must lie in [0, 1].")

    rng = np.random.default_rng(seed)
    grid_shape = (int(grid_size),)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    prior = normalize_grid_density(np.exp(prior_concentration * np.cos(x)), cell_volume)
    process_noise = make_von_mises_like_noise(grid_shape, noise_concentration)
    transition = TorusAdditiveGridTransition.for_grid_shape(process_noise, grid_shape)

    rows: list[SmoothingGainRow] = []
    for trial in range(n_trials):
        states = np.empty(time_steps, dtype=float)
        states[0] = np.mod(rng.vonmises(0.0, prior_concentration), 2.0 * np.pi)
        states[1:] = np.mod(
            states[0] + np.cumsum(rng.vonmises(0.0, noise_concentration, size=time_steps - 1)),
            2.0 * np.pi,
        )
        shifted_components = rng.random(time_steps) < outlier_probability
        measurement_noise = rng.vonmises(0.0, likelihood_concentration, size=time_steps)
        measurement_noise += shifted_components * outlier_offset
        measurements = np.mod(states + measurement_noise, 2.0 * np.pi)

        likelihoods = []
        for measurement in measurements:
            residual = measurement - x
            values = (
                (1.0 - outlier_probability) * np.exp(likelihood_concentration * np.cos(residual))
                + outlier_probability
                * np.exp(likelihood_concentration * np.cos(residual - outlier_offset))
            )
            likelihoods.append(normalize_grid_density(values, cell_volume))
        likelihood_array = np.stack(likelihoods, axis=0)

        filtered = _run_figf_forward_filter_with_prior(
            likelihood_array,
            process_noise,
            prior,
            cell_volume,
        )
        smoothed = grid_backward_information_smoother(
            filtered,
            likelihood_array,
            transition,
            cell_volume=cell_volume,
        ).smoothed
        filter_means = _density_circular_means_1d(filtered, cell_volume)
        smoother_means = _density_circular_means_1d(smoothed, cell_volume)
        filter_errors = _circular_abs_difference(filter_means, states)
        smoother_errors = _circular_abs_difference(smoother_means, states)
        rows.extend(
            SmoothingGainRow(
                trial=trial,
                time_step=t,
                filter_error_rad=float(filter_errors[t]),
                smoother_error_rad=float(smoother_errors[t]),
            )
            for t in range(time_steps)
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


def write_figf_pwc_csv(rows: Sequence[FIGFPWCBenchmarkRow], output_path: str | Path) -> Path:
    """Write FIGF/PWC benchmark rows to ``output_path``."""

    return _write_csv(rows, output_path, FIGF_PWC_CSV_COLUMNS)


def write_smoothing_evaluation_csv(rows: Sequence[SmoothingEvaluationRow], output_path: str | Path) -> Path:
    """Write raw smoothing-evaluation rows to ``output_path``."""

    return _write_csv(rows, output_path, SMOOTHING_EVALUATION_CSV_COLUMNS)


def write_smoothing_gain_csv(rows: Sequence[SmoothingGainRow], output_path: str | Path) -> Path:
    """Write raw state-truth smoothing-gain rows to ``output_path``."""

    return _write_csv(rows, output_path, SMOOTHING_GAIN_CSV_COLUMNS)


def make_pwc_additive_transition_kernel_1d(
    grid_size: int,
    concentration: float,
    *,
    quadrature_points: int = 8,
) -> NDArray[np.float64]:
    """Cell-average an additive von-Mises transition into a circulant kernel."""

    grid_size = int(grid_size)
    quadrature_points = int(quadrature_points)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    if quadrature_points < 1:
        raise ValueError("quadrature_points must be at least one.")

    cell_width = 2.0 * np.pi / grid_size
    cell_delta = cell_width * np.arange(grid_size)
    offsets = cell_width * (np.arange(quadrature_points, dtype=float) + 0.5) / quadrature_points
    subcell_delta = offsets[:, None] - offsets[None, :]
    kernel = _von_mises_density_1d(
        cell_delta[:, None, None] + subcell_delta[None, :, :],
        concentration,
    ).mean(axis=(1, 2))
    return kernel / (np.sum(kernel) * cell_width)


def make_pwc_additive_transition_density_matrix_1d(
    grid_size: int,
    concentration: float,
    *,
    quadrature_points: int = 8,
) -> NDArray[np.float64]:
    """Cell-average an additive von-Mises transition for a 1-D PWC basis."""

    kernel = make_pwc_additive_transition_kernel_1d(
        grid_size,
        concentration,
        quadrature_points=quadrature_points,
    )
    offsets = (np.arange(grid_size)[:, None] - np.arange(grid_size)[None, :]) % grid_size
    transition = kernel[offsets]
    cell_width = 2.0 * np.pi / grid_size
    column_integrals = np.sum(transition, axis=0, keepdims=True) * cell_width
    return transition / column_integrals


def _write_csv(rows, output_path: str | Path, fieldnames: Sequence[str]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def _max_grid_normalization_error(values: NDArray[np.float64], cell_volume: float) -> float:
    integrals = np.sum(values, axis=tuple(range(1, values.ndim))) * cell_volume
    return float(np.max(np.abs(integrals - 1.0)))


def _run_figf_forward_filter(
    likelihoods: NDArray[np.float64],
    noise: NDArray[np.float64],
    cell_volume: float,
) -> NDArray[np.float64]:
    filtered = []
    current = normalize_grid_density(likelihoods[0], cell_volume)
    filtered.append(current)
    for likelihood in likelihoods[1:]:
        predicted = _forward_predict_additive_fft(current, noise, cell_volume)
        current = normalize_grid_density(predicted * likelihood, cell_volume)
        filtered.append(current)
    return np.stack(filtered, axis=0)


def _run_figf_forward_filter_with_prior(
    likelihoods: NDArray[np.float64],
    noise: NDArray[np.float64],
    prior: NDArray[np.float64],
    cell_volume: float,
) -> NDArray[np.float64]:
    filtered = []
    current = normalize_grid_density(prior * likelihoods[0], cell_volume)
    filtered.append(current)
    for likelihood in likelihoods[1:]:
        predicted = _forward_predict_additive_fft(current, noise, cell_volume)
        current = normalize_grid_density(predicted * likelihood, cell_volume)
        filtered.append(current)
    return np.stack(filtered, axis=0)


def _run_dense_transition_forward_filter(
    likelihoods: NDArray[np.float64],
    transition: DenseGridTransition,
    cell_volume: float,
) -> NDArray[np.float64]:
    filtered = []
    current = normalize_grid_density(likelihoods[0], cell_volume)
    filtered.append(current)
    for t, likelihood in enumerate(likelihoods[1:]):
        predicted = transition.forward_predict(current, t)
        current = normalize_grid_density(predicted * likelihood, cell_volume)
        filtered.append(current)
    return np.stack(filtered, axis=0)


def _run_pwc_smoother_1d(
    likelihoods: NDArray[np.float64],
    noise_concentration: float,
    *,
    quadrature_points: int,
) -> NDArray[np.float64]:
    grid_size = likelihoods.shape[1]
    cell_volume = cell_volume_for_grid((grid_size,))
    kernel = make_pwc_additive_transition_kernel_1d(
        grid_size,
        noise_concentration,
        quadrature_points=quadrature_points,
    )
    filtered = _run_pwc_forward_filter(likelihoods, kernel, cell_volume)
    result = grid_backward_information_smoother(
        filtered,
        likelihoods,
        lambda message, _t: _pwc_backward_predict_fft(message, kernel, cell_volume),
        cell_volume=cell_volume,
    )
    return result.smoothed


def _run_pwc_forward_filter(
    likelihoods: NDArray[np.float64],
    kernel: NDArray[np.float64],
    cell_volume: float,
) -> NDArray[np.float64]:
    filtered = []
    current = normalize_grid_density(likelihoods[0], cell_volume)
    filtered.append(current)
    for likelihood in likelihoods[1:]:
        predicted = _pwc_forward_predict_fft(current, kernel, cell_volume)
        current = normalize_grid_density(predicted * likelihood, cell_volume)
        filtered.append(current)
    return np.stack(filtered, axis=0)


def _pwc_forward_predict_fft(
    density: NDArray[np.float64],
    kernel: NDArray[np.float64],
    cell_volume: float,
) -> NDArray[np.float64]:
    predicted = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(density)).real
    return normalize_grid_density(np.maximum(predicted * cell_volume, 0.0), cell_volume)


def _pwc_backward_predict_fft(
    message: NDArray[np.float64],
    kernel: NDArray[np.float64],
    cell_volume: float,
) -> NDArray[np.float64]:
    predicted = np.fft.ifft(np.conj(np.fft.fft(kernel)) * np.fft.fft(message)).real
    return np.maximum(predicted * cell_volume, 0.0)


def _forward_predict_additive_fft(
    density: NDArray[np.float64],
    noise: NDArray[np.float64],
    cell_volume: float,
) -> NDArray[np.float64]:
    predicted = np.fft.ifftn(np.fft.fftn(noise) * np.fft.fftn(density)).real
    return normalize_grid_density(np.maximum(predicted * cell_volume, 0.0), cell_volume)


def _evaluate_figfan_1d(smoothed: NDArray[np.float64], evaluation_grid_size: int) -> NDArray[np.float64]:
    grid_shape = (int(evaluation_grid_size),)
    return np.stack(
        [fourier_to_grid(grid_to_fourier(values), grid_shape=grid_shape) for values in smoothed],
        axis=0,
    )


def _evaluate_figfdn_1d(smoothed: NDArray[np.float64], evaluation_grid_size: int) -> NDArray[np.float64]:
    grid_shape = (int(evaluation_grid_size),)
    cell_volume = cell_volume_for_grid(grid_shape)
    evaluated = []
    for values in smoothed:
        sqrt_values = np.sqrt(np.maximum(values, 0.0))
        dense_sqrt = fourier_to_grid(grid_to_fourier(sqrt_values), grid_shape=grid_shape)
        dense_values = dense_sqrt**2
        evaluated.append(dense_values / (np.sum(dense_values) * cell_volume))
    return np.stack(evaluated, axis=0)


def _evaluate_pwc_1d(smoothed: NDArray[np.float64], evaluation_grid_size: int) -> NDArray[np.float64]:
    grid_size = smoothed.shape[1]
    dense_indices = (np.arange(evaluation_grid_size) * grid_size // evaluation_grid_size).astype(int)
    evaluated = smoothed[:, dense_indices]
    cell_volume = cell_volume_for_grid((int(evaluation_grid_size),))
    integrals = np.sum(evaluated, axis=1, keepdims=True) * cell_volume
    return evaluated / integrals


def _make_figf_pwc_row(
    *,
    method: str,
    grid_size: int,
    repetition: int,
    filter_runtime_s: float,
    smoother_runtime_s: float,
    evaluation_runtime_s: float,
    evaluated: NDArray[np.float64],
    reference: NDArray[np.float64],
    reference_cell_volume: float,
) -> FIGFPWCBenchmarkRow:
    l1_by_time = np.sum(np.abs(evaluated - reference), axis=1) * reference_cell_volume
    normalization_error = _max_grid_normalization_error(evaluated, reference_cell_volume)
    return FIGFPWCBenchmarkRow(
        method=method,
        grid_size=grid_size,
        repetition=repetition,
        filter_runtime_s=filter_runtime_s,
        smoother_runtime_s=smoother_runtime_s,
        evaluation_runtime_s=evaluation_runtime_s,
        runtime_s=filter_runtime_s + smoother_runtime_s + evaluation_runtime_s,
        mean_l1_error_to_reference=float(np.mean(l1_by_time)),
        max_l1_error_to_reference=float(np.max(l1_by_time)),
        min_evaluated_density=float(np.min(evaluated)),
        max_normalization_error=normalization_error,
    )


def _make_smoothing_evaluation_row(
    *,
    method: str,
    parameter: int,
    repetition: int,
    runtime_s: float,
    evaluated: NDArray[np.float64],
    mean_reference: NDArray[np.float64],
    l1_reference: NDArray[np.float64],
    reference_cell_volume: float,
    means: NDArray[np.float64] | None = None,
) -> SmoothingEvaluationRow:
    if means is None:
        means = _density_circular_means_1d(evaluated, reference_cell_volume)
    mean_errors = _circular_abs_difference(means, mean_reference)
    l1_errors = np.sum(np.abs(evaluated - l1_reference), axis=1) * reference_cell_volume
    return SmoothingEvaluationRow(
        method=method,
        parameter=int(parameter),
        repetition=int(repetition),
        runtime_s=float(runtime_s),
        mean_error_rad=float(np.mean(mean_errors)),
        max_mean_error_rad=float(np.max(mean_errors)),
        l1_error=float(np.mean(l1_errors)),
        max_l1_error=float(np.max(l1_errors)),
        min_evaluated_density=float(np.min(evaluated)),
        max_normalization_error=_max_grid_normalization_error(evaluated, reference_cell_volume),
    )


def _evaluation_row_with_timing(
    template: SmoothingEvaluationRow,
    repetition: int,
    runtime_s: float,
) -> SmoothingEvaluationRow:
    return SmoothingEvaluationRow(
        method=template.method,
        parameter=template.parameter,
        repetition=int(repetition),
        runtime_s=float(runtime_s),
        mean_error_rad=template.mean_error_rad,
        max_mean_error_rad=template.max_mean_error_rad,
        l1_error=template.l1_error,
        max_l1_error=template.max_l1_error,
        min_evaluated_density=template.min_evaluated_density,
        max_normalization_error=template.max_normalization_error,
    )


def _density_circular_means_1d(values: NDArray[np.float64], cell_volume: float) -> NDArray[np.float64]:
    grid_size = values.shape[1]
    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    moments = np.sum(values * np.exp(1j * angles)[None, :], axis=1) * cell_volume
    return np.mod(np.angle(moments), 2.0 * np.pi)


def _pwc_circular_means_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return exact mean directions for cell-centered equal-width PWC values."""

    grid_size = values.shape[1]
    cell_centers = 2.0 * np.pi * (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    moments = np.sum(values * np.exp(1j * cell_centers)[None, :], axis=1)
    return np.mod(np.angle(moments), 2.0 * np.pi)


def _run_von_mises_ffbsi_1d(
    likelihoods: NDArray[np.float64],
    noise_concentration: float,
    n_particles: int,
    *,
    seed: int,
):
    rng = np.random.default_rng(seed)
    filtered = bootstrap_von_mises_particle_filter_1d(
        likelihoods,
        noise_concentration,
        n_particles,
        rng=rng,
    )
    smoothed = ffbsi_von_mises_particle_smoother_1d(
        filtered,
        noise_concentration,
        n_particles,
        rng=rng,
    )
    return filtered, smoothed


def _particle_trajectories_to_wrapped_normal_kde_1d(
    trajectories: NDArray[np.float64],
    evaluation_grid_size: int,
    *,
    bandwidth_scale: float,
) -> NDArray[np.float64]:
    """Reconstruct marginal densities with wrapped-normal kernels.

    Linear cloud-in-cell binning avoids a grid-origin bias. Convolution with a
    wrapped normal of standard deviation ``bandwidth_scale * N**(-1/5)`` is
    then performed spectrally on the evaluation grid.
    """

    samples = np.asarray(trajectories, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("trajectories must have shape (n_trajectories, time_steps).")
    if evaluation_grid_size <= 0:
        raise ValueError("evaluation_grid_size must be positive.")
    if bandwidth_scale <= 0.0:
        raise ValueError("bandwidth_scale must be positive.")

    n_trajectories, time_steps = samples.shape
    grid_size = int(evaluation_grid_size)
    cell_volume = cell_volume_for_grid((grid_size,))
    bandwidth = max(cell_volume, float(bandwidth_scale) * n_trajectories ** (-0.2))
    frequencies = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    kernel_multiplier = np.exp(-0.5 * bandwidth**2 * frequencies**2)

    densities = np.empty((time_steps, grid_size), dtype=float)
    for t in range(time_steps):
        positions = np.mod(samples[:, t], 2.0 * np.pi) * grid_size / (2.0 * np.pi)
        left = np.floor(positions).astype(np.int64) % grid_size
        fraction = positions - np.floor(positions)
        masses = np.bincount(left, weights=1.0 - fraction, minlength=grid_size)
        masses += np.bincount((left + 1) % grid_size, weights=fraction, minlength=grid_size)
        binned_density = masses / (n_trajectories * cell_volume)
        density = np.fft.ifft(np.fft.fft(binned_density) * kernel_multiplier).real
        densities[t] = normalize_grid_density(np.maximum(density, 0.0), cell_volume)
    return densities


def _path_particle_smoother_summary_1d(
    likelihoods: NDArray[np.float64],
    noise_concentration: float,
    n_particles: int,
    *,
    density_grid_size: int | None,
    seed: int,
    chunk_size: int,
) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    time_steps = likelihoods.shape[0]
    n_remaining = int(n_particles)
    log_weight_offset = -np.inf
    total_weight = 0.0
    moments = np.zeros(time_steps, dtype=np.complex128)
    histogram = None if density_grid_size is None else np.zeros((time_steps, int(density_grid_size)), dtype=float)

    while n_remaining > 0:
        current_chunk_size = min(int(chunk_size), n_remaining)
        paths = _sample_torus_paths_1d(current_chunk_size, time_steps, noise_concentration, rng)
        log_weights = _path_log_likelihoods_1d(paths, likelihoods)
        chunk_max = float(np.max(log_weights))
        new_offset = max(log_weight_offset, chunk_max)
        if np.isneginf(new_offset):
            scale_old = 0.0
        else:
            scale_old = 0.0 if np.isneginf(log_weight_offset) else float(np.exp(log_weight_offset - new_offset))
        scaled_weights = np.exp(log_weights - new_offset)

        total_weight = total_weight * scale_old + float(np.sum(scaled_weights))
        moments = moments * scale_old + np.sum(scaled_weights[:, None] * np.exp(1j * paths), axis=0)
        if histogram is not None:
            histogram *= scale_old
            indices = np.floor(paths * density_grid_size / (2.0 * np.pi)).astype(int) % density_grid_size
            for t in range(time_steps):
                histogram[t] += np.bincount(indices[:, t], weights=scaled_weights, minlength=density_grid_size)

        log_weight_offset = new_offset
        n_remaining -= current_chunk_size

    means = np.mod(np.angle(moments / total_weight), 2.0 * np.pi)
    result: dict[str, NDArray[np.float64]] = {"means": means}
    if histogram is not None:
        cell_volume = cell_volume_for_grid((int(density_grid_size),))
        density = histogram / (total_weight * cell_volume)
        result["density"] = density
    return result


def _path_particle_filter_summary_1d(
    likelihoods: NDArray[np.float64],
    noise_concentration: float,
    n_particles: int,
    *,
    density_grid_size: int | None,
    seed: int,
    chunk_size: int,
) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    time_steps = likelihoods.shape[0]
    n_remaining = int(n_particles)
    log_weight_offsets = np.full(time_steps, -np.inf, dtype=float)
    total_weights = np.zeros(time_steps, dtype=float)
    moments = np.zeros(time_steps, dtype=np.complex128)
    histogram = None if density_grid_size is None else np.zeros((time_steps, int(density_grid_size)), dtype=float)

    while n_remaining > 0:
        current_chunk_size = min(int(chunk_size), n_remaining)
        paths = _sample_torus_paths_1d(current_chunk_size, time_steps, noise_concentration, rng)
        cumulative_log_weights = np.zeros(current_chunk_size, dtype=float)
        for t, likelihood in enumerate(likelihoods):
            values = _periodic_linear_interpolate_1d(likelihood, paths[:, t])
            cumulative_log_weights += np.log(np.maximum(values, np.finfo(float).tiny))
            chunk_max = float(np.max(cumulative_log_weights))
            new_offset = max(log_weight_offsets[t], chunk_max)
            if np.isneginf(new_offset):
                scale_old = 0.0
            else:
                scale_old = 0.0 if np.isneginf(log_weight_offsets[t]) else float(np.exp(log_weight_offsets[t] - new_offset))
            scaled_weights = np.exp(cumulative_log_weights - new_offset)

            total_weights[t] = total_weights[t] * scale_old + float(np.sum(scaled_weights))
            moments[t] = moments[t] * scale_old + np.sum(scaled_weights * np.exp(1j * paths[:, t]))
            if histogram is not None:
                histogram[t] *= scale_old
                indices = np.floor(paths[:, t] * density_grid_size / (2.0 * np.pi)).astype(int) % density_grid_size
                histogram[t] += np.bincount(indices, weights=scaled_weights, minlength=density_grid_size)

            log_weight_offsets[t] = new_offset
        n_remaining -= current_chunk_size

    means = np.mod(np.angle(moments / total_weights), 2.0 * np.pi)
    result: dict[str, NDArray[np.float64]] = {"means": means}
    if histogram is not None:
        cell_volume = cell_volume_for_grid((int(density_grid_size),))
        density = histogram / (total_weights[:, None] * cell_volume)
        result["density"] = density
    return result


def _sample_torus_paths_1d(
    n_particles: int,
    time_steps: int,
    noise_concentration: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    paths = np.empty((n_particles, time_steps), dtype=float)
    paths[:, 0] = rng.uniform(0.0, 2.0 * np.pi, size=n_particles)
    if time_steps > 1:
        innovations = rng.vonmises(0.0, noise_concentration, size=(n_particles, time_steps - 1))
        paths[:, 1:] = np.mod(paths[:, [0]] + np.cumsum(innovations, axis=1), 2.0 * np.pi)
    return paths


def _path_log_likelihoods_1d(
    paths: NDArray[np.float64],
    likelihoods: NDArray[np.float64],
) -> NDArray[np.float64]:
    log_weights = np.zeros(paths.shape[0], dtype=float)
    for t, likelihood in enumerate(likelihoods):
        values = _periodic_linear_interpolate_1d(likelihood, paths[:, t])
        log_weights += np.log(np.maximum(values, np.finfo(float).tiny))
    return log_weights


def _periodic_linear_interpolate_1d(values: NDArray[np.float64], angles: NDArray[np.float64]) -> NDArray[np.float64]:
    positions = np.mod(angles, 2.0 * np.pi) * values.size / (2.0 * np.pi)
    left = np.floor(positions).astype(int) % values.size
    right = (left + 1) % values.size
    fraction = positions - np.floor(positions)
    return (1.0 - fraction) * values[left] + fraction * values[right]


def _circular_abs_difference(left, right) -> NDArray[np.float64]:
    difference = np.mod(np.asarray(left) - np.asarray(right) + np.pi, 2.0 * np.pi) - np.pi
    return np.abs(difference)


def _positive_int_tuple(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def _von_mises_density_1d(angle: NDArray[np.float64], concentration: float) -> NDArray[np.float64]:
    return np.exp(concentration * np.cos(angle)) / (2.0 * np.pi * np.i0(concentration))
