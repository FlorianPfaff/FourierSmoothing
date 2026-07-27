"""Representation-faithful runners for the paper benchmarks.

FIGF stores point values, whereas the PWC baseline stores one constant value
per interval cell. The analytic likelihood must therefore be sampled at FIGF
nodes but averaged over PWC cells. This module preserves the public benchmark
API while applying that projection consistently.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from . import experiments as _experiments
from .experiments import (
    FIGFPWCBenchmarkRow,
    SmoothingEvaluationRow,
    SmoothingRuntimeRow,
    _evaluate_figfan_1d,
    _evaluate_figfdn_1d,
    _evaluate_pwc_1d,
    _make_figf_pwc_row,
    _run_dense_transition_forward_filter,
    _run_figf_forward_filter,
    make_pwc_additive_transition_density_matrix_1d,
    make_sharp_multimodal_likelihoods,
    make_von_mises_like_noise,
)
from .smoother import (
    DenseGridTransition,
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    grid_backward_information_smoother,
    normalize_grid_density,
)

# Capture the original functions before package-level compatibility aliases are
# installed in ``fourier_smoothing.__init__``.
_ORIGINAL_RUN_SMOOTHING_EVALUATION = _experiments.run_smoothing_evaluation
_ORIGINAL_RUN_SMOOTHING_RUNTIME_EVALUATION = _experiments.run_smoothing_runtime_evaluation
_ORIGINAL_MAKE_LIKELIHOODS = _experiments.make_sharp_multimodal_likelihoods


def make_pwc_cell_averaged_likelihoods_1d(
    grid_shape: Sequence[int],
    time_steps: int,
    *,
    sharpness: float,
    quadrature_points: int = 8,
) -> NDArray[np.float64]:
    """Project the analytic paper likelihood onto equal-width PWC cells.

    The value stored in cell ``j`` approximates

    ``(1 / cell_width) * integral_{cell j} likelihood_t(x) dx``

    using midpoint quadrature. This is the measurement-update factor for a
    density represented as constant inside each interval cell.
    """

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 1:
        raise ValueError("make_pwc_cell_averaged_likelihoods_1d supports only 1-D grids.")
    if shape[0] <= 0:
        raise ValueError("grid_shape must contain a positive size.")
    if time_steps < 1:
        raise ValueError("time_steps must be at least one.")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")
    quadrature_points = int(quadrature_points)
    if quadrature_points < 1:
        raise ValueError("quadrature_points must be at least one.")

    grid_size = shape[0]
    cell_width = 2.0 * np.pi / grid_size
    offsets = cell_width * (np.arange(quadrature_points, dtype=float) + 0.5) / quadrature_points
    x = cell_width * np.arange(grid_size, dtype=float)[:, None] + offsets[None, :]

    likelihoods = []
    for t in range(time_steps):
        phase = 0.43 * (t + 1)
        point_values = (
            0.04
            + np.exp(sharpness * np.cos(x - phase))
            + 0.65 * np.exp(0.85 * sharpness * np.cos(x - phase - 2.35))
            + 0.35 * np.exp(0.55 * sharpness * np.cos(2.0 * x + phase))
        )
        cell_averages = np.mean(point_values, axis=1)
        likelihoods.append(normalize_grid_density(cell_averages, cell_width))
    return np.stack(likelihoods, axis=0)


@contextmanager
def _cell_average_pwc_projection(quadrature_points: int) -> Iterator[None]:
    """Temporarily make ``grid_offset=0.5`` request PWC cell averages.

    The established main runners already distinguish FIGF and PWC likelihoods
    by passing ``grid_offset=0.5`` only for PWC. A scoped replacement therefore
    corrects the projection without duplicating the complete experiment logic.
    """

    def representation_aware_likelihoods(
        grid_shape: Sequence[int],
        time_steps: int,
        *,
        sharpness: float,
        grid_offset: float = 0.0,
    ) -> NDArray[np.float64]:
        if np.isclose(grid_offset, 0.5):
            return make_pwc_cell_averaged_likelihoods_1d(
                grid_shape,
                time_steps,
                sharpness=sharpness,
                quadrature_points=quadrature_points,
            )
        return _ORIGINAL_MAKE_LIKELIHOODS(
            grid_shape,
            time_steps,
            sharpness=sharpness,
            grid_offset=grid_offset,
        )

    previous = _experiments.make_sharp_multimodal_likelihoods
    _experiments.make_sharp_multimodal_likelihoods = representation_aware_likelihoods
    try:
        yield
    finally:
        _experiments.make_sharp_multimodal_likelihoods = previous


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
    """Evaluate the paper smoothers with representation-correct projections."""

    with _cell_average_pwc_projection(pwc_quadrature_points):
        return _ORIGINAL_RUN_SMOOTHING_EVALUATION(
            figf_grid_sizes=figf_grid_sizes,
            pwc_grid_sizes=pwc_grid_sizes,
            pf_particle_counts=pf_particle_counts,
            repetitions=repetitions,
            time_steps=time_steps,
            likelihood_sharpness=likelihood_sharpness,
            noise_concentration=noise_concentration,
            l1_reference_grid_size=l1_reference_grid_size,
            mean_reference_particles=mean_reference_particles,
            mean_reference_repetitions=mean_reference_repetitions,
            particle_kde_bandwidth_scale=particle_kde_bandwidth_scale,
            pwc_quadrature_points=pwc_quadrature_points,
            seed=seed,
        )


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
    """Time the same representation-correct recursions used for paper errors."""

    with _cell_average_pwc_projection(pwc_quadrature_points):
        return _ORIGINAL_RUN_SMOOTHING_RUNTIME_EVALUATION(
            figf_grid_sizes=figf_grid_sizes,
            pwc_grid_sizes=pwc_grid_sizes,
            pf_particle_counts=pf_particle_counts,
            repetitions=repetitions,
            time_steps=time_steps,
            likelihood_sharpness=likelihood_sharpness,
            noise_concentration=noise_concentration,
            particle_likelihood_grid_size=particle_likelihood_grid_size,
            pwc_quadrature_points=pwc_quadrature_points,
            seed=seed,
        )


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
    """Compare FIGF outputs with a cell-average PWC smoother.

    Unlike the main evaluation, the historical diagnostic originally shared one
    likelihood array between FIGF and PWC. It therefore needs an explicit split:
    FIGF receives nodal values and PWC receives cell averages.
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
    reference_filtered = _run_figf_forward_filter(
        reference_likelihoods,
        reference_noise,
        reference_cell_volume,
    )
    reference_transition = TorusAdditiveGridTransition.for_grid_shape(
        reference_noise,
        reference_shape,
    )
    reference_smoothed = grid_backward_information_smoother(
        reference_filtered,
        reference_likelihoods,
        reference_transition,
        cell_volume=reference_cell_volume,
    ).smoothed

    rows: list[FIGFPWCBenchmarkRow] = []
    for grid_size_value in grid_sizes:
        grid_size = int(grid_size_value)
        if grid_size <= 0:
            raise ValueError("grid sizes must be positive.")
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        figf_likelihoods = make_sharp_multimodal_likelihoods(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
        )
        pwc_likelihoods = make_pwc_cell_averaged_likelihoods_1d(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
            quadrature_points=pwc_quadrature_points,
        )
        figf_noise = make_von_mises_like_noise(grid_shape, noise_concentration)
        figf_transition = TorusAdditiveGridTransition.for_grid_shape(figf_noise, grid_shape)
        pwc_transition = DenseGridTransition.for_grid_shape(
            make_pwc_additive_transition_density_matrix_1d(
                grid_size,
                noise_concentration,
                quadrature_points=pwc_quadrature_points,
            ),
            grid_shape,
            cell_volume=cell_volume,
        )

        for repetition in range(repetitions):
            start = time.perf_counter()
            figf_filtered = _run_figf_forward_filter(figf_likelihoods, figf_noise, cell_volume)
            figf_filter_runtime = time.perf_counter() - start

            start = time.perf_counter()
            figf_result = grid_backward_information_smoother(
                figf_filtered,
                figf_likelihoods,
                figf_transition,
                cell_volume=cell_volume,
            )
            figf_smoother_runtime = time.perf_counter() - start

            for method, evaluator in (
                ("FIGFAN", _evaluate_figfan_1d),
                ("FIGFDN", _evaluate_figfdn_1d),
            ):
                start = time.perf_counter()
                evaluated = evaluator(figf_result.smoothed, reference_grid_size)
                evaluation_runtime = time.perf_counter() - start
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

            start = time.perf_counter()
            pwc_filtered = _run_dense_transition_forward_filter(
                pwc_likelihoods,
                pwc_transition,
                cell_volume,
            )
            pwc_filter_runtime = time.perf_counter() - start

            start = time.perf_counter()
            pwc_result = grid_backward_information_smoother(
                pwc_filtered,
                pwc_likelihoods,
                pwc_transition,
                cell_volume=cell_volume,
            )
            pwc_smoother_runtime = time.perf_counter() - start

            start = time.perf_counter()
            pwc_evaluated = _evaluate_pwc_1d(pwc_result.smoothed, reference_grid_size)
            evaluation_runtime = time.perf_counter() - start
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
