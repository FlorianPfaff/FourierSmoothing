"""Representation-faithful runners for the Fourier-smoothing paper benchmarks.

The FIGF stores point values, whereas the PWC baseline stores one constant
value per interval cell. Consequently, the analytic likelihood must be sampled
at FIGF nodes but averaged over PWC cells. The original benchmark runner used
midpoint samples for the PWC likelihood. This module keeps the public benchmark
API while applying the correct cell-average projection.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .experiments import (
    FIGFPWCBenchmarkRow,
    SmoothingEvaluationRow,
    SmoothingRuntimeRow,
    _evaluation_row_with_timing,
    _evaluate_figfan_1d,
    _evaluate_figfdn_1d,
    _evaluate_pwc_1d,
    _make_figf_pwc_row,
    _make_smoothing_evaluation_row,
    _particle_trajectories_to_wrapped_normal_kde_1d,
    _positive_int_tuple,
    _pwc_backward_predict_fft,
    _pwc_circular_means_1d,
    _run_figf_forward_filter,
    _run_pwc_forward_filter,
    _run_pwc_smoother_1d,
    _run_von_mises_ffbsi_1d,
    make_pwc_additive_transition_density_matrix_1d,
    make_pwc_additive_transition_kernel_1d,
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


def make_pwc_cell_averaged_likelihoods_1d(
    grid_shape: Sequence[int],
    time_steps: int,
    *,
    sharpness: float,
    quadrature_points: int = 8,
) -> NDArray[np.float64]:
    """Project the paper likelihood onto equal-width PWC cells.

    The returned value in cell ``j`` is a midpoint-quadrature approximation to

    ``(1 / cell_width) * integral_{cell j} likelihood_t(x) dx``.

    This is the correct measurement-update factor when the prior density is
    represented as constant inside each cell. It differs from evaluating the
    likelihood only at the cell midpoint, particularly on coarse grids.
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

    FIGF likelihoods are evaluated at grid nodes. PWC likelihoods and
    transitions are averaged over their cells with midpoint quadrature.
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
            figf_filtered = _run_figf_forward_filter(figf_likelihoods, figf_noise, cell_volume)
            figf_filter_runtime = time.perf_counter() - figf_filter_start

            figf_smoother_start = time.perf_counter()
            figf_result = grid_backward_information_smoother(
                figf_filtered,
                figf_likelihoods,
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
            pwc_filtered = _run_dense_transition_forward_filter(
                pwc_likelihoods,
                pwc_transition,
                cell_volume,
            )
            pwc_filter_runtime = time.perf_counter() - pwc_filter_start

            pwc_smoother_start = time.perf_counter()
            pwc_result = grid_backward_information_smoother(
                pwc_filtered,
                pwc_likelihoods,
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
    """Evaluate the paper smoothers with representation-correct projections."""

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
    if pwc_quadrature_points < 1:
        raise ValueError("pwc_quadrature_points must be at least one.")

    reference_shape = (int(l1_reference_grid_size),)
    reference_cell_volume = cell_volume_for_grid(reference_shape)
    reference_likelihoods = make_sharp_multimodal_likelihoods(
        reference_shape,
        time_steps,
        sharpness=likelihood_sharpness,
    )
    reference_pwc_likelihoods = make_pwc_cell_averaged_likelihoods_1d(
        reference_shape,
        time_steps,
        sharpness=likelihood_sharpness,
        quadrature_points=pwc_quadrature_points,
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
        reference_moments.append(
            np.mean(np.exp(1j * reference_smoother.trajectories), axis=0)
        )
    mean_reference = np.mod(np.angle(np.mean(reference_moments, axis=0)), 2.0 * np.pi)

    rows: list[SmoothingEvaluationRow] = []
    seed_sequence = np.random.SeedSequence(seed + 1000)
    pf_seeds = iter(seed_sequence.spawn(len(pf_particle_counts) * repetitions))

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
            result = grid_backward_information_smoother(
                filtered,
                likelihoods,
                transition,
                cell_volume=cell_volume,
            )
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
        likelihoods = make_pwc_cell_averaged_likelihoods_1d(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
            quadrature_points=pwc_quadrature_points,
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
    """Time the same representation-correct recursions used for paper errors."""

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
            grid_backward_information_smoother(
                filtered,
                likelihoods,
                transition,
                cell_volume=cell_volume,
            )
            runtime = time.perf_counter() - start
            rows.append(SmoothingRuntimeRow("FIGFAN", grid_size, repetition, runtime))
            rows.append(SmoothingRuntimeRow("FIGFDN", grid_size, repetition, runtime))

    for grid_size in pwc_grid_sizes:
        grid_shape = (grid_size,)
        cell_volume = cell_volume_for_grid(grid_shape)
        likelihoods = make_pwc_cell_averaged_likelihoods_1d(
            grid_shape,
            time_steps,
            sharpness=likelihood_sharpness,
            quadrature_points=pwc_quadrature_points,
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
