"""Sufficient-statistic helpers enabled by torus smoothing."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .smoother import cell_volume_for_grid, normalize_grid_density


def torus_increment_density_from_pairwise(
    pairwise_density: ArrayLike,
    grid_shape: Iterable[int],
    cell_volume: float,
) -> NDArray[np.float64]:
    """Map pairwise smoothed densities to transition-increment densities.

    For additive torus dynamics, this computes the quadrature approximation

    ``q_t[w] = integral p(x_t=x, x_{t+1}=x+w | z_1:T) dx``.

    ``pairwise_density`` may have shape ``(N, N)`` or ``(..., N, N)``. The last
    two axes use the convention ``[current_index, next_index]``. The returned
    array has shape ``grid_shape`` or ``(..., *grid_shape)`` and is normalized as
    a density under the supplied uniform ``cell_volume``.
    """

    shape = _grid_shape(grid_shape)
    volume = _positive_volume(cell_volume)

    density = _finite_nonnegative_array(pairwise_density, "pairwise_density")
    n_grid = int(np.prod(shape))
    if density.ndim < 2 or density.shape[-2:] != (n_grid, n_grid):
        raise ValueError(f"pairwise_density must end in shape {(n_grid, n_grid)}, got {density.shape}")

    coordinates = np.indices(shape).reshape(len(shape), -1)
    offsets = tuple(
        (coordinates[axis][None, :] - coordinates[axis][:, None]) % shape[axis]
        for axis in range(len(shape))
    )
    offset_flat = np.ravel_multi_index(offsets, shape).reshape(-1)

    leading_shape = density.shape[:-2]
    batches = density.reshape((-1, n_grid, n_grid))
    increments = np.zeros((batches.shape[0], n_grid), dtype=np.float64)
    for batch_index, joint in enumerate(batches):
        np.add.at(increments[batch_index], offset_flat, joint.reshape(-1) * volume)

    increment_grid = increments.reshape((*leading_shape, *shape))
    return _normalize_last_grid_axes(increment_grid, shape, volume)


def torus_increment_density_from_messages(
    filtered_density: ArrayLike,
    future_information: ArrayLike,
    noise_density: ArrayLike,
    *,
    cell_volume: float | None = None,
    normalize_noise: bool = True,
) -> NDArray[np.float64]:
    """Compute an additive-torus increment posterior without a pairwise matrix.

    Let ``f[j]`` be the filtering density at time ``t`` and ``u[i]`` the product
    of the next likelihood and backward message. For additive noise ``w``, the
    increment posterior satisfies

    ``q[r] proportional to p_w[r] * sum_j f[j] * u[j+r]``.

    The cyclic correlation is evaluated by FFT, giving ``O(N log N)`` time and
    ``O(N)`` working memory instead of materializing the ``N x N`` pairwise
    density. The result is normalized as a density on the supplied grid.
    """

    filtered = _finite_nonnegative_array(filtered_density, "filtered_density")
    future = _finite_nonnegative_array(future_information, "future_information")
    noise = _finite_nonnegative_array(noise_density, "noise_density")
    if filtered.shape != future.shape or filtered.shape != noise.shape:
        raise ValueError(
            "filtered_density, future_information, and noise_density must have identical grid shapes"
        )
    if filtered.ndim < 1:
        raise ValueError("grid arrays must have at least one dimension")

    shape = tuple(int(value) for value in filtered.shape)
    volume = cell_volume_for_grid(shape) if cell_volume is None else _positive_volume(cell_volume)
    if normalize_noise:
        noise = normalize_grid_density(noise, volume)

    correlation = np.fft.ifftn(
        np.conj(np.fft.fftn(filtered)) * np.fft.fftn(future)
    ).real
    correlation = np.maximum(correlation * volume, 0.0)
    unnormalized = noise * correlation
    integral = float(np.sum(unnormalized) * volume)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("increment posterior must have positive finite integral")
    return unnormalized / integral


def torus_increment_densities_from_smoother(
    filtered: ArrayLike,
    likelihoods: ArrayLike,
    backward_messages: ArrayLike,
    noise_density: ArrayLike,
    *,
    cell_volume: float | None = None,
    normalize_noise: bool = True,
) -> NDArray[np.float64]:
    """Compute all additive-noise increment posteriors from smoother messages.

    ``noise_density`` may have shape ``grid_shape`` or
    ``(T-1, *grid_shape)``. The output has shape ``(T-1, *grid_shape)`` and can
    be averaged to form the nonparametric M-step update for a time-invariant
    process-noise density.
    """

    f = _finite_nonnegative_array(filtered, "filtered")
    ell = _finite_nonnegative_array(likelihoods, "likelihoods")
    beta = _finite_nonnegative_array(backward_messages, "backward_messages")
    if f.ndim < 2 or f.shape[0] < 1:
        raise ValueError("filtered must have shape (T, *grid_shape)")
    if ell.shape != f.shape or beta.shape != f.shape:
        raise ValueError("filtered, likelihoods, and backward_messages must have identical shapes")

    grid_shape = tuple(int(value) for value in f.shape[1:])
    volume = cell_volume_for_grid(grid_shape) if cell_volume is None else _positive_volume(cell_volume)
    noise = _finite_nonnegative_array(noise_density, "noise_density")
    if noise.shape == grid_shape:
        time_varying = False
    elif noise.shape == (max(f.shape[0] - 1, 0), *grid_shape):
        time_varying = True
    else:
        raise ValueError(
            f"noise_density must have shape {grid_shape} or {(max(f.shape[0] - 1, 0), *grid_shape)}, "
            f"got {noise.shape}"
        )

    increments = np.empty((max(f.shape[0] - 1, 0), *grid_shape), dtype=np.float64)
    for t in range(f.shape[0] - 1):
        future_information = ell[t + 1] * beta[t + 1]
        noise_t = noise[t] if time_varying else noise
        increments[t] = torus_increment_density_from_messages(
            f[t],
            future_information,
            noise_t,
            cell_volume=volume,
            normalize_noise=normalize_noise,
        )
    return increments


def average_increment_density(increment_densities: ArrayLike, cell_volume: float) -> NDArray[np.float64]:
    """Average time-indexed increment densities and renormalize the result."""

    densities = _finite_nonnegative_array(increment_densities, "increment_densities")
    if densities.ndim < 2 or densities.shape[0] < 1:
        raise ValueError("increment_densities must have shape (T-1, *grid_shape)")
    averaged = np.mean(densities, axis=0)
    integral = float(np.sum(averaged) * _positive_volume(cell_volume))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("average increment density must have positive finite integral")
    return averaged / integral


def _normalize_last_grid_axes(
    values: NDArray[np.float64], grid_shape: tuple[int, ...], cell_volume: float
) -> NDArray[np.float64]:
    integration_axes = tuple(range(values.ndim - len(grid_shape), values.ndim))
    integrals = np.sum(values, axis=integration_axes, keepdims=True) * cell_volume
    if not np.all(np.isfinite(integrals)) or np.any(integrals <= 0.0):
        raise ValueError("increment density must have positive finite integral")
    return values / integrals


def _grid_shape(grid_shape: Iterable[int]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in grid_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("grid_shape must contain positive dimensions")
    return shape


def _positive_volume(cell_volume: float) -> float:
    volume = float(cell_volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("cell_volume must be positive and finite")
    return volume


def _finite_nonnegative_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array
