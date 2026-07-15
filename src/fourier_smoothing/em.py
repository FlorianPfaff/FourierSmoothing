"""Sufficient-statistic helpers enabled by pairwise torus smoothing."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


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

    shape = tuple(int(value) for value in grid_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("grid_shape must contain positive dimensions")
    volume = float(cell_volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("cell_volume must be positive and finite")

    density = np.asarray(pairwise_density, dtype=np.float64)
    n_grid = int(np.prod(shape))
    if density.ndim < 2 or density.shape[-2:] != (n_grid, n_grid):
        raise ValueError(f"pairwise_density must end in shape {(n_grid, n_grid)}, got {density.shape}")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("pairwise_density must be finite and nonnegative")

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
    integration_axes = tuple(range(increment_grid.ndim - len(shape), increment_grid.ndim))
    integrals = np.sum(increment_grid, axis=integration_axes, keepdims=True) * volume
    if not np.all(np.isfinite(integrals)) or np.any(integrals <= 0.0):
        raise ValueError("increment density must have positive finite integral")
    return increment_grid / integrals


def average_increment_density(increment_densities: ArrayLike, cell_volume: float) -> NDArray[np.float64]:
    """Average time-indexed increment densities and renormalize the result."""

    densities = np.asarray(increment_densities, dtype=np.float64)
    if densities.ndim < 2 or densities.shape[0] < 1:
        raise ValueError("increment_densities must have shape (T-1, *grid_shape)")
    if not np.all(np.isfinite(densities)) or np.any(densities < 0.0):
        raise ValueError("increment_densities must be finite and nonnegative")
    averaged = np.mean(densities, axis=0)
    integral = float(np.sum(averaged) * float(cell_volume))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("average increment density must have positive finite integral")
    return averaged / integral
