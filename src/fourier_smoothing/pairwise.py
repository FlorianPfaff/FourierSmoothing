"""Pairwise smoothed marginals for grid-based torus smoothers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .smoother import cell_volume_for_grid, normalize_grid_density


@dataclass(frozen=True)
class GridPairwiseSmoothingResult:
    """Pairwise fixed-interval smoothing output.

    ``joint[t, j, i]`` approximates
    ``p(x_t=x_j, x_{t+1}=x_i | z_1, ..., z_T)``. Grid dimensions are flattened
    into the ``j`` and ``i`` axes. Integrating either axis with ``cell_volume``
    recovers the corresponding one-time smoothed marginal when the supplied
    filtering sequence is consistent with the transition model.
    """

    joint: NDArray[np.float64]
    normalizers: NDArray[np.float64]
    grid_shape: tuple[int, ...]
    cell_volume: float


def grid_pairwise_smoothed_marginals(
    filtered: ArrayLike,
    likelihoods: ArrayLike,
    backward_messages: ArrayLike,
    transition_density: ArrayLike,
    *,
    cell_volume: float | None = None,
    normalize_transition_columns: bool = True,
) -> GridPairwiseSmoothingResult:
    """Compute pairwise smoothed marginals on a uniform grid.

    Parameters
    ----------
    filtered
        Filtering posteriors with shape ``(T, *grid_shape)``.
    likelihoods
        Likelihood values with the same shape as ``filtered``.
    backward_messages
        Backward information messages returned by the grid smoother.
    transition_density
        A matrix with shape ``(N, N)`` or a sequence with shape
        ``(T-1, N, N)``. Entry ``[i, j]`` is the transition density from current
        grid point ``j`` to next grid point ``i``.
    cell_volume
        Uniform quadrature weight. It defaults to the torus cell volume implied
        by ``grid_shape``.
    normalize_transition_columns
        Normalize each transition-density column under the quadrature rule.

    Notes
    -----
    The output requires ``O(T N^2)`` memory. This is intended for diagnostics,
    expectation-maximization sufficient statistics, and moderate grid sizes.
    """

    f = _finite_nonnegative_array(filtered, "filtered")
    ell = _finite_nonnegative_array(likelihoods, "likelihoods")
    beta = _finite_nonnegative_array(backward_messages, "backward_messages")
    if f.ndim < 2 or f.shape[0] < 1:
        raise ValueError("filtered must have shape (T, *grid_shape)")
    if ell.shape != f.shape or beta.shape != f.shape:
        raise ValueError("filtered, likelihoods, and backward_messages must have identical shapes")

    time_steps = f.shape[0]
    grid_shape = tuple(int(value) for value in f.shape[1:])
    n_grid = int(np.prod(grid_shape))
    if cell_volume is None:
        cell_volume = cell_volume_for_grid(grid_shape)
    cell_volume = float(cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError("cell_volume must be positive and finite")

    transition = _finite_nonnegative_array(transition_density, "transition_density")
    if transition.shape == (n_grid, n_grid):
        time_varying = False
    elif transition.shape == (max(time_steps - 1, 0), n_grid, n_grid):
        time_varying = True
    else:
        raise ValueError(
            "transition_density must have shape "
            f"{(n_grid, n_grid)} or {(max(time_steps - 1, 0), n_grid, n_grid)}, got {transition.shape}"
        )

    joint = np.empty((max(time_steps - 1, 0), n_grid, n_grid), dtype=np.float64)
    normalizers = np.empty(max(time_steps - 1, 0), dtype=np.float64)

    for t in range(time_steps - 1):
        matrix = transition[t] if time_varying else transition
        if normalize_transition_columns:
            matrix = _normalize_transition_matrix(matrix, cell_volume)

        current = f[t].reshape(-1)
        future_information = (ell[t + 1] * beta[t + 1]).reshape(-1)
        unnormalized = current[:, None] * matrix.T * future_information[None, :]
        normalizer = float(np.sum(unnormalized) * cell_volume**2)
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError(f"pairwise normalizer at time {t} must be positive and finite, got {normalizer}")
        joint[t] = unnormalized / normalizer
        normalizers[t] = normalizer

    return GridPairwiseSmoothingResult(
        joint=joint,
        normalizers=normalizers,
        grid_shape=grid_shape,
        cell_volume=cell_volume,
    )


def torus_additive_transition_density_matrix(
    noise_density: ArrayLike,
    *,
    cell_volume: float | None = None,
    normalize_noise: bool = True,
) -> NDArray[np.float64]:
    """Build ``K[i,j] = p_w(x_i-x_j)`` for an equidistant torus grid.

    The returned dense matrix is useful for pairwise marginals and independent
    validation of the FFT correlation implementation. It requires ``O(N^2)``
    memory and should therefore be restricted to moderate grids.
    """

    noise = _finite_nonnegative_array(noise_density, "noise_density")
    if noise.ndim < 1:
        raise ValueError("noise_density must have at least one dimension")
    grid_shape = tuple(int(value) for value in noise.shape)
    if cell_volume is None:
        cell_volume = cell_volume_for_grid(grid_shape)
    cell_volume = float(cell_volume)
    if normalize_noise:
        noise = normalize_grid_density(noise, cell_volume)

    coordinates = np.indices(grid_shape).reshape(len(grid_shape), -1)
    offsets = tuple(
        (coordinates[axis][:, None] - coordinates[axis][None, :]) % grid_shape[axis]
        for axis in range(len(grid_shape))
    )
    return np.asarray(noise[offsets], dtype=np.float64)


def pairwise_current_marginal(pairwise_density: ArrayLike, cell_volume: float) -> NDArray[np.float64]:
    """Integrate a flattened pairwise density over its next-state axis."""

    density = _finite_nonnegative_array(pairwise_density, "pairwise_density")
    if density.ndim < 2:
        raise ValueError("pairwise_density must have at least two dimensions")
    return np.sum(density, axis=-1) * float(cell_volume)


def pairwise_next_marginal(pairwise_density: ArrayLike, cell_volume: float) -> NDArray[np.float64]:
    """Integrate a flattened pairwise density over its current-state axis."""

    density = _finite_nonnegative_array(pairwise_density, "pairwise_density")
    if density.ndim < 2:
        raise ValueError("pairwise_density must have at least two dimensions")
    return np.sum(density, axis=-2) * float(cell_volume)


def _normalize_transition_matrix(matrix: NDArray[np.float64], cell_volume: float) -> NDArray[np.float64]:
    column_integrals = np.sum(matrix, axis=0, keepdims=True) * cell_volume
    if not np.all(np.isfinite(column_integrals)) or np.any(column_integrals <= 0.0):
        raise ValueError("transition-density columns must have positive finite integrals")
    return matrix / column_integrals


def _finite_nonnegative_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array
