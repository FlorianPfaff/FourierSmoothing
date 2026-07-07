"""Particle-smoothing baseline utilities on the one-dimensional torus.

These routines are intentionally lightweight and self-contained. They are meant
as paper baselines for the spectral smoother, not as a general-purpose SMC
library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class ParticleFilterResult:
    """Bootstrap particle-filter output for a one-dimensional torus model."""

    particles: NDArray[np.float64]  # shape: (T, n_particles)
    weights: NDArray[np.float64]  # shape: (T, n_particles)


@dataclass(frozen=True)
class ParticleSmoothingResult:
    """Backward-simulation output for a one-dimensional torus model."""

    trajectories: NDArray[np.float64]  # shape: (n_trajectories, T)
    mean_directions: NDArray[np.float64]  # shape: (T,)


def periodic_linear_interpolate_1d(values: ArrayLike, angles: ArrayLike) -> NDArray[np.float64]:
    """Linearly interpolate periodic grid values on ``[0, 2*pi)``."""

    grid_values = np.asarray(values, dtype=float)
    if grid_values.ndim != 1 or grid_values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    xs = np.mod(np.asarray(angles, dtype=float), TWOPI)
    positions = xs * grid_values.size / TWOPI
    left = np.floor(positions).astype(int) % grid_values.size
    right = (left + 1) % grid_values.size
    fraction = positions - np.floor(positions)
    return (1.0 - fraction) * grid_values[left] + fraction * grid_values[right]


def circular_mean(angles: ArrayLike, weights: ArrayLike | None = None, axis: int = 0) -> NDArray[np.float64]:
    """Weighted circular mean on the torus."""

    angle_array = np.asarray(angles, dtype=float)
    if weights is None:
        resultant = np.mean(np.exp(1j * angle_array), axis=axis)
    else:
        weight_array = np.asarray(weights, dtype=float)
        weight_sum = np.sum(weight_array, axis=axis)
        resultant = np.sum(weight_array * np.exp(1j * angle_array), axis=axis) / weight_sum
    return np.mod(np.angle(resultant), TWOPI)


def systematic_resample(weights: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    """Return systematic-resampling ancestor indices."""

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array.")
    w = _normalize_weights(w)
    positions = (rng.random() + np.arange(w.size)) / w.size
    cumulative = np.cumsum(w)
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def sample_from_grid_density_1d(
    density_values: ArrayLike,
    n_samples: int,
    rng: np.random.Generator,
    *,
    jitter_within_cell: bool = True,
) -> NDArray[np.float64]:
    """Sample approximately from a periodic density represented on an equidistant grid."""

    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative.")
    density = np.asarray(density_values, dtype=float)
    if density.ndim != 1 or density.size == 0:
        raise ValueError("density_values must be a non-empty one-dimensional array.")
    probabilities = _normalize_weights(np.maximum(density, 0.0))
    indices = rng.choice(density.size, size=n_samples, p=probabilities)
    offsets = rng.random(n_samples) if jitter_within_cell else np.zeros(n_samples)
    return TWOPI * (indices + offsets) / density.size


def bootstrap_particle_filter_1d(
    likelihoods: ArrayLike,
    noise_density: ArrayLike,
    n_particles: int,
    *,
    rng: np.random.Generator | int | None = None,
    resample: bool = True,
) -> ParticleFilterResult:
    """Bootstrap particle filter for ``x[t+1] = x[t] + w[t] mod 2*pi``.

    ``likelihoods[t]`` is evaluated on an equidistant grid. The initial prior is
    uniform on the torus.
    """

    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    rng = _as_rng(rng)
    likelihood_array = np.asarray(likelihoods, dtype=float)
    if likelihood_array.ndim != 2 or likelihood_array.shape[0] < 1:
        raise ValueError("likelihoods must have shape (T, n_grid).")
    noise = np.asarray(noise_density, dtype=float)
    if noise.ndim != 1 or noise.size != likelihood_array.shape[1]:
        raise ValueError("noise_density must be one-dimensional with the likelihood grid size.")

    time_steps = likelihood_array.shape[0]
    particles = np.empty((time_steps, n_particles), dtype=float)
    weights = np.empty((time_steps, n_particles), dtype=float)

    particles[0] = rng.uniform(0.0, TWOPI, size=n_particles)
    weights[0] = _normalize_weights(periodic_linear_interpolate_1d(likelihood_array[0], particles[0]))

    for t in range(1, time_steps):
        if resample:
            ancestor_indices = systematic_resample(weights[t - 1], rng)
        else:
            ancestor_indices = rng.choice(n_particles, size=n_particles, p=weights[t - 1])
        innovations = sample_from_grid_density_1d(noise, n_particles, rng)
        particles[t] = np.mod(particles[t - 1, ancestor_indices] + innovations, TWOPI)
        weights[t] = _normalize_weights(periodic_linear_interpolate_1d(likelihood_array[t], particles[t]))

    return ParticleFilterResult(particles=particles, weights=weights)


def ffbsi_particle_smoother_1d(
    filter_result: ParticleFilterResult,
    noise_density: ArrayLike,
    n_trajectories: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> ParticleSmoothingResult:
    """Forward-filtering backward-simulation smoother for the 1-D torus model."""

    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive.")
    rng = _as_rng(rng)
    noise = np.asarray(noise_density, dtype=float)
    if noise.ndim != 1 or noise.size == 0:
        raise ValueError("noise_density must be a non-empty one-dimensional array.")

    particles = np.asarray(filter_result.particles, dtype=float)
    weights = np.asarray(filter_result.weights, dtype=float)
    if particles.shape != weights.shape or particles.ndim != 2:
        raise ValueError("particles and weights must both have shape (T, n_particles).")

    time_steps, n_particles = particles.shape
    trajectories = np.empty((n_trajectories, time_steps), dtype=float)

    for trajectory_index in range(n_trajectories):
        idx = rng.choice(n_particles, p=_normalize_weights(weights[-1]))
        trajectories[trajectory_index, -1] = particles[-1, idx]
        next_state = trajectories[trajectory_index, -1]

        for t in range(time_steps - 2, -1, -1):
            transition_values = transition_density_from_noise_1d(noise, next_state - particles[t])
            backward_weights = weights[t] * transition_values
            idx = rng.choice(n_particles, p=_normalize_weights(backward_weights))
            next_state = particles[t, idx]
            trajectories[trajectory_index, t] = next_state

    return ParticleSmoothingResult(
        trajectories=trajectories,
        mean_directions=circular_mean(trajectories, axis=0),
    )


def transition_density_from_noise_1d(noise_density: ArrayLike, displacement: ArrayLike) -> NDArray[np.float64]:
    """Evaluate ``p_w(displacement mod 2*pi)`` from a grid density."""

    return periodic_linear_interpolate_1d(noise_density, np.mod(displacement, TWOPI))


def _normalize_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(w.shape, 1.0 / w.size)
    return w / total


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
