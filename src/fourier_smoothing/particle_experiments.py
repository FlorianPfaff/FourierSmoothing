"""Particle-smoother baseline experiments for the Fourier smoothing paper."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .experiments import filtered_from_likelihoods, make_identity_likelihoods, make_von_mises_like_noise
from .particle import bootstrap_particle_filter_1d, circular_mean, ffbsi_particle_smoother_1d
from .smoother import TorusAdditiveGridTransition, cell_volume_for_grid, grid_backward_information_smoother, torus_grid


@dataclass(frozen=True)
class ParticleBaselineRow:
    """One CSV-ready particle-smoothing baseline row."""

    n_particles: int
    n_trajectories: int
    repetition: int
    runtime_s: float
    mean_abs_circular_error_to_grid: float
    max_abs_circular_error_to_grid: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "n_particles": self.n_particles,
            "n_trajectories": self.n_trajectories,
            "repetition": self.repetition,
            "runtime_s": self.runtime_s,
            "mean_abs_circular_error_to_grid": self.mean_abs_circular_error_to_grid,
            "max_abs_circular_error_to_grid": self.max_abs_circular_error_to_grid,
        }


PARTICLE_BASELINE_CSV_COLUMNS = [
    "n_particles",
    "n_trajectories",
    "repetition",
    "runtime_s",
    "mean_abs_circular_error_to_grid",
    "max_abs_circular_error_to_grid",
]


def run_particle_baseline_benchmark(
    n_particles_values: Iterable[int] = (100, 300, 1000),
    *,
    n_trajectories: int = 200,
    repetitions: int = 5,
    grid_size: int = 257,
    time_steps: int = 4,
    noise_concentration: float = 3.0,
    seed: int = 1,
) -> list[ParticleBaselineRow]:
    """Compare a torus FFBSi particle smoother to a dense-grid smoother."""

    n_particles_values = tuple(int(value) for value in n_particles_values)
    if len(n_particles_values) == 0:
        raise ValueError("n_particles_values must contain at least one entry.")
    if any(value <= 0 for value in n_particles_values):
        raise ValueError("n_particles values must be positive.")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive.")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    grid_shape = (int(grid_size),)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods = make_identity_likelihoods(grid_shape, time_steps)
    filtered = filtered_from_likelihoods(likelihoods, cell_volume)
    noise = make_von_mises_like_noise(grid_shape, noise_concentration)
    transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
    grid_smoothed = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)
    (x_grid,) = torus_grid(grid_shape)
    reference_mean = circular_mean(x_grid[None, :], weights=grid_smoothed.smoothed, axis=1)

    rows: list[ParticleBaselineRow] = []
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(seed_sequence.spawn(len(n_particles_values) * repetitions))

    for n_particles in n_particles_values:
        for repetition in range(repetitions):
            rng = np.random.default_rng(next(child_seeds))
            start = time.perf_counter()
            particle_filter = bootstrap_particle_filter_1d(likelihoods, noise, n_particles, rng=rng)
            particle_smoother = ffbsi_particle_smoother_1d(
                particle_filter,
                noise,
                n_trajectories,
                rng=rng,
            )
            runtime = time.perf_counter() - start
            errors = circular_abs_difference(particle_smoother.mean_directions, reference_mean)
            rows.append(
                ParticleBaselineRow(
                    n_particles=n_particles,
                    n_trajectories=n_trajectories,
                    repetition=repetition,
                    runtime_s=runtime,
                    mean_abs_circular_error_to_grid=float(np.mean(errors)),
                    max_abs_circular_error_to_grid=float(np.max(errors)),
                )
            )
    return rows


def circular_abs_difference(left, right) -> np.ndarray:
    """Smallest absolute angular difference between two angle arrays."""

    difference = np.mod(np.asarray(left) - np.asarray(right) + np.pi, 2.0 * np.pi) - np.pi
    return np.abs(difference)


def write_particle_baseline_csv(rows: Sequence[ParticleBaselineRow], output_path: str | Path) -> Path:
    """Write particle baseline rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICLE_BASELINE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path
