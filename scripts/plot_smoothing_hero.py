#!/usr/bin/env python
"""Create a space-time smoothing illustration for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fourier_smoothing import (
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    grid_backward_information_smoother,
    make_von_mises_like_noise,
    normalize_grid_density,
    torus_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("../2026-07-FourierSmoothing-Paper/figures"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--grid-size", type=int, default=255)
    parser.add_argument("--time-steps", type=int, default=9)
    parser.add_argument("--likelihood-sharpness", type=float, default=7.0)
    parser.add_argument("--noise-concentration", type=float, default=2.2)
    parser.add_argument("--prior-concentration", type=float, default=0.45)
    parser.add_argument("--filename", default="smoothing_space_time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    filtered, backward, smoothed = _make_smoothing_case(
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        likelihood_sharpness=args.likelihood_sharpness,
        noise_concentration=args.noise_concentration,
        prior_concentration=args.prior_concentration,
    )
    written = _plot_space_time_smoothing(
        filtered,
        backward,
        smoothed,
        args.figures_dir / args.filename,
        args.formats,
    )
    for path in written:
        print(path)


def _make_smoothing_case(
    *,
    grid_size: int,
    time_steps: int,
    likelihood_sharpness: float,
    noise_concentration: float,
    prior_concentration: float,
):
    grid_shape = (int(grid_size),)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods = _make_branching_likelihoods(grid_shape, int(time_steps), likelihood_sharpness)
    noise = make_von_mises_like_noise(grid_shape, noise_concentration)
    initial_prior = _make_broad_initial_prior(grid_shape, prior_concentration)
    filtered = _run_forward_filter(likelihoods, noise, cell_volume, initial_prior)
    transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
    result = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)
    return filtered, result.backward_messages, result.smoothed


def _make_branching_likelihoods(grid_shape: tuple[int], time_steps: int, sharpness: float) -> np.ndarray:
    if time_steps < 2:
        raise ValueError("time_steps must be at least two.")
    if sharpness <= 0.0:
        raise ValueError("likelihood-sharpness must be positive.")

    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    times = np.arange(time_steps, dtype=float)
    progress = times / max(float(time_steps - 1), 1.0)

    true_centers = _wrap_to_2pi(-1.15 + 2.45 * progress)
    distractor_centers = _wrap_to_2pi(1.35 - 0.62 * progress + 0.22 * np.sin(1.4 * np.pi * progress))
    true_weights = 0.70 + 1.25 * progress**1.6
    distractor_weights = 1.15 * (1.0 - 0.88 * progress**1.25)

    likelihoods = []
    for true_center, distractor_center, true_weight, distractor_weight in zip(
        true_centers, distractor_centers, true_weights, distractor_weights
    ):
        values = (
            0.025
            + true_weight * _von_mises_bump(x, true_center, sharpness)
            + distractor_weight * _von_mises_bump(x, distractor_center, 0.93 * sharpness)
        )
        likelihoods.append(normalize_grid_density(values, cell_volume))
    return np.stack(likelihoods, axis=0)


def _make_broad_initial_prior(grid_shape: tuple[int], concentration: float) -> np.ndarray:
    if concentration <= 0.0:
        raise ValueError("prior-concentration must be positive.")

    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    values = 0.70 + _von_mises_bump(x, _wrap_to_2pi(-0.35), concentration)
    return normalize_grid_density(values, cell_volume)


def _von_mises_bump(x: np.ndarray, center: float, concentration: float) -> np.ndarray:
    return np.exp(concentration * np.cos(x - center))


def _wrap_to_2pi(angle):
    return np.mod(angle, 2.0 * np.pi)


def _run_forward_filter(
    likelihoods: np.ndarray, noise: np.ndarray, cell_volume: float, initial_prior: np.ndarray
) -> np.ndarray:
    filtered = []
    current = normalize_grid_density(initial_prior * likelihoods[0], cell_volume)
    filtered.append(current)
    for likelihood in likelihoods[1:]:
        predicted = np.fft.ifft(np.fft.fft(noise) * np.fft.fft(current)).real
        predicted = normalize_grid_density(np.maximum(predicted * cell_volume, 0.0), cell_volume)
        current = normalize_grid_density(predicted * likelihood, cell_volume)
        filtered.append(current)
    return np.stack(filtered, axis=0)


def _plot_space_time_smoothing(
    filtered: np.ndarray,
    backward: np.ndarray,
    smoothed: np.ndarray,
    output_base: Path,
    formats: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    panels = [
        ("Filtered density", _column_normalize(filtered), filtered),
        ("Backward message", _column_normalize(backward), None),
        ("Smoothed density", _column_normalize(smoothed), smoothed),
    ]
    time_steps = filtered.shape[0]
    extent = (-0.5, time_steps - 0.5, -np.pi, np.pi)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharey=True, constrained_layout=True)
    image = None
    for ax, (title, values, mean_source) in zip(axes, panels):
        image = ax.imshow(
            _center_angle_axis(values).T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        if mean_source is not None:
            _plot_circular_mean(ax, mean_source)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time step")
        ax.set_xticks(range(time_steps))
        ax.tick_params(axis="both", labelsize=8)

    axes[0].set_ylabel("angle")
    axes[0].set_yticks([-np.pi, 0.0, np.pi])
    axes[0].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015)
    colorbar.set_label("relative contrast", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    written = []
    for fmt in formats:
        output_path = output_base.with_suffix(f".{fmt}")
        fig.savefig(output_path, bbox_inches="tight")
        written.append(output_path)
    plt.close(fig)
    return written


def _column_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    arr = arr - np.min(arr, axis=1, keepdims=True)
    maxima = np.max(arr, axis=1, keepdims=True)
    maxima[maxima <= 0.0] = 1.0
    return arr / maxima


def _center_angle_axis(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(values, axes=1)


def _plot_circular_mean(ax, density: np.ndarray) -> None:
    grid_size = density.shape[1]
    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    moments = np.sum(density * np.exp(1j * angles)[None, :], axis=1)
    means = np.mod(np.angle(moments) + np.pi, 2.0 * np.pi) - np.pi
    times = np.arange(density.shape[0], dtype=float)
    plot_times, plot_means = _break_wrapped_line(times, means)
    ax.plot(plot_times, plot_means, color="white", linewidth=1.2, alpha=0.95)
    ax.plot(plot_times, plot_means, color="black", linewidth=0.35, alpha=0.65)


def _break_wrapped_line(times: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plot_times = [times[0]]
    plot_angles = [angles[0]]
    for previous_time, current_time, previous_angle, current_angle in zip(times[:-1], times[1:], angles[:-1], angles[1:]):
        if abs(current_angle - previous_angle) > np.pi:
            plot_times.extend([(previous_time + current_time) / 2.0, (previous_time + current_time) / 2.0])
            plot_angles.extend([np.nan, np.nan])
        plot_times.append(current_time)
        plot_angles.append(current_angle)
    return np.asarray(plot_times), np.asarray(plot_angles)


if __name__ == "__main__":
    main()
