#!/usr/bin/env python
"""Reproduce a representative state-truth FIGF filtering/smoothing example.

Run in FlorianPfaff/FourierSmoothing, after `pip install -e '.[paper]'`.
The simulation and RNG draw order follow run_smoothing_gain_evaluation at
revision 4f210310306daaee115f7abedf2f435fca34cefb. Original repository functions
perform the filtering, smoothing and FIGFDN density reconstruction.

Default selection: closest trial to the componentwise medians of filtering and
smoothing MAEs, with both distances divided by their interquartile ranges.
MAEs exclude the final time, where filtering and smoothing coincide.
The density snapshot maximizes the filtered secondary/primary peak-height
ratio among peaks separated by >= 0.75 rad, excluding the terminal time.
Neither selection rule optimizes the observed smoothing gain.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fourier_smoothing.smoother import (
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    grid_backward_information_smoother,
    normalize_grid_density,
    torus_grid,
)
from fourier_smoothing.experiments import (
    _density_circular_means_1d,
    _evaluate_figfan_1d,
    _evaluate_figfdn_1d,
    _run_figf_forward_filter_with_prior,
    make_von_mises_like_noise,
)

UPSTREAM_REVISION = "4f210310306daaee115f7abedf2f435fca34cefb"
EXPECTED_PUBLISHED_MAES = np.array([0.29729433864255206, 0.1921700633147927])


@dataclass(frozen=True)
class Config:
    n_trials: int = 500
    grid_size: int = 1023
    time_steps: int = 20
    prior_concentration: float = 1.0
    noise_concentration: float = 8.0
    likelihood_concentration: float = 12.0
    outlier_probability: float = 0.3
    outlier_offset: float = 2.35
    seed: int = 21

    def validate(self) -> None:
        if self.n_trials < 1 or self.time_steps < 2 or self.grid_size < 3:
            raise ValueError("Need n_trials >= 1, time_steps >= 2 and grid_size >= 3.")
        if self.grid_size % 2 == 0:
            raise ValueError("Use an odd grid size to retain the Fourier interpolation convention.")
        kappas = (self.prior_concentration, self.noise_concentration,
                  self.likelihood_concentration)
        if not all(np.isfinite(k) and k > 0 for k in kappas):
            raise ValueError("Concentrations must be finite and positive.")
        if not np.isfinite(self.outlier_probability) or not 0 <= self.outlier_probability <= 1:
            raise ValueError("outlier_probability must lie in [0, 1].")
        if not np.isfinite(self.outlier_offset):
            raise ValueError("outlier_offset must be finite.")


def wrap(angle: NDArray | float) -> NDArray:
    """Return angles in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def simulate_trial(rng: np.random.Generator, c: Config) -> dict[str, NDArray]:
    """Preserve the upstream simulator's RNG calls and draw order exactly."""
    states = np.empty(c.time_steps, dtype=float)
    states[0] = np.mod(rng.vonmises(0.0, c.prior_concentration), 2.0 * np.pi)
    states[1:] = np.mod(
        states[0] + np.cumsum(rng.vonmises(0.0, c.noise_concentration, size=c.time_steps - 1)),
        2.0 * np.pi,
    )
    shifted_components = rng.random(c.time_steps) < c.outlier_probability
    measurement_noise = rng.vonmises(0.0, c.likelihood_concentration, size=c.time_steps)
    measurement_noise += shifted_components * c.outlier_offset
    measurements = np.mod(states + measurement_noise, 2.0 * np.pi)
    return dict(states=states, measurements=measurements,
                shifted_components=shifted_components)


def run_trial(observations: dict[str, NDArray], c: Config) -> dict[str, NDArray]:
    """Use the repository's filter, backward-information smoother and means."""
    (x,) = torus_grid((c.grid_size,))
    omega = cell_volume_for_grid((c.grid_size,))
    prior = normalize_grid_density(np.exp(c.prior_concentration * np.cos(x)), omega)
    noise = make_von_mises_like_noise((c.grid_size,), c.noise_concentration)
    likelihoods = []
    for measurement in observations["measurements"]:
        residual = measurement - x
        values = (
            (1.0 - c.outlier_probability)
            * np.exp(c.likelihood_concentration * np.cos(residual))
            + c.outlier_probability
            * np.exp(c.likelihood_concentration * np.cos(residual - c.outlier_offset))
        )
        likelihoods.append(normalize_grid_density(values, omega))
    likelihoods = np.stack(likelihoods)
    filtered = _run_figf_forward_filter_with_prior(likelihoods, noise, prior, omega)
    transition = TorusAdditiveGridTransition.for_grid_shape(noise, (c.grid_size,))
    result = grid_backward_information_smoother(filtered, likelihoods, transition,
                                               cell_volume=omega)
    filter_means = _density_circular_means_1d(filtered, omega)
    smoother_means = _density_circular_means_1d(result.smoothed, omega)
    return dict(**observations, grid=x, likelihoods=likelihoods, filtered=filtered,
                smoothed=result.smoothed, backward=result.backward_messages,
                filter_means=filter_means, smoother_means=smoother_means)


def trial_errors(result: dict[str, NDArray]) -> NDArray:
    return np.stack([
        np.abs(wrap(result["filter_means"] - result["states"])),
        np.abs(wrap(result["smoother_means"] - result["states"])),
    ], axis=-1)


def representative_trial(maes: NDArray) -> tuple[int, NDArray]:
    """Two-dimensional median-nearest rule; stable ties favor the first row."""
    if maes.ndim != 2 or maes.shape[0] == 0 or maes.shape[1] != 2 or not np.all(np.isfinite(maes)):
        raise ValueError("maes must be a finite n-by-2 array.")
    median = np.median(maes, axis=0)
    iqr = np.percentile(maes, 75, axis=0) - np.percentile(maes, 25, axis=0)
    scale = np.where(iqr > np.finfo(float).eps, iqr, 1.0)
    scores = np.sum(((maes - median) / scale) ** 2, axis=1)
    return int(np.argmin(scores)), scores


def separated_mode_ratio(density: NDArray, grid: NDArray,
                         min_separation: float = 0.75) -> float:
    """Secondary/primary local-peak height on a periodic grid (not mode mass)."""
    p = np.asarray(density)
    peaks = np.flatnonzero((p > np.roll(p, 1)) & (p >= np.roll(p, -1)))
    if peaks.size < 2:
        return 0.0
    primary = int(peaks[np.argmax(p[peaks])])
    separated = peaks[np.abs(wrap(grid[peaks] - grid[primary])) >= min_separation]
    if separated.size == 0 or p[primary] <= 0:
        return 0.0
    return float(np.max(p[separated]) / p[primary])


def choose_snapshot(result: dict[str, NDArray]) -> tuple[int, NDArray]:
    ratios = np.array([separated_mode_ratio(p, result["grid"])
                       for p in result["filtered"][:-1]])
    return int(np.argmax(ratios)), ratios


def bootstrap_reduction(errors: NDArray, repetitions: int = 2000,
                        seed: int = 22) -> dict:
    trial_maes = np.mean(errors[:, :-1], axis=1)
    means = np.mean(trial_maes, axis=0)
    indices = np.random.default_rng(seed).integers(0, len(errors),
                                                 size=(repetitions, len(errors)))
    resampled = np.mean(trial_maes[indices], axis=1)
    reductions = 100 * (1 - resampled[:, 1] / resampled[:, 0])
    return dict(filter_mae_rad=float(means[0]), smoother_mae_rad=float(means[1]),
                reduction_percent=float(100 * (1 - means[1] / means[0])),
                reduction_ci95_percent=np.quantile(reductions, [0.025, 0.975]).tolist())


def prepare_data(c: Config, output: Path, trial: int | None = None,
                 snapshot: int | None = None) -> dict:
    """Re-run the full study, select a typical trial, save all numerical evidence."""
    c.validate()
    if trial is not None and not 0 <= trial < c.n_trials:
        raise ValueError("Selected trial index is outside the simulated ensemble.")
    if snapshot is not None and not 0 <= snapshot < c.time_steps - 1:
        raise ValueError("Snapshot must be before the terminal observation.")
    requested_trial, requested_snapshot = trial, snapshot
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(c.seed)
    errors = np.empty((c.n_trials, c.time_steps, 2))
    observations = []
    max_norm_error = 0.0
    max_terminal_error = 0.0
    omega = 2 * np.pi / c.grid_size
    for i in range(c.n_trials):
        obs = simulate_trial(rng, c)
        observations.append(obs)
        result = run_trial(obs, c)
        errors[i] = trial_errors(result)
        max_norm_error = max(max_norm_error, float(np.max(np.abs(
            np.sum(result["filtered"], axis=1) * omega - 1))),
            float(np.max(np.abs(np.sum(result["smoothed"], axis=1) * omega - 1))))
        max_terminal_error = max(max_terminal_error, float(np.max(np.abs(
            result["filtered"][-1] - result["smoothed"][-1]))))
    maes = errors[:, :-1].mean(axis=1)
    automatic_trial, scores = representative_trial(maes)
    trial = automatic_trial if trial is None else trial
    if not 0 <= trial < c.n_trials:
        raise ValueError("Selected trial index is outside the simulated ensemble.")
    selected = run_trial(observations[trial], c)
    automatic_snapshot, ratios = choose_snapshot(selected)
    snapshot = automatic_snapshot if snapshot is None else snapshot
    if not 0 <= snapshot < c.time_steps - 1:
        raise ValueError("Snapshot must be before the terminal observation.")

    # Odd-to-odd Fourier reconstruction; no Gaussian approximation or temporal interpolation.
    dense_size = max(4095, 4 * c.grid_size + 3)
    if dense_size % 2 == 0:
        dense_size += 1
    dense_grid = np.linspace(0, 2 * np.pi, dense_size, endpoint=False)
    dense_filter = _evaluate_figfdn_1d(selected["filtered"], dense_size)
    dense_smooth = _evaluate_figfdn_1d(selected["smoothed"], dense_size)
    an_filter = _evaluate_figfan_1d(selected["filtered"], dense_size)
    an_smooth = _evaluate_figfan_1d(selected["smoothed"], dense_size)
    an_dn_max = float(max(np.max(np.abs(an_filter - dense_filter)),
                          np.max(np.abs(an_smooth - dense_smooth))))
    np.savez_compressed(output / "tracking_example_data.npz", **selected,
                        dense_grid=dense_grid, dense_filtered=dense_filter,
                        dense_smoothed=dense_smooth, all_trial_errors=errors,
                        all_states=np.stack([o["states"] for o in observations]),
                        all_measurements=np.stack([o["measurements"] for o in observations]),
                        selected_trial=trial, selected_time=snapshot)

    columns = ["time_step", "true_angle_rad", "observation_rad", "filter_mean_rad",
               "smoother_mean_rad", "shifted_noise_component", "filter_error_rad",
               "smoother_error_rad"]
    with (output / "tracking_example_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for t in range(c.time_steps):
            writer.writerow([t, selected["states"][t], selected["measurements"][t],
                             selected["filter_means"][t], selected["smoother_means"][t],
                             int(selected["shifted_components"][t]), *errors[trial, t]])
    with (output / "tracking_example_trial_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "filter_mae_rad", "smoother_mae_rad",
                         "reduction_percent", "selection_score", "selected"])
        for i, (fm, sm) in enumerate(maes):
            writer.writerow([i, fm, sm, 100 * (1 - sm / fm), scores[i], int(i == trial)])
    with (output / "tracking_example_snapshot.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_rad", "filtered_density_per_rad", "smoothed_density_per_rad"])
        writer.writerows(zip(dense_grid, dense_filter[snapshot], dense_smooth[snapshot]))

    summary = bootstrap_reduction(errors, seed=c.seed + 1)
    try:
        checkout = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=Path(__file__).resolve().parent,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        checkout = None
    metadata = dict(
        source_repository="FlorianPfaff/FourierSmoothing",
        source_revision=UPSTREAM_REVISION,
        local_checkout_revision=checkout,
        config=asdict(c),
        selection_rule="argmin squared IQR-standardized distance to the componentwise medians of trial filter/smoother MAEs; terminal time excluded",
        automatic_trial_index=automatic_trial,
        selected_trial_index=trial,
        selected_trial_number=trial + 1,
        trial_override=requested_trial is not None,
        requested_trial_index=requested_trial,
        snapshot_override=requested_snapshot is not None,
        requested_time_step=requested_snapshot,
        snapshot_rule="largest filtered secondary/primary peak-height ratio, peaks >= 0.75 rad apart, terminal time excluded",
        selected_time_step=snapshot,
        automatic_time_step=automatic_snapshot,
        filtering_second_to_first_peak_height_ratio=float(ratios[snapshot]),
        selected_trial=dict(filter_mae_rad=float(maes[trial, 0]),
                            smoother_mae_rad=float(maes[trial, 1]),
                            reduction_percent=float(100 * (1 - maes[trial, 1] / maes[trial, 0])),
                            filter_mae_percentile=float(100 * np.mean(maes[:, 0] <= maes[trial, 0])),
                            smoother_mae_percentile=float(100 * np.mean(maes[:, 1] <= maes[trial, 1]))),
        all_trials=summary,
        density_reconstruction="FIGFDN: Fourier interpolation of square roots, squaring, then normalization",
        dense_grid_size=dense_size,
        max_abs_figfan_figfdn_difference=an_dn_max,
        max_grid_normalization_error=max_norm_error,
        max_terminal_density_difference=max_terminal_error,
        python_version=platform.python_version(), numpy_version=np.__version__,
    )
    if c == Config():
        delta = np.asarray([summary["filter_mae_rad"], summary["smoother_mae_rad"]]) - EXPECTED_PUBLISHED_MAES
        metadata["max_abs_published_mae_difference"] = float(np.max(np.abs(delta)))
        if np.max(np.abs(delta)) > 1e-10:
            raise AssertionError(f"Default run does not reproduce the published MAEs: {delta}")
    (output / "tracking_example_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


# Plotting code is defined below. Every output is a separate figure.

def break_at_wrap(times: NDArray, angles: NDArray) -> tuple[NDArray, NDArray]:
    """Break lines at the 0/2pi seam instead of drawing misleading long jumps."""
    times = np.asarray(times, dtype=float)
    angles = np.mod(np.asarray(angles, dtype=float), 2 * np.pi)
    jumps = np.flatnonzero(np.abs(np.diff(angles)) > np.pi) + 1
    return np.insert(times, jumps, np.nan), np.insert(angles, jumps, np.nan)


def make_figures(output: Path, figures_dir: Path | None = None,
                 show: bool = False) -> list[Path]:
    """Render separate PDF/SVG/PNG figures (heatmap meshes are rasterized)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figures_dir = output / "figures" if figures_dir is None else figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((output / "tracking_example_metadata.json").read_text(encoding="utf-8"))
    with np.load(output / "tracking_example_data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    times = np.arange(data["states"].size)
    trial = int(data["selected_trial"])
    snapshot = int(data["selected_time"])
    T = int(times[-1])
    paths = []

    def base(title: str, xlabel: str, ylabel: str, height: float = 3.7):
        fig = plt.figure(figsize=(6.4, height), layout="constrained")
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
        return fig, ax

    def angle_axis(ax, axis="y"):
        ticks = np.arange(5) * np.pi / 2
        labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        if axis == "y":
            ax.set_ylim(0, 2 * np.pi)
            ax.set_yticks(ticks, labels)
        else:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks(ticks, labels)

    def save(fig, name):
        for extension in ("pdf", "svg", "png"):
            path = figures_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.04)
            paths.append(path)
        if show:
            plt.show()
        plt.close(fig)

    fig, ax = base("Angular tracking with ambiguous measurements",
                   r"Time step $t$", "Angle [rad]")
    tx, yy = break_at_wrap(times, data["states"])
    ax.plot(tx, yy, linewidth=1.9, marker="o", markersize=3.8, label="True angle")
    ax.plot(times, data["measurements"], linestyle="none", marker="x",
            markersize=6.3, markeredgewidth=1.5, label="Observation")
    tx, yy = break_at_wrap(times, data["filter_means"])
    ax.plot(tx, yy, linewidth=1.7, linestyle="--", marker="s", markersize=3.4,
            label="Filtering mean")
    tx, yy = break_at_wrap(times, data["smoother_means"])
    ax.plot(tx, yy, linewidth=1.9, linestyle="-.", marker="^", markersize=4,
            label="Smoothing mean")
    angle_axis(ax)
    ax.set_xlim(-0.4, T + 0.4)
    ax.set_xticks([0, 4, 8, 12, 16, T] if T == 19 else np.arange(0, T + 1, max(1, T // 5)))
    ax.legend(loc="upper right", ncol=2, fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    values = metadata["selected_trial"]
    ax.text(0.02, 0.79,
            f"Trial {trial + 1} of {metadata['config']['n_trials']}  |  "
            f"MAE: {values['filter_mae_rad']:.3f} → {values['smoother_mae_rad']:.3f} rad",
            transform=ax.transAxes, fontsize=10)
    save(fig, "tracking_trajectory")

    # Initial-time snapshot is always included. Also show the most ambiguous
    # filtered density, chosen by the explicit peak-ratio rule, not the gain.
    grid = data["dense_grid"]
    closed_grid = np.r_[grid, 2 * np.pi]
    for t in sorted({0, snapshot}):
        f = data["dense_filtered"][t]
        s = data["dense_smoothed"][t]
        top = 1.12 * max(float(f.max()), float(s.max()))
        fig, ax = base(f"Posterior density at $t={t}$", "Angle [rad]",
                       r"Probability density [rad$^{-1}$]")
        # Calling plot in the same order keeps the default automatic color cycle
        # consistent with the trajectory figure without fixing any colors.
        ax.plot([data["states"][t]] * 2, [0, top], linestyle=":", linewidth=1.5,
                label="True angle")
        ax.plot([data["measurements"][t]] * 2, [0, top], linestyle=(0, (2, 3)),
                linewidth=1.1, label="Observation")
        ax.plot(closed_grid, np.r_[f, f[0]], linestyle="--", linewidth=2.0,
                label=rf"Filtering $p(\theta_{t}\mid z_{{0:{t}}})$")
        ax.plot(closed_grid, np.r_[s, s[0]], linestyle="-.", linewidth=2.0,
                label=rf"Smoothing $p(\theta_{t}\mid z_{{0:{T}}})$")
        angle_axis(ax, "x")
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        save(fig, f"tracking_density_t{t:02d}")

    # Both heatmaps use the same linear density scale and discrete time cells.
    # Never normalize individual columns to their own maxima.
    vmax = float(max(data["dense_filtered"].max(), data["dense_smoothed"].max()))
    for key, title, name in (
        ("dense_filtered", r"Filtering densities $p(\theta_t\mid z_{0:t})$", "tracking_filter_density"),
        ("dense_smoothed", rf"Smoothed densities $p(\theta_t\mid z_{{0:{T}}})$", "tracking_smoother_density"),
    ):
        fig, ax = base(title, r"Time step $t$", "Angle [rad]", height=3.9)
        step = 2 * np.pi / grid.size
        angle_edges = np.arange(grid.size + 1) * step - step / 2
        heat = ax.pcolormesh(np.arange(T + 2) - 0.5, angle_edges, data[key].T,
                             shading="flat", vmin=0, vmax=vmax, rasterized=True)
        # Close the tiny last half-cell across the angular seam.
        ax.pcolormesh(np.arange(T + 2) - 0.5,
                      [2 * np.pi - step / 2, 2 * np.pi], data[key][:, :1].T,
                      shading="flat", vmin=0, vmax=vmax, rasterized=True)
        angle_axis(ax)
        ax.set_xlim(-0.5, T + 0.5)
        ax.set_xticks([0, 4, 8, 12, 16, T] if T == 19 else np.arange(0, T + 1, max(1, T // 5)))
        colorbar = fig.colorbar(heat, ax=ax, pad=0.025)
        colorbar.set_label(r"Probability density [rad$^{-1}$]", fontsize=10.5)
        colorbar.ax.tick_params(labelsize=9.5)
        save(fig, name)

    metadata["matplotlib_version"] = matplotlib.__version__
    metadata["initial_density_snapshot_time_step"] = 0
    metadata["heatmap_scale"] = dict(type="linear", vmin=0.0, vmax=vmax,
                                      per_column_normalization=False)
    (output / "tracking_example_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_tracking_example"))
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=1023)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--trial", type=int, help="Optional zero-based trial override; recorded in metadata.")
    parser.add_argument("--snapshot", type=int, help="Optional zero-based density-snapshot time override.")
    parser.add_argument("--plot-only", action="store_true", help="Render saved NPZ data without simulating again.")
    parser.add_argument("--no-plots", action="store_true", help="Only generate data and metadata.")
    parser.add_argument("--show", action="store_true", help="Display each figure after saving it.")
    args = parser.parse_args()
    if args.plot_only and args.no_plots:
        parser.error("--plot-only and --no-plots cannot be combined.")
    config = Config(n_trials=args.n_trials, grid_size=args.grid_size,
                    time_steps=args.time_steps, seed=args.seed)
    if not args.plot_only:
        metadata = prepare_data(config, args.output_dir, args.trial, args.snapshot)
        print(json.dumps(metadata, indent=2))
    if not args.no_plots:
        for path in make_figures(args.output_dir, args.figures_dir, show=args.show):
            print(path)


if __name__ == "__main__":
    main()
