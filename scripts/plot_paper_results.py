#!/usr/bin/env python
"""Plot generated CSV results into the paper repository's figures directory."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("../2026-07-FourierSmoothing-Paper/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("../2026-07-FourierSmoothing-Paper/figures"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    smoothing_summary_csv = args.results_dir / "smoothing_evaluation_summary.csv"
    figf_pwc_csv = args.results_dir / "figf_pwc_benchmark.csv"
    identity_csv = args.results_dir / "identity_torus_benchmark.csv"
    negativity_csv = args.results_dir / "truncation_negativity_diagnostic.csv"
    particle_csv = args.results_dir / "particle_smoother_baseline.csv"

    written = []
    if smoothing_summary_csv.exists():
        written.extend(_plot_smoothing_evaluation(smoothing_summary_csv, args.figures_dir, args.formats))
    if figf_pwc_csv.exists():
        written.extend(_plot_figf_pwc_benchmark(figf_pwc_csv, args.figures_dir, args.formats))
    if identity_csv.exists():
        written.extend(_plot_identity_benchmark(identity_csv, args.figures_dir, args.formats))
    if negativity_csv.exists():
        written.extend(_plot_negativity_diagnostic(negativity_csv, args.figures_dir, args.formats))
    if particle_csv.exists():
        written.extend(_plot_particle_baseline(particle_csv, args.figures_dir, args.formats))

    if not written:
        raise FileNotFoundError(
            f"No known result CSVs found in {args.results_dir}. Run the result-generation scripts first."
        )
    for path in written:
        print(path)


def _plot_smoothing_evaluation(csv_path: Path, figures_dir: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _configure_paper_style(plt)

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "method": row["method"],
                    "parameter": int(row["parameter"]),
                    "runtime_s": float(row["runtime_s_mean"]),
                    "runtime_s_std": float(row["runtime_s_std"]),
                    "mean_error_rad": float(row["mean_error_rad_mean"]),
                    "mean_error_rad_std": float(row["mean_error_rad_std"]),
                    "l1_error": float(row["l1_error_mean"]),
                    "l1_error_std": float(row["l1_error_std"]),
                }
            )

    method_order = ["FIGFAN", "FIGFDN", "PF", "PWC"]
    methods = [method for method in method_order if any(row["method"] == method for row in rows)]
    written = []

    written.extend(
        _plot_metric_by_parameter(
            rows,
            methods,
            metric="mean_error_rad",
            ylabel="mean-direction error [rad]",
            title="Mean error",
            output_base=figures_dir / "smoothing_mean_error_by_parameter",
            formats=formats,
            log_y=True,
        )
    )
    written.extend(
        _plot_metric_by_parameter(
            rows,
            methods,
            metric="l1_error",
            ylabel=r"mean $L^1$ error",
            title=r"$L^1$ density error",
            output_base=figures_dir / "smoothing_l1_error_by_parameter",
            formats=formats,
            log_y=True,
        )
    )
    runtime_rows = _runtime_parameter_rows(rows)
    written.extend(
        _plot_metric_by_parameter(
            runtime_rows,
            [method for method in ["FIGF", "PF", "PWC"] if any(row["method"] == method for row in runtime_rows)],
            metric="runtime_s",
            ylabel="runtime [ms]",
            title="Runtime",
            output_base=figures_dir / "smoothing_runtime_by_parameter",
            formats=formats,
            log_y=True,
        )
    )
    written.extend(
        _plot_metric_by_runtime(
            rows,
            methods,
            metric="mean_error_rad",
            ylabel="mean-direction error [rad]",
            title="Mean error over runtime",
            output_base=figures_dir / "smoothing_mean_error_by_runtime",
            formats=formats,
        )
    )
    written.extend(
        _plot_metric_by_runtime(
            rows,
            methods,
            metric="l1_error",
            ylabel=r"mean $L^1$ error",
            title=r"$L^1$ error over runtime",
            output_base=figures_dir / "smoothing_l1_error_by_runtime",
            formats=formats,
        )
    )
    written.extend(
        _plot_runtime_accuracy_summary(
            rows,
            methods,
            runtime_rows,
            figures_dir / "smoothing_runtime_accuracy_summary",
            formats,
        )
    )
    return written


def _plot_metric_by_parameter(
    rows: list[dict[str, str | int | float]],
    methods: list[str],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_base: Path,
    formats: list[str],
    log_y: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    _draw_metric_by_parameter(ax, rows, methods, metric=metric, ylabel=ylabel, title=title, log_y=log_y)
    written = _save_all(fig, output_base, formats)
    plt.close(fig)
    return written


def _draw_metric_by_parameter(ax, rows, methods, *, metric: str, ylabel: str, title: str, log_y: bool) -> None:
    import numpy as np  # pylint: disable=import-outside-toplevel

    for method in methods:
        method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: row["parameter"])
        parameters = np.asarray([row["parameter"] for row in method_rows], dtype=float)
        values = np.asarray([_metric_plot_value(row, metric) for row in method_rows], dtype=float)
        deviations = np.asarray([_metric_plot_deviation(row, metric) for row in method_rows], dtype=float)
        deviations = np.minimum(deviations, 0.8 * values)
        ax.errorbar(
            parameters,
            values,
            yerr=deviations,
            label=method,
            capsize=1.5,
            elinewidth=0.65,
            **_method_plot_style(method),
        )
    ax.set_xlabel(r"grid points $L$ / particles $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    if log_y and all(row[metric] > 0.0 for row in rows):
        ax.set_yscale("log")
        if metric == "runtime_s":
            ax.yaxis.set_major_locator(_runtime_log_locator())
            ax.yaxis.set_major_formatter(_plain_number_formatter())
            ax.yaxis.set_minor_formatter(_blank_formatter())
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, which="major", color="#B8B8B8", linewidth=0.45, alpha=0.65)


def _plot_metric_by_runtime(
    rows: list[dict[str, str | int | float]],
    methods: list[str],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_base: Path,
    formats: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    _draw_metric_by_runtime(ax, rows, methods, metric=metric, ylabel=ylabel, title=title)
    written = _save_all(fig, output_base, formats)
    plt.close(fig)
    return written


def _draw_metric_by_runtime(ax, rows, methods, *, metric: str, ylabel: str, title: str) -> None:
    import numpy as np  # pylint: disable=import-outside-toplevel

    for method in methods:
        method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: row["runtime_s"])
        runtimes = 1000.0 * np.asarray([row["runtime_s"] for row in method_rows], dtype=float)
        values = np.asarray([row[metric] for row in method_rows], dtype=float)
        runtime_deviations = 1000.0 * np.asarray([row["runtime_s_std"] for row in method_rows], dtype=float)
        metric_deviations = np.asarray([row[f"{metric}_std"] for row in method_rows], dtype=float)
        runtime_deviations = np.minimum(runtime_deviations, 0.8 * runtimes)
        metric_deviations = np.minimum(metric_deviations, 0.8 * values)
        style = _method_plot_style(method)
        style["linestyle"] = "none"
        ax.errorbar(
            runtimes,
            values,
            xerr=runtime_deviations,
            yerr=metric_deviations,
            label=method,
            capsize=1.5,
            elinewidth=0.65,
            **style,
        )
    ax.set_xlabel("runtime [ms]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if all(row["runtime_s"] > 0.0 for row in rows):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(_runtime_log_locator())
        ax.xaxis.set_major_formatter(_plain_number_formatter())
        ax.xaxis.set_minor_formatter(_blank_formatter())
    if all(row[metric] > 0.0 for row in rows):
        ax.set_yscale("log")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, which="major", color="#B8B8B8", linewidth=0.45, alpha=0.65)


def _plot_runtime_accuracy_summary(
    rows: list[dict[str, str | int | float]],
    methods: list[str],
    runtime_rows: list[dict[str, str | int | float]],
    output_base: Path,
    formats: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.05), constrained_layout=True)
    runtime_methods = [
        method for method in ["FIGF", "PF", "PWC"] if any(row["method"] == method for row in runtime_rows)
    ]
    _draw_metric_by_parameter(
        axes[0],
        runtime_rows,
        runtime_methods,
        metric="runtime_s",
        ylabel="runtime [ms]",
        title="(a) Runtime scaling",
        log_y=True,
    )
    _draw_metric_by_runtime(
        axes[1],
        rows,
        methods,
        metric="mean_error_rad",
        ylabel="mean-direction error [rad]",
        title="(b) Mean error",
    )
    _draw_metric_by_runtime(
        axes[2],
        rows,
        methods,
        metric="l1_error",
        ylabel=r"mean $L^1$ error",
        title=r"(c) $L^1$ density error",
    )
    written = _save_all(fig, output_base, formats)
    plt.close(fig)
    return written


def _metric_plot_value(row: dict[str, str | int | float], metric: str) -> float:
    value = float(row[metric])
    if metric == "runtime_s":
        return 1000.0 * value
    return value


def _metric_plot_deviation(row: dict[str, str | int | float], metric: str) -> float:
    value = float(row.get(f"{metric}_std", 0.0))
    if metric == "runtime_s":
        return 1000.0 * value
    return value


def _runtime_parameter_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    runtime_rows = []
    for row in rows:
        if row["method"] == "FIGFDN":
            continue
        if row["method"] == "FIGFAN":
            copied = dict(row)
            copied["method"] = "FIGF"
            runtime_rows.append(copied)
        else:
            runtime_rows.append(row)
    return runtime_rows


def _plain_number_formatter():
    from matplotlib.ticker import FuncFormatter  # pylint: disable=import-outside-toplevel

    return FuncFormatter(lambda value, _position: f"{value:g}")


def _runtime_log_locator():
    from matplotlib.ticker import LogLocator  # pylint: disable=import-outside-toplevel

    return LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=12)


def _blank_formatter():
    from matplotlib.ticker import NullFormatter  # pylint: disable=import-outside-toplevel

    return NullFormatter()


def _method_plot_style(method: str) -> dict[str, object]:
    styles = {
        "FIGF": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
        "FIGFAN": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
        "FIGFDN": {
            "marker": "s",
            "linestyle": "--",
            "color": "#E69F00",
            "markerfacecolor": "none",
            "markeredgewidth": 0.9,
        },
        "PF": {"marker": "^", "linestyle": "-.", "color": "#009E73"},
        "PWC": {
            "marker": "D",
            "linestyle": ":",
            "color": "#D55E00",
            "markerfacecolor": "none",
            "markeredgewidth": 0.9,
        },
    }
    return {"markersize": 3.6, **styles.get(method, {"marker": "o", "linestyle": "-"})}


def _configure_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.05,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _plot_figf_pwc_benchmark(csv_path: Path, figures_dir: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    grouped_l1: dict[tuple[str, int], list[float]] = defaultdict(list)
    grouped_runtime: dict[tuple[str, int], list[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["grid_size"]))
            grouped_l1[key].append(float(row["mean_l1_error_to_reference"]))
            runtime_method = "FIGF" if row["method"] in {"FIGFAN", "FIGFDN"} else row["method"]
            grouped_runtime[(runtime_method, int(row["grid_size"]))].append(float(row["smoother_runtime_s"]))

    method_order = ["FIGFAN", "FIGFDN", "PWC"]
    methods = [method for method in method_order if any(m == method for m, _ in grouped_l1)]
    written = []

    fig, ax = plt.subplots()
    for method in methods:
        grid_sizes = sorted(grid_size for m, grid_size in grouped_l1 if m == method)
        ax.plot(grid_sizes, [mean(grouped_l1[(method, n)]) for n in grid_sizes], marker="o", label=method)
    ax.set_xlabel("grid size")
    ax.set_ylabel("mean L1 error to fine-grid reference")
    ax.set_title("Smoothed density error")
    if all(value > 0.0 for values in grouped_l1.values() for value in values):
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "figf_pwc_l1_error", formats))
    plt.close(fig)

    fig, ax = plt.subplots()
    runtime_methods = [method for method in ["FIGF", "PWC"] if any(m == method for m, _ in grouped_runtime)]
    for method in runtime_methods:
        grid_sizes = sorted(grid_size for m, grid_size in grouped_runtime if m == method)
        ax.plot(grid_sizes, [mean(grouped_runtime[(method, n)]) for n in grid_sizes], marker="o", label=method)
    ax.set_xlabel("grid size")
    ax.set_ylabel("backward smoothing runtime [s]")
    ax.set_title("Backward pass runtime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "figf_pwc_runtime", formats))
    plt.close(fig)

    return written


def _plot_identity_benchmark(csv_path: Path, figures_dir: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    grouped_runtime: dict[tuple[str, int], list[float]] = defaultdict(list)
    grouped_difference: dict[tuple[str, int], list[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["grid_size"]))
            grouped_runtime[key].append(float(row["runtime_s"]))
            grouped_difference[key].append(float(row["max_abs_difference_to_grid"]))

    methods = sorted({method for method, _ in grouped_runtime})
    written = []

    fig, ax = plt.subplots()
    for method in methods:
        grid_sizes = sorted(grid_size for m, grid_size in grouped_runtime if m == method)
        ax.plot(grid_sizes, [mean(grouped_runtime[(method, n)]) for n in grid_sizes], marker="o", label=method)
    ax.set_xlabel("grid size / number of Fourier coefficients")
    ax.set_ylabel("runtime per smoothing run [s]")
    ax.set_title("Identity torus smoother runtime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "identity_torus_runtime", formats))
    plt.close(fig)

    fig, ax = plt.subplots()
    for method in methods:
        if method == "grid":
            continue
        grid_sizes = sorted(grid_size for m, grid_size in grouped_difference if m == method)
        ax.plot(grid_sizes, [mean(grouped_difference[(method, n)]) for n in grid_sizes], marker="o", label=method)
    ax.set_xlabel("grid size / number of Fourier coefficients")
    ax.set_ylabel("max absolute difference to grid reference")
    ax.set_title("Fourier smoother difference to grid reference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "identity_torus_difference", formats))
    plt.close(fig)

    return written


def _plot_negativity_diagnostic(csv_path: Path, figures_dir: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    grouped_mass: dict[float, list[tuple[int, float]]] = defaultdict(list)
    grouped_l1: dict[float, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sharpness = float(row["sharpness"])
            n_coefficients = int(row["n_coefficients"])
            grouped_mass[sharpness].append((n_coefficients, float(row["negative_mass"])))
            grouped_l1[sharpness].append((n_coefficients, float(row["l1_error_to_dense_grid"])))

    written = []
    fig, ax = plt.subplots()
    for sharpness in sorted(grouped_mass):
        values = sorted(grouped_mass[sharpness])
        ax.plot([n for n, _ in values], [v for _, v in values], marker="o", label=f"sharpness={sharpness:g}")
    ax.set_xlabel("number of Fourier coefficients")
    ax.set_ylabel("integrated negative part")
    ax.set_title("Negative mass after identity-Fourier truncation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "truncation_negative_mass", formats))
    plt.close(fig)

    fig, ax = plt.subplots()
    for sharpness in sorted(grouped_l1):
        values = sorted(grouped_l1[sharpness])
        ax.plot([n for n, _ in values], [v for _, v in values], marker="o", label=f"sharpness={sharpness:g}")
    ax.set_xlabel("number of Fourier coefficients")
    ax.set_ylabel("L1 error to dense-grid reference")
    ax.set_title("Dense-grid error after identity-Fourier truncation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "truncation_l1_error", formats))
    plt.close(fig)

    return written


def _plot_particle_baseline(csv_path: Path, figures_dir: Path, formats: list[str]) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    grouped_runtime: dict[int, list[float]] = defaultdict(list)
    grouped_mean_error: dict[int, list[float]] = defaultdict(list)
    grouped_max_error: dict[int, list[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_particles = int(row["n_particles"])
            grouped_runtime[n_particles].append(float(row["runtime_s"]))
            grouped_mean_error[n_particles].append(float(row["mean_abs_circular_error_to_grid"]))
            grouped_max_error[n_particles].append(float(row["max_abs_circular_error_to_grid"]))

    n_values = sorted(grouped_runtime)
    written = []

    fig, ax = plt.subplots()
    ax.plot(n_values, [mean(grouped_runtime[n]) for n in n_values], marker="o")
    ax.set_xlabel("number of particles")
    ax.set_ylabel("runtime per smoothing run [s]")
    ax.set_title("Particle smoother runtime")
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "particle_smoother_runtime", formats))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(n_values, [mean(grouped_mean_error[n]) for n in n_values], marker="o", label="mean over time")
    ax.plot(n_values, [mean(grouped_max_error[n]) for n in n_values], marker="o", label="max over time")
    ax.set_xlabel("number of particles")
    ax.set_ylabel("absolute circular error to grid smoother [rad]")
    ax.set_title("Particle smoother error to grid reference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    written.extend(_save_all(fig, figures_dir / "particle_smoother_error", formats))
    plt.close(fig)

    return written


def _save_all(fig, path_without_suffix: Path, formats: list[str]) -> list[Path]:
    written = []
    for fmt in formats:
        output_path = path_without_suffix.with_suffix(f".{fmt}")
        fig.savefig(output_path, bbox_inches="tight")
        written.append(output_path)
    return written


if __name__ == "__main__":
    main()
