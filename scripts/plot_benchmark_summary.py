#!/usr/bin/env python
"""Generate the compact four-panel benchmark summary used by the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_runtime_accuracy_column import (
    FULL_WIDTH_IN,
    PLOT_ORDER,
    _configure_paper_style,
    _draw_curve,
    _finish_axis,
    _legend_handles,
    _line_style,
    _method_rows,
    _parameter_xlim,
    _read_reference_diagnostics,
    _read_rows,
    _runtime_ms_and_iqr,
    _runtime_xlim,
    _save_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("../2026-07-FourierSmoothing-Paper/results"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../2026-07-FourierSmoothing-Paper/figures"),
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--filename", default="smoothing_benchmark_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.results_dir / "smoothing_evaluation_summary.csv")
    reference_mean_error, reference_l1 = _read_reference_diagnostics(
        args.results_dir / "reference_stability.json"
    )
    _configure_paper_style()
    written = _plot_benchmark_summary(
        rows,
        args.figures_dir / args.filename,
        args.formats,
        reference_mean_error=reference_mean_error,
        reference_l1=reference_l1,
    )
    for path in written:
        print(path)


def _plot_benchmark_summary(
    rows: list[dict[str, str | int | float]],
    output_base: Path,
    formats: list[str],
    *,
    reference_mean_error: float,
    reference_l1: float,
) -> list[Path]:
    """Render a compact, nonredundant four-panel benchmark plate."""

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH_IN, 3.98))
    mean_ax, l1_ax, runtime_ax, tradeoff_ax = axes.flat

    for ax, metric, ylabel, reference_value in (
        (mean_ax, "mean_error_rad", "mean-direction error [rad]", reference_mean_error),
        (l1_ax, "l1_error", r"mean $L^1$ discrepancy", reference_l1),
    ):
        for method in PLOT_ORDER:
            method_rows = _method_rows(rows, method, sort_key="parameter")
            _draw_curve(
                ax,
                method_rows,
                method,
                x_key="parameter",
                y_key=metric,
                shade_iqr=method == "PF",
            )
        ax.axhline(
            reference_value,
            color="0.42",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*_parameter_xlim(rows))
        ax.set_xlabel(r"grid points $L$ / particles $N$")
        ax.set_ylabel(ylabel)
        _finish_axis(ax)

    mean_ax.set_ylim(2.4e-4, 8.0e-2)
    l1_ax.set_ylim(5.5e-5, 6.0e-1)

    for method, source_method in (("FIGF", "FIGFAN"), ("PWC", "PWC"), ("PF", "PF")):
        method_rows = _method_rows(rows, source_method, sort_key="parameter")
        x = np.asarray([float(row["parameter"]) for row in method_rows])
        y = np.asarray([_runtime_ms_and_iqr(row)[0] for row in method_rows])
        y_low = np.asarray([_runtime_ms_and_iqr(row)[3] for row in method_rows])
        y_high = np.asarray([_runtime_ms_and_iqr(row)[4] for row in method_rows])
        runtime_ax.errorbar(
            x,
            y,
            yerr=[y_low, y_high],
            capsize=1.7,
            elinewidth=0.6,
            zorder=5,
            **_line_style(method),
        )
    runtime_ax.set_xscale("log")
    runtime_ax.set_yscale("log")
    runtime_ax.set_xlim(*_parameter_xlim(rows))
    runtime_ax.set_ylim(0.75, 210.0)
    runtime_ax.set_xlabel(r"grid points $L$ / particles $N$")
    runtime_ax.set_ylabel("median runtime [ms]")
    _finish_axis(runtime_ax)

    for method in PLOT_ORDER:
        method_rows = _method_rows(rows, method, sort_key="runtime_s_median")
        _draw_curve(
            tradeoff_ax,
            method_rows,
            method,
            x_key="runtime_s_median",
            y_key="l1_error",
            x_iqr=True,
            y_iqr=method == "PF",
        )
    tradeoff_ax.axhline(
        reference_l1,
        color="0.42",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    tradeoff_ax.set_xscale("log")
    tradeoff_ax.set_yscale("log")
    tradeoff_ax.set_xlim(*_runtime_xlim(rows))
    tradeoff_ax.set_ylim(5.5e-5, 6.0e-1)
    tradeoff_ax.set_xlabel("median runtime [ms]")
    tradeoff_ax.set_ylabel(r"mean $L^1$ discrepancy")
    _finish_axis(tradeoff_ax)

    for ax in (l1_ax, tradeoff_ax):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelleft=False, labelright=True)

    titles = (
        "(a) Mean-direction error",
        r"(b) $L^1$ reference discrepancy",
        "(c) Runtime scaling",
        r"(d) $L^1$ accuracy--runtime tradeoff",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", pad=3.0, fontsize=7.5)

    fig.subplots_adjust(
        left=0.078,
        right=0.922,
        top=0.955,
        bottom=0.145,
        wspace=0.19,
        hspace=0.42,
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.45,
    )

    written = _save_all(fig, (output_base,), formats)
    plt.close(fig)
    return written


if __name__ == "__main__":
    main()
