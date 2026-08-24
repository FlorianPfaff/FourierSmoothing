#!/usr/bin/env python
"""Generate publication figures and the shared PGFPlots data files.

The script keeps the numerical representation used by the paper (one whitespace-
separated data file per method), but renders the publication figures at their
final IEEE print size.  This avoids the cramped three-row column figure and
keeps the Matplotlib previews identical to the figures included in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


TEX_PT_PER_IN = 72.27
IEEE_TEXTWIDTH_PT = 516.0
FULL_WIDTH_IN = IEEE_TEXTWIDTH_PT / TEX_PT_PER_IN
FIGURE_HEIGHT_IN = 2.78

REFERENCE_MEAN_ERROR_DEFAULT = 2.979694047679854e-4
REFERENCE_L1_DEFAULT = 7.299654857145697e-5

PGF_METHOD_SLUGS = {
    "FIGFAN": "figfan",
    "FIGFDN": "figfdn",
    "PF": "pf",
    "PWC": "pwc",
}
METHOD_ORDER = ("FIGFAN", "FIGFDN", "PF", "PWC")
PLOT_ORDER = ("FIGFDN", "FIGFAN", "PWC", "PF")

COLORS = {
    "FIGF": "#0072B2",
    "FIGFAN": "#0072B2",
    "FIGFDN": "#E69F00",
    "PF": "#009E73",
    "PWC": "#D55E00",
}
MARKERS = {
    "FIGF": "o",
    "FIGFAN": "o",
    "FIGFDN": "s",
    "PF": "^",
    "PWC": "D",
}
LINESTYLES = {
    "FIGF": "-",
    "FIGFAN": "-",
    "FIGFDN": "--",
    "PF": "-.",
    "PWC": ":",
}
MARKER_SIZES = {
    "FIGF": 3.8,
    "FIGFAN": 3.4,
    "FIGFDN": 5.0,
    "PF": 4.4,
    "PWC": 4.5,
}


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.results_dir / "smoothing_evaluation_summary.csv"
    rows = _read_rows(csv_path)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.extend(_write_pgfplot_data(rows, args.figures_dir / "data"))

    reference_mean_error, reference_l1 = _read_reference_diagnostics(
        args.results_dir / "reference_stability.json"
    )
    _configure_paper_style()
    written.extend(
        _plot_accuracy_by_parameter(
            rows,
            args.figures_dir / "smoothing_accuracy_by_parameter",
            args.formats,
            reference_mean_error=reference_mean_error,
            reference_l1=reference_l1,
        )
    )
    written.extend(
        _plot_runtime_accuracy_summary(
            rows,
            (
                args.figures_dir / "smoothing_runtime_accuracy_summary",
                # Backward-compatible alias used by tests and older artifacts.
                args.figures_dir / "smoothing_runtime_accuracy_column",
            ),
            args.formats,
            reference_mean_error=reference_mean_error,
            reference_l1=reference_l1,
        )
    )

    for path in written:
        print(path)


def _read_rows(csv_path: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "method": row["method"],
                    "parameter": int(row["parameter"]),
                    "runtime_s_mean": float(row["runtime_s_mean"]),
                    "runtime_s_median": float(row["runtime_s_median"]),
                    "runtime_s_q25": float(row["runtime_s_q25"]),
                    "runtime_s_q75": float(row["runtime_s_q75"]),
                    "mean_error_rad": float(row["mean_error_rad_mean"]),
                    "mean_error_rad_q25": float(row["mean_error_rad_q25"]),
                    "mean_error_rad_q75": float(row["mean_error_rad_q75"]),
                    "l1_error": float(row["l1_error_mean"]),
                    "l1_error_q25": float(row["l1_error_q25"]),
                    "l1_error_q75": float(row["l1_error_q75"]),
                }
            )
    if not rows:
        raise ValueError(f"No smoothing-evaluation rows found in {csv_path}")
    return rows


def _write_pgfplot_data(
    rows: list[dict[str, str | int | float]],
    data_dir: Path,
) -> list[Path]:
    """Write one shared PGFPlots data file per method.

    Accuracy values are arithmetic means over repetitions. Runtime is represented
    by its median and interquartile range because the particle timings are
    right-skewed. The arithmetic runtime mean remains available for audit.
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    header = (
        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error mean_error_q25 "
        "mean_error_q75 mean_error_err_low mean_error_err_high l1_error l1_error_q25 "
        "l1_error_q75 l1_error_err_low l1_error_err_high"
    )
    for method, slug in PGF_METHOD_SLUGS.items():
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["parameter"],
        )
        if not method_rows:
            continue
        path = data_dir / f"smoothing_evaluation_{slug}.dat"
        lines = [header]
        for row in method_rows:
            mean_ms = 1000.0 * float(row["runtime_s_mean"])
            median_ms, q25_ms, q75_ms, err_low_ms, err_high_ms = _runtime_ms_and_iqr(row)
            lines.append(
                " ".join(
                    [
                        str(int(row["parameter"])),
                        _format_data_value(mean_ms),
                        _format_data_value(median_ms),
                        _format_data_value(q25_ms),
                        _format_data_value(q75_ms),
                        _format_data_value(err_low_ms),
                        _format_data_value(err_high_ms),
                        _format_data_value(float(row["mean_error_rad"])),
                        _format_data_value(float(row["mean_error_rad_q25"])),
                        _format_data_value(float(row["mean_error_rad_q75"])),
                        _format_data_value(
                            float(row["mean_error_rad"])
                            - float(row["mean_error_rad_q25"])
                        ),
                        _format_data_value(
                            float(row["mean_error_rad_q75"])
                            - float(row["mean_error_rad"])
                        ),
                        _format_data_value(float(row["l1_error"])),
                        _format_data_value(float(row["l1_error_q25"])),
                        _format_data_value(float(row["l1_error_q75"])),
                        _format_data_value(
                            float(row["l1_error"]) - float(row["l1_error_q25"])
                        ),
                        _format_data_value(
                            float(row["l1_error_q75"]) - float(row["l1_error"])
                        ),
                    ]
                )
            )
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
        written.append(path)
    return written


def _format_data_value(value: float) -> str:
    return f"{value:.12g}"


def _read_reference_diagnostics(path: Path) -> tuple[float, float]:
    if not path.exists():
        return REFERENCE_MEAN_ERROR_DEFAULT, REFERENCE_L1_DEFAULT
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return (
        float(
            data.get(
                "mean_between_run_mean_direction_deviation_rad",
                REFERENCE_MEAN_ERROR_DEFAULT,
            )
        ),
        float(data.get("pwc_reference_refinement_mean_l1", REFERENCE_L1_DEFAULT)),
    )


def _plot_accuracy_by_parameter(
    rows: list[dict[str, str | int | float]],
    output_base: Path,
    formats: list[str],
    *,
    reference_mean_error: float,
    reference_l1: float,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, FIGURE_HEIGHT_IN))

    panels = (
        ("mean_error_rad", "mean-direction error [rad]", reference_mean_error),
        ("l1_error", r"mean $L^1$ discrepancy", reference_l1),
    )
    for ax, (metric, ylabel, reference_value) in zip(axes, panels):
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

    axes[0].set_ylim(2.4e-4, 8.0e-2)
    axes[1].set_ylim(5.5e-5, 6.0e-1)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].tick_params(axis="y", labelleft=False, labelright=True)

    fig.subplots_adjust(left=0.078, right=0.922, top=0.975, bottom=0.44, wspace=0.18)
    fig.text(0.285, 0.27, "(a) Mean-direction error", ha="center", va="center", fontsize=7.4)
    fig.text(
        0.715,
        0.27,
        r"(b) $L^1$ reference discrepancy",
        ha="center",
        va="center",
        fontsize=7.4,
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.45,
    )

    written = _save_all(fig, (output_base,), formats)
    plt.close(fig)
    return written


def _plot_runtime_accuracy_summary(
    rows: list[dict[str, str | int | float]],
    output_bases: tuple[Path, ...],
    formats: list[str],
    *,
    reference_mean_error: float,
    reference_l1: float,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, FIGURE_HEIGHT_IN))

    runtime_ax = axes[0]
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

    panels = (
        (axes[1], "mean_error_rad", "mean-direction error [rad]", reference_mean_error),
        (axes[2], "l1_error", r"mean $L^1$ discrepancy", reference_l1),
    )
    for ax, metric, ylabel, reference_value in panels:
        for method in PLOT_ORDER:
            method_rows = _method_rows(rows, method, sort_key="runtime_s_median")
            _draw_curve(
                ax,
                method_rows,
                method,
                x_key="runtime_s_median",
                y_key=metric,
                x_iqr=True,
                y_iqr=method == "PF",
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
        ax.set_xlim(*_runtime_xlim(rows))
        ax.set_xlabel("median runtime [ms]")
        ax.set_ylabel(ylabel)
        _finish_axis(ax)

    axes[1].set_ylim(2.4e-4, 8.0e-2)
    axes[2].set_ylim(5.5e-5, 6.0e-1)
    axes[2].yaxis.tick_right()
    axes[2].yaxis.set_label_position("right")
    axes[2].tick_params(axis="y", labelleft=False, labelright=True)

    fig.subplots_adjust(left=0.075, right=0.925, top=0.975, bottom=0.44, wspace=0.32)
    for xpos, label in zip(
        (0.19, 0.50, 0.81),
        ("(a) Runtime scaling", "(b) Mean-direction error", r"(c) $L^1$ reference discrepancy"),
    ):
        fig.text(xpos, 0.27, label, ha="center", va="center", fontsize=7.3)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
        handletextpad=0.42,
    )

    written = _save_all(fig, output_bases, formats)
    plt.close(fig)
    return written


def _draw_curve(
    ax: plt.Axes,
    rows: list[dict[str, str | int | float]],
    method: str,
    *,
    x_key: str,
    y_key: str,
    shade_iqr: bool = False,
    x_iqr: bool = False,
    y_iqr: bool = False,
) -> None:
    x = np.asarray([_plot_value(row, x_key) for row in rows], dtype=float)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=float)

    if shade_iqr:
        y_low = np.asarray([float(row[f"{y_key}_q25"]) for row in rows], dtype=float)
        y_high = np.asarray([float(row[f"{y_key}_q75"]) for row in rows], dtype=float)
        ax.fill_between(x, y_low, y_high, color=COLORS[method], alpha=0.13, linewidth=0, zorder=2)

    if x_iqr or y_iqr:
        xerr = None
        if x_iqr:
            medians = np.asarray([_runtime_ms_and_iqr(row)[0] for row in rows])
            xerr = [
                np.asarray([_runtime_ms_and_iqr(row)[3] for row in rows]),
                np.asarray([_runtime_ms_and_iqr(row)[4] for row in rows]),
            ]
            x = medians
        yerr = None
        if y_iqr:
            yerr = [
                y - np.asarray([float(row[f"{y_key}_q25"]) for row in rows]),
                np.asarray([float(row[f"{y_key}_q75"]) for row in rows]) - y,
            ]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            capsize=1.7,
            elinewidth=0.6,
            zorder=5,
            **_line_style(method),
        )
        return

    ax.plot(x, y, zorder=5, **_line_style(method))


def _plot_value(row: dict[str, str | int | float], key: str) -> float:
    value = float(row[key])
    if key.startswith("runtime_s"):
        return 1000.0 * value
    return value


def _method_rows(
    rows: list[dict[str, str | int | float]], method: str, *, sort_key: str
) -> list[dict[str, str | int | float]]:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"Missing rows for method {method}")
    return sorted(selected, key=lambda row: float(row[sort_key]))


def _runtime_ms_and_iqr(
    row: dict[str, str | int | float],
) -> tuple[float, float, float, float, float]:
    median_ms = 1000.0 * float(row["runtime_s_median"])
    q25_ms = 1000.0 * float(row["runtime_s_q25"])
    q75_ms = 1000.0 * float(row["runtime_s_q75"])
    return median_ms, q25_ms, q75_ms, median_ms - q25_ms, q75_ms - median_ms


def _parameter_xlim(rows: list[dict[str, str | int | float]]) -> tuple[float, float]:
    values = np.asarray([float(row["parameter"]) for row in rows], dtype=float)
    return max(0.7, 0.7 * float(np.min(values))), 1.2 * float(np.max(values))


def _runtime_xlim(rows: list[dict[str, str | int | float]]) -> tuple[float, float]:
    values = np.asarray([1000.0 * float(row["runtime_s_median"]) for row in rows])
    return max(0.05, 0.72 * float(np.min(values))), 1.5 * float(np.max(values))


def _line_style(method: str) -> dict[str, object]:
    return {
        "color": COLORS[method],
        "marker": MARKERS[method],
        "linestyle": LINESTYLES[method],
        "linewidth": 1.45,
        "markersize": MARKER_SIZES[method],
        "markerfacecolor": "white" if method in {"FIGFDN", "PWC"} else COLORS[method],
        "markeredgecolor": COLORS[method],
        "markeredgewidth": 0.9,
    }


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], label=method, **_line_style(method))
        for method in METHOD_ORDER
    ]


def _finish_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="0.78", linewidth=0.55, alpha=0.55)
    ax.grid(True, which="minor", color="0.88", linewidth=0.35, alpha=0.50)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.75, pad=1.6)
    ax.tick_params(axis="both", which="minor", length=1.7, width=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_axisbelow(True)


def _configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _save_all(
    fig: plt.Figure,
    output_bases: tuple[Path, ...],
    formats: list[str],
) -> list[Path]:
    written: list[Path] = []
    for output_base in output_bases:
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for suffix in formats:
            path = output_base.with_suffix(f".{suffix}")
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
            written.append(path)
    return written


if __name__ == "__main__":
    main()
