#!/usr/bin/env python
"""Generate the paper's column-width runtime/accuracy summary figure and PGF data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLUMN_WIDTH_IN = 3.45
PGF_METHOD_SLUGS = {
    "FIGFAN": "figfan",
    "FIGFDN": "figfdn",
    "PF": "pf",
    "PWC": "pwc",
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

    for path in _write_pgfplot_data(rows, args.figures_dir / "data"):
        print(path)
    for path in _plot_runtime_accuracy_column(
        rows,
        args.figures_dir / "smoothing_runtime_accuracy_column",
        args.formats,
    ):
        print(path)


def _read_rows(csv_path: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "method": row["method"],
                    "parameter": int(row["parameter"]),
                    "runtime_s": float(row["runtime_s_mean"]),
                    "mean_error_rad": float(row["mean_error_rad_mean"]),
                    "l1_error": float(row["l1_error_mean"]),
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

    The files are consumed by both the representation-size and runtime-tradeoff
    figures, so the manuscript has one numerical source per method instead of
    duplicating coordinates in multiple TeX files.
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for method, slug in PGF_METHOD_SLUGS.items():
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["parameter"],
        )
        if not method_rows:
            continue
        path = data_dir / f"smoothing_evaluation_{slug}.dat"
        lines = ["n runtime_ms mean_error l1_error"]
        for row in method_rows:
            lines.append(
                " ".join(
                    [
                        str(int(row["parameter"])),
                        _format_data_value(1000.0 * float(row["runtime_s"])),
                        _format_data_value(float(row["mean_error_rad"])),
                        _format_data_value(float(row["l1_error"])),
                    ]
                )
            )
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
        written.append(path)
    return written


def _format_data_value(value: float) -> str:
    return f"{value:.12g}"


def _plot_runtime_accuracy_column(
    rows: list[dict[str, str | int | float]],
    output_base: Path,
    formats: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _configure_paper_style(plt)
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 5.05))

    runtime_rows = _runtime_parameter_rows(rows)
    runtime_methods = [method for method in ("FIGF", "PF", "PWC") if _has_method(runtime_rows, method)]
    accuracy_methods = [
        method for method in ("FIGFAN", "FIGFDN", "PF", "PWC") if _has_method(rows, method)
    ]

    _draw_metric_by_parameter(
        axes[0],
        runtime_rows,
        runtime_methods,
        metric="runtime_s",
        ylabel="runtime [ms]",
        title="(a) Runtime Scaling",
    )
    _draw_metric_by_runtime(
        axes[1],
        rows,
        accuracy_methods,
        metric="mean_error_rad",
        ylabel="mean-direction error [rad]",
        title="(b) Mean-Direction Error",
    )
    _draw_metric_by_runtime(
        axes[2],
        rows,
        accuracy_methods,
        metric="l1_error",
        ylabel=r"mean $L^1$ error",
        title=r"(c) $L^1$ Density Error",
    )

    axes[0].legend(loc="best", ncol=3, columnspacing=0.8, handletextpad=0.35)
    axes[1].legend(
        loc="lower right",
        ncol=2,
        framealpha=0.9,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.22, right=0.985, top=0.985, bottom=0.09, hspace=0.48)

    written = _save_all(fig, output_base, formats)
    plt.close(fig)
    return written


def _draw_metric_by_parameter(
    ax,
    rows: list[dict[str, str | int | float]],
    methods: list[str],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for method in methods:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["parameter"],
        )
        parameters = [float(row["parameter"]) for row in method_rows]
        values = [_metric_plot_value(row, metric) for row in method_rows]
        ax.plot(parameters, values, label=method, **_method_plot_style(method))

    ax.set_xlabel(r"grid points $L$ / particles $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=2.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish_axis(ax)


def _draw_metric_by_runtime(
    ax,
    rows: list[dict[str, str | int | float]],
    methods: list[str],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for method in methods:
        # Parameter order exposes how each method evolves as its representation is
        # refined. Measured runtime can fluctuate slightly, so a curve may move
        # left before continuing right; the paper caption states this explicitly.
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["parameter"],
        )
        runtimes_ms = [1000.0 * float(row["runtime_s"]) for row in method_rows]
        values = [float(row[metric]) for row in method_rows]
        ax.plot(runtimes_ms, values, label=method, **_method_plot_style(method))

    ax.set_xlabel("runtime [ms]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=2.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish_axis(ax)


def _finish_axis(ax) -> None:
    ax.grid(True, which="major", color="#B8B8B8", linewidth=0.45, alpha=0.65)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.75, pad=1.5)
    ax.tick_params(axis="both", which="minor", length=1.7, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def _runtime_parameter_rows(
    rows: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    runtime_rows: list[dict[str, str | int | float]] = []
    for row in rows:
        if row["method"] == "FIGFDN":
            continue
        copied = dict(row)
        if copied["method"] == "FIGFAN":
            copied["method"] = "FIGF"
        runtime_rows.append(copied)
    return runtime_rows


def _metric_plot_value(row: dict[str, str | int | float], metric: str) -> float:
    value = float(row[metric])
    return 1000.0 * value if metric == "runtime_s" else value


def _has_method(rows: list[dict[str, str | int | float]], method: str) -> bool:
    return any(row["method"] == method for row in rows)


def _method_plot_style(method: str) -> dict[str, object]:
    styles = {
        "FIGF": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
        "FIGFAN": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
        "FIGFDN": {
            "marker": "s",
            "linestyle": "--",
            "color": "#E69F00",
            "markerfacecolor": "none",
            "markeredgewidth": 0.8,
        },
        "PF": {"marker": "^", "linestyle": "-.", "color": "#009E73"},
        "PWC": {
            "marker": "D",
            "linestyle": ":",
            "color": "#D55E00",
            "markerfacecolor": "none",
            "markeredgewidth": 0.8,
        },
    }
    return {"markersize": 3.2, "linewidth": 1.0, **styles[method]}


def _configure_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _save_all(fig, output_base: Path, formats: list[str]) -> list[Path]:
    written = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    return written


if __name__ == "__main__":
    main()
