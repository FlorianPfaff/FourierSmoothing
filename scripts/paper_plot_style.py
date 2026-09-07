"""Shared final-print-size style, following S3F-Improved-Paper.

Reference: scripts/generate_s1r2_paper_figures.py in
FlorianPfaff/S3F-Improved-Paper (044914c0b002816a6fa3a8a86d55d03831cf0d50).
Keep method identity fixed across figures; never displace coincident data.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TEXT_WIDTH = 516.0 / 72.27
COLUMN_WIDTH = 252.0 / 72.27
LABEL_SIZE = 8.0
TICK_SIZE = 7.0
LEGEND_SIZE = 7.0
METHODS = ("FIGFAN", "FIGFDN", "PF", "PWC")
# Draw the large hollow square first, so the coincident blue dot remains visible.
DRAW_ORDER = ("FIGFDN", "PWC", "PF", "FIGFAN")
COLORS = {"FIGF": "#1f77b4", "FIGFAN": "#1f77b4", "FIGFDN": "#ff7f0e",
          "PF": "#d62728", "PWC": "#2ca02c"}
MARKERS = {"FIGF": "o", "FIGFAN": "o", "FIGFDN": "s", "PF": "D", "PWC": "^"}
DASHES = {"FIGF": "-", "FIGFAN": "-", "FIGFDN": "--", "PF": ":", "PWC": "-."}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": TICK_SIZE, "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE, "axes.titleweight": "normal",
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.facecolor": "white",
    })


def line_style(method: str) -> dict:
    return dict(color=COLORS[method], marker=MARKERS[method],
                linestyle=DASHES[method], linewidth=1.6,
                markersize=5.6 if method == "FIGFDN" else (3.3 if method in {"FIGF", "FIGFAN"} else 4.2),
                markerfacecolor="white" if method in {"FIGFDN", "PWC"} else COLORS[method],
                markeredgewidth=0.85, markeredgecolor=COLORS[method])


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(False, which="minor")
    ax.tick_params(which="major", length=3, width=0.8, pad=2)
    ax.tick_params(which="minor", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def legend_handles(runtime: bool = False) -> list[Line2D]:
    methods = ("FIGF", "PF", "PWC") if runtime else METHODS
    return [Line2D([], [], label=m, **line_style(m)) for m in methods]
