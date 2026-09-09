#!/usr/bin/env python
"""Canonical paper renderer; presentation only, never rerun benchmark timings.

Run with --results-dir PAPER/results --figures-dir PAPER/figures.
Each chart is rendered independently at its final printed size. PyMuPDF then
places the vector PDFs side by side without scaling their fonts or line widths.
Legacy plotting scripts remain available for historical artifact reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import pymupdf as fitz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_plot_style import (TEXT_WIDTH, COLUMN_WIDTH, DRAW_ORDER,
                              COLORS, configure, line_style, style_axis, legend_handles)
from plot_runtime_accuracy_column import _read_rows, _read_reference_diagnostics


SPACE_TIME_PANELS = ("filtering", "backward", "smoothing")
SPACE_TIME_WIDTH = 0.32 * TEXT_WIDTH
SPACE_TIME_HEIGHT = 2.5
SPACE_TIME_CASE = dict(grid_size=255, time_steps=9, likelihood_sharpness=7.0,
                       noise_concentration=2.2, prior_concentration=0.45)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, path: Path) -> None:
    """Keep the explicit page size; tight cropping would change printed fonts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        if box is not None and (box.x0 < -1 or box.y0 < -1 or
                               box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Labels extend outside the figure: {path.name}")
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def preview(path: Path) -> None:
    with fitz.open(path) as document:
        document[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(path.with_suffix(".png"))


def interval_bars(ax, x, y, low, high, *, horizontal: bool, color: str) -> None:
    """Draw actual quartile endpoints, even if a mean lies outside its IQR."""
    if horizontal:
        ax.hlines(y, low, high, colors=color, linewidth=0.65, alpha=0.65, zorder=2)
        for endpoint in (low, high):
            ax.plot(endpoint, y, "|", color=color, markersize=3.0, markeredgewidth=0.65, alpha=0.65, zorder=2)
    else:
        ax.vlines(x, low, high, colors=color, linewidth=0.65, alpha=0.65, zorder=2)
        for endpoint in (low, high):
            ax.plot(x, endpoint, "_", color=color, markersize=3.0, markeredgewidth=0.65, alpha=0.65, zorder=2)


def chart(rows, metric: str, by_runtime: bool, reference: float | None,
          path: Path, width: float, height: float, title: str = "", own_legend: bool = False) -> None:
    fig = plt.figure(figsize=(width, height))
    left, right, bottom, top = 0.55, 0.08, 0.43, (0.48 if own_legend else 0.27)
    ax = fig.add_axes([left / width, bottom / height,
                       (width - left - right) / width, (height - top - bottom) / height])
    runtime_panel = metric == "runtime_s_median"
    plotted_x, plotted_y = [], []
    for method in DRAW_ORDER:
        if runtime_panel and method == "FIGFDN":
            continue  # One recursion, not two separately timed smoothers.
        selected = sorted((r for r in rows if r["method"] == method), key=lambda r: r["parameter"])
        if not selected:
            raise ValueError(f"Missing method {method}")
        x = np.array([1000 * r["runtime_s_median"] if by_runtime else r["parameter"] for r in selected])
        y = np.array([r[metric] for r in selected]) * (1000 if runtime_panel else 1)
        if by_runtime:
            low = np.array([1000 * r["runtime_s_q25"] for r in selected])
            high = np.array([1000 * r["runtime_s_q75"] for r in selected])
            interval_bars(ax, x, y, low, high, horizontal=True, color=COLORS[method])
            plotted_x.extend([*low, *high])
        if runtime_panel or method == "PF":
            key = "runtime_s" if runtime_panel else metric
            scale = 1000 if runtime_panel else 1
            low = np.array([scale * r[f"{key}_q25"] for r in selected])
            high = np.array([scale * r[f"{key}_q75"] for r in selected])
            interval_bars(ax, x, y, low, high, horizontal=False, color=COLORS[method])
            plotted_y.extend([*low, *high])
        ax.plot(x, y, zorder=5 if method == "FIGFAN" else 4, **line_style(method))
        plotted_x.extend(x)
        plotted_y.extend(y)
    if reference is not None:
        ax.axhline(reference, color="0.45", linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)
        plotted_y.append(reference)
    if min(plotted_x) <= 0 or min(plotted_y) <= 0:
        raise ValueError("Logarithmic publication plots require strictly positive values")
    ax.set(xscale="log", yscale="log",
           xlim=(min(plotted_x) * 0.72, max(plotted_x) * 1.35),
           ylim=(min(plotted_y) * 0.58, max(plotted_y) * 1.6))
    ax.set_xlabel("median runtime / ms" if by_runtime else r"grid points $L$ / particles $N$", labelpad=2)
    ax.set_ylabel({"runtime_s_median": "median runtime / ms", "mean_error_rad": "mean error / rad",
                   "l1_error": r"mean $L^1$ distance"}[metric], labelpad=2)
    if title:
        ax.set_title(title, pad=6)
    style_axis(ax)
    if own_legend:
        fig.legend(handles=legend_handles(runtime_panel), loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.56, 0.99), handlelength=2.1,
                   columnspacing=0.85, labelspacing=0.25)
    save(fig, path)


def assemble(panels: list[Path], target: Path, *, legend: bool) -> None:
    """Place final-size charts on one vector page, with a single shared legend."""
    sources = [fitz.open(path) for path in panels]
    width = TEXT_WIDTH * 72
    height = max(s[0].rect.height for s in sources)
    gap = (width - sum(s[0].rect.width for s in sources)) / max(len(sources) - 1, 1)
    if gap < -0.1:
        raise ValueError("Panel widths exceed manuscript text width")
    document = fitz.open()
    page = document.new_page(width=width, height=height + (23 if legend else 0))
    x = 0.0
    for source in sources:
        rect = source[0].rect
        page.show_pdf_page(fitz.Rect(x, 0, x + rect.width, rect.height), source, 0)
        x += rect.width + gap
    if legend:
        fig = plt.figure(figsize=(TEXT_WIDTH, 23 / 72))
        handles = legend_handles()
        handles.append(Line2D([], [], color="0.45", linestyle=(0, (3, 2)), linewidth=0.85,
                              label="reference diagnostic"))
        fig.legend(handles=handles, loc="center", ncol=5, frameon=False,
                   handlelength=2.1, columnspacing=1.1, handletextpad=0.4)
        with tempfile.TemporaryDirectory() as directory:
            legend_path = Path(directory) / "legend.pdf"
            save(fig, legend_path)
            with fitz.open(legend_path) as source:
                page.show_pdf_page(fitz.Rect(0, height, width, height + 23), source, 0)
    document.save(target, garbage=4, deflate=True, no_new_id=True)
    document.close()
    for source in sources:
        source.close()
    preview(target)


def hero(target: Path) -> list[Path]:
    """Reuse the existing illustrative case without changing its model or data."""
    from plot_smoothing_hero import _make_smoothing_case, _column_normalize, _plot_circular_mean
    from matplotlib import patheffects
    fields = _make_smoothing_case(**SPACE_TIME_CASE)
    panels = []
    width, height = SPACE_TIME_WIDTH, SPACE_TIME_HEIGHT
    for name, field in zip(SPACE_TIME_PANELS, fields):
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([0.42 / width, 0.40 / height, (width - 0.54) / width, 1.51 / height])
        image = ax.imshow(_column_normalize(field).T, origin="lower", aspect="auto",
                          extent=(-0.5, 8.5, 0, 2 * np.pi), cmap="viridis", vmin=0, vmax=1,
                          interpolation="nearest", rasterized=True)
        if name != "backward":
            line = _plot_circular_mean(ax, field)
            line.set_linewidth(1.1)
            line.set_path_effects([patheffects.Stroke(linewidth=1.9, foreground="black", alpha=0.55), patheffects.Normal()])
        ax.set(xlabel="time step", xticks=[0, 2, 4, 6, 8],
               yticks=[0, np.pi, 2 * np.pi], yticklabels=["0", r"$\pi$", r"$2\pi$"])
        ax.set_ylabel("angle / rad", labelpad=1)
        ax.tick_params(length=2.5, pad=2)
        cax = fig.add_axes([0.42 / width, 2.03 / height, (width - 0.54) / width, 0.07 / height])
        colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
        colorbar.set_label("relative value", labelpad=3)
        cax.xaxis.set_label_position("top")
        cax.xaxis.set_ticks_position("top")
        cax.tick_params(length=2, pad=1.5)
        panel = target.with_name(f"{target.stem}_{name}.pdf")
        save(fig, panel)
        preview(panel)
        panels.append(panel)
    assemble(panels, target, legend=False)
    return [*panels, target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--only-space-time", action="store_true",
                        help="Update only the illustration and its manifest entries; require an existing manifest.")
    args = parser.parse_args()
    configure()
    inputs = [args.results_dir / "smoothing_evaluation_summary.csv", args.results_dir / "reference_stability.json"]
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("Both frozen summary CSV and reference diagnostics are required")
    hashes = {path.name: digest(path) for path in inputs}
    rows = _read_rows(inputs[0])
    reference_mean, reference_l1 = _read_reference_diagnostics(inputs[1])
    output = args.figures_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "publication_manifest.json"
    if args.only_space_time:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["inputs_sha256"] != hashes:
            raise ValueError("Existing publication manifest does not match the frozen inputs")
        for name, checksum in manifest["pdf_sha256"].items():
            if digest(output / name) != checksum:
                raise ValueError(f"Existing figure does not match the manifest: {name}")
    jobs = [("smoothing_runtime_by_parameter", "runtime_s_median", False, None),
            ("smoothing_mean_error_by_parameter", "mean_error_rad", False, reference_mean),
            ("smoothing_l1_error_by_parameter", "l1_error", False, reference_l1),
            ("smoothing_mean_error_by_runtime", "mean_error_rad", True, reference_mean),
            ("smoothing_l1_error_by_runtime", "l1_error", True, reference_l1)]
    generated = []
    for name, metric, runtime, reference in ([] if args.only_space_time else jobs):
        path = output / f"{name}.pdf"
        chart(rows, metric, runtime, reference, path, COLUMN_WIDTH, 2.45, own_legend=True)
        preview(path)
        generated.append(path)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for name, indices, titles in ([] if args.only_space_time else [
            ("smoothing_accuracy_by_parameter", [1, 2], ["(a) Mean-direction error", r"(b) $L^1$ distance to reference"]),
            ("smoothing_runtime_accuracy_summary", [0, 3, 4], ["(a) Runtime scaling", "(b) Mean-direction error", r"(c) $L^1$ distance to reference"]),
        ]):
            width = (TEXT_WIDTH - 0.14 * (len(indices) - 1)) / len(indices)
            panels = []
            for number, (index, title) in enumerate(zip(indices, titles)):
                _, metric, runtime, reference = jobs[index]
                panel = directory / f"panel-{number}.pdf"
                chart(rows, metric, runtime, reference, panel, width, 2.18, title=title)
                panels.append(panel)
            path = output / f"{name}.pdf"
            assemble(panels, path, legend=True)
            generated.append(path)
    generated.extend(hero(output / "smoothing_space_time.pdf"))
    if hashes != {path.name: digest(path) for path in inputs}:
        raise RuntimeError("Rendering changed numerical inputs")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unknown"
    if not args.only_space_time:
        manifest = {"schema_version": 1, "renderer_commit": revision,
                    "style_reference": "FlorianPfaff/S3F-Improved-Paper@044914c0b002816a6fa3a8a86d55d03831cf0d50",
                    "inputs_sha256": hashes, "pdf_sha256": {},
                    "accuracy_statistic": "arithmetic mean; PF IQR endpoints", "runtime_statistic": "median and IQR",
                    "line_order": "increasing grid/particle count; no coordinate jitter",
                    "reference_diagnostics": {"mean_error_rad": reference_mean, "l1_error": reference_l1},
                    "plotted_rows": rows}
    manifest["pdf_sha256"].update({p.name: digest(p) for p in generated})
    manifest["space_time_subfigures"] = {
        "renderer": "scripts/render_publication_figures.py",
        "renderer_base_commit": revision,
        "sources_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                           ("render_publication_figures.py", "plot_smoothing_hero.py", "paper_plot_style.py")},
        "panel_width_pdf_pt": SPACE_TIME_WIDTH * 72,
        "panel_height_pdf_pt": SPACE_TIME_HEIGHT * 72,
        "case": SPACE_TIME_CASE,
        "normalization": "independent min-max rescaling within each time column",
        "colorbar": "horizontal, above each heatmap; relative value",
        "captions": "LaTeX subfloats; no embedded panel titles",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
