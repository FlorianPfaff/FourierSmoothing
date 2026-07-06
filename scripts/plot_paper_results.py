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

    identity_csv = args.results_dir / "identity_torus_benchmark.csv"
    negativity_csv = args.results_dir / "truncation_negativity_diagnostic.csv"

    written = []
    if identity_csv.exists():
        written.extend(_plot_identity_benchmark(identity_csv, args.figures_dir, args.formats))
    if negativity_csv.exists():
        written.extend(_plot_negativity_diagnostic(negativity_csv, args.figures_dir, args.formats))

    if not written:
        raise FileNotFoundError(
            f"No known result CSVs found in {args.results_dir}. Run the result-generation scripts first."
        )
    for path in written:
        print(path)


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


def _save_all(fig, path_without_suffix: Path, formats: list[str]) -> list[Path]:
    written = []
    for fmt in formats:
        output_path = path_without_suffix.with_suffix(f".{fmt}")
        fig.savefig(output_path, bbox_inches="tight")
        written.append(output_path)
    return written


if __name__ == "__main__":
    main()
