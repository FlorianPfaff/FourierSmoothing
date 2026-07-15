"""LaTeX table writers for Fourier smoothing paper result CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


TABLE_FILENAMES = {
    "identity": "identity_runtime.tex",
    "negativity": "truncation_negativity.tex",
    "particle": "particle_baseline.tex",
    "figf_pwc": "figf_pwc_benchmark.tex",
    "smoothing": "smoothing_evaluation.tex",
}


def write_latex_tables(results_dir: str | Path, tables_dir: str | Path) -> list[Path]:
    """Write LaTeX summary tables for all known result CSVs in ``results_dir``.

    Missing CSV files are skipped. For the main smoothing evaluation, the
    pre-aggregated summary CSV is preferred over the raw repetitions. The
    function returns the paths that were written.
    """

    results_path = Path(results_dir)
    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)

    writers = [
        ("identity_torus_benchmark.csv", tables_path / TABLE_FILENAMES["identity"], _identity_table),
        ("truncation_negativity_diagnostic.csv", tables_path / TABLE_FILENAMES["negativity"], _negativity_table),
        ("particle_smoother_baseline.csv", tables_path / TABLE_FILENAMES["particle"], _particle_table),
        ("figf_pwc_benchmark.csv", tables_path / TABLE_FILENAMES["figf_pwc"], _figf_pwc_table),
        ("smoothing_evaluation_summary.csv", tables_path / TABLE_FILENAMES["smoothing"], _smoothing_summary_table),
        ("smoothing_evaluation_raw.csv", tables_path / TABLE_FILENAMES["smoothing"], _smoothing_raw_table),
    ]

    written: list[Path] = []
    written_outputs: set[Path] = set()
    for csv_name, output_path, writer in writers:
        if output_path in written_outputs:
            continue
        csv_path = results_path / csv_name
        if csv_path.exists():
            rows = _read_csv(csv_path)
            if rows:
                output_path.write_text(writer(rows), encoding="utf-8")
                written.append(output_path)
                written_outputs.add(output_path)
    return written


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _identity_table(rows: Sequence[Mapping[str, str]]) -> str:
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["grid_size"]))].append(row)

    body = []
    for method, grid_size in sorted(grouped, key=lambda item: (item[0], item[1])):
        items = grouped[(method, grid_size)]
        body.append(
            [
                _latex_escape(method),
                str(grid_size),
                _fmt(mean(float(item["runtime_s"]) for item in items)),
                _fmt(mean(float(item["max_abs_difference_to_grid"]) for item in items)),
                _fmt(max(float(item["max_normalization_error"]) for item in items)),
            ]
        )
    return _tabular(
        columns="lrrrr",
        header=["method", "grid size", "runtime [s]", "max diff.", "norm. err."],
        rows=body,
        caption_comment="Identity-torus smoother benchmark summary.",
    )


def _negativity_table(rows: Sequence[Mapping[str, str]]) -> str:
    grouped: dict[tuple[float, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["sharpness"]), int(row["n_coefficients"]))].append(row)

    body = []
    for sharpness, n_coefficients in sorted(grouped, key=lambda item: (item[0], item[1])):
        items = grouped[(sharpness, n_coefficients)]
        body.append(
            [
                _fmt(sharpness),
                str(n_coefficients),
                _fmt(mean(float(item["negative_mass"]) for item in items)),
                _fmt(mean(float(item["max_negative_undershoot"]) for item in items)),
                _fmt(mean(float(item["l1_error_to_dense_grid"]) for item in items)),
            ]
        )
    return _tabular(
        columns="rrrrr",
        header=["sharpness", "coeffs", "neg. mass", "max undershoot", "$L^1$ err."],
        rows=body,
        caption_comment="Truncation-induced negativity diagnostic summary.",
    )


def _particle_table(rows: Sequence[Mapping[str, str]]) -> str:
    grouped: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["n_particles"])].append(row)

    body = []
    for n_particles in sorted(grouped):
        items = grouped[n_particles]
        body.append(
            [
                str(n_particles),
                str(int(round(mean(float(item["n_trajectories"]) for item in items)))),
                _fmt(mean(float(item["runtime_s"]) for item in items)),
                _fmt(mean(float(item["mean_abs_circular_error_to_grid"]) for item in items)),
                _fmt(mean(float(item["max_abs_circular_error_to_grid"]) for item in items)),
            ]
        )
    return _tabular(
        columns="rrrrr",
        header=["particles", "trajectories", "runtime [s]", "mean err. [rad]", "max err. [rad]"],
        rows=body,
        caption_comment="Particle smoother baseline summary.",
    )


def _figf_pwc_table(rows: Sequence[Mapping[str, str]]) -> str:
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["grid_size"]))].append(row)

    body = []
    for method, grid_size in sorted(grouped, key=lambda item: (item[0], item[1])):
        items = grouped[(method, grid_size)]
        body.append(
            [
                _latex_escape(method),
                str(grid_size),
                _fmt(mean(float(item["runtime_s"]) for item in items)),
                _fmt(mean(float(item["mean_l1_error_to_reference"]) for item in items)),
                _fmt(max(float(item["max_l1_error_to_reference"]) for item in items)),
            ]
        )
    return _tabular(
        columns="lrrrr",
        header=["method", "grid size", "runtime [s]", "mean $L^1$", "max $L^1$"],
        rows=body,
        caption_comment="FIGF/PWC reconstruction benchmark summary.",
    )


def _smoothing_summary_table(rows: Sequence[Mapping[str, str]]) -> str:
    body = []
    for row in sorted(rows, key=lambda item: (item["method"], int(item["parameter"]))):
        body.append(
            [
                _latex_escape(row["method"]),
                str(int(row["parameter"])),
                _fmt(float(row["runtime_s_mean"])),
                _fmt(float(row["mean_error_rad_mean"])),
                _fmt(float(row["l1_error_mean"])),
            ]
        )
    return _tabular(
        columns="lrrrr",
        header=["method", "parameter", "runtime [s]", "mean err. [rad]", "$L^1$ err."],
        rows=body,
        caption_comment="Main smoothing evaluation summary.",
    )


def _smoothing_raw_table(rows: Sequence[Mapping[str, str]]) -> str:
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["parameter"]))].append(row)

    body = []
    for method, parameter in sorted(grouped, key=lambda item: (item[0], item[1])):
        items = grouped[(method, parameter)]
        body.append(
            [
                _latex_escape(method),
                str(parameter),
                _fmt(mean(float(item["runtime_s"]) for item in items)),
                _fmt(mean(float(item["mean_error_rad"]) for item in items)),
                _fmt(mean(float(item["l1_error"]) for item in items)),
            ]
        )
    return _tabular(
        columns="lrrrr",
        header=["method", "parameter", "runtime [s]", "mean err. [rad]", "$L^1$ err."],
        rows=body,
        caption_comment="Main smoothing evaluation summary generated from raw repetitions.",
    )


def _tabular(*, columns: str, header: Sequence[str], rows: Sequence[Sequence[str]], caption_comment: str) -> str:
    lines = [
        f"% {caption_comment}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _fmt(value: float) -> str:
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value < 1.0e-3 or abs_value >= 1.0e4):
        return f"{value:.3e}"
    return f"{value:.4g}"


def _latex_escape(value: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    escaped = value
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped
