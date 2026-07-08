#!/usr/bin/env python
"""Generate FIGF/PWC/PF smoothing evaluation CSVs for the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fourier_smoothing import run_smoothing_evaluation, write_smoothing_evaluation_csv


METRIC_NAMES = [
    "runtime_s",
    "mean_error_rad",
    "max_mean_error_rad",
    "l1_error",
    "max_l1_error",
    "min_evaluated_density",
    "max_normalization_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated CSV files. Use the paper repo's results/ directory for paper artifacts.",
    )
    parser.add_argument("--figf-grid-sizes", type=int, nargs="+", default=[15, 31, 63, 127, 255, 511, 1023, 2047, 4095])
    parser.add_argument("--pwc-grid-sizes", type=int, nargs="+", default=[15, 31, 63, 127, 255, 511, 1023, 2047, 4095])
    parser.add_argument("--pf-particle-counts", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=5)
    parser.add_argument("--likelihood-sharpness", type=float, default=5.0)
    parser.add_argument("--noise-concentration", type=float, default=4.0)
    parser.add_argument("--l1-reference-grid-size", type=int, default=8193)
    parser.add_argument("--mean-reference-particles", type=int, default=100_000)
    parser.add_argument("--particle-chunk-size", type=int, default=100_000)
    parser.add_argument("--pwc-quadrature-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--raw-filename", default="smoothing_evaluation_raw.csv")
    parser.add_argument("--summary-filename", default="smoothing_evaluation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_smoothing_evaluation(
        figf_grid_sizes=args.figf_grid_sizes,
        pwc_grid_sizes=args.pwc_grid_sizes,
        pf_particle_counts=args.pf_particle_counts,
        repetitions=args.repetitions,
        time_steps=args.time_steps,
        likelihood_sharpness=args.likelihood_sharpness,
        noise_concentration=args.noise_concentration,
        l1_reference_grid_size=args.l1_reference_grid_size,
        mean_reference_particles=args.mean_reference_particles,
        particle_chunk_size=args.particle_chunk_size,
        pwc_quadrature_points=args.pwc_quadrature_points,
        seed=args.seed,
    )
    raw_path = write_smoothing_evaluation_csv(rows, args.output_dir / args.raw_filename)
    summary = _summarize_rows([row.as_dict() for row in rows])
    summary_path = _write_summary_csv(summary, args.output_dir / args.summary_filename)
    print(raw_path)
    print(summary_path)


def _summarize_rows(records: list[dict[str, str | int | float]]) -> list[dict[str, object]]:
    try:
        from pyrecest.evaluation import summarize_parameter_sweep_records
        return summarize_parameter_sweep_records(records, metric_names=METRIC_NAMES)
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from pyrecest.evaluation.parameter_sweeps import summarize_parameter_sweep_records

        return summarize_parameter_sweep_records(records, metric_names=METRIC_NAMES)
    except (ImportError, ModuleNotFoundError):
        return _fallback_summarize_parameter_sweep_records(records)


def _fallback_summarize_parameter_sweep_records(records: list[dict[str, str | int | float]]) -> list[dict[str, object]]:
    from collections import defaultdict
    from statistics import mean, pstdev

    grouped = defaultdict(list)
    for record in records:
        grouped[(str(record["method"]), int(record["parameter"]))].append(record)
    summaries = []
    for (method, parameter), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "method": method,
            "parameter": parameter,
            "n_repetitions": len(group),
        }
        for metric in METRIC_NAMES:
            values = [float(record[metric]) for record in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = pstdev(values)
        summaries.append(summary)
    return summaries


def _write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "parameter",
        "n_repetitions",
        *(f"{metric}_mean" for metric in METRIC_NAMES),
        *(f"{metric}_std" for metric in METRIC_NAMES),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


if __name__ == "__main__":
    main()
