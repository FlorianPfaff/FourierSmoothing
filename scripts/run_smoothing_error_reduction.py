#!/usr/bin/env python
"""Generate state-truth filtering-versus-smoothing gain CSVs for the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fourier_smoothing import run_smoothing_gain_evaluation, write_smoothing_gain_csv


SUMMARY_COLUMNS = [
    "horizon",
    "n_trials",
    "n_time_steps",
    "filter_mae_rad",
    "smoother_mae_rad",
    "reduction_percent",
    "reduction_ci_low_percent",
    "reduction_ci_high_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=1023)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--prior-concentration", type=float, default=1.0)
    parser.add_argument("--noise-concentration", type=float, default=8.0)
    parser.add_argument("--likelihood-concentration", type=float, default=12.0)
    parser.add_argument("--outlier-probability", type=float, default=0.3)
    parser.add_argument("--outlier-offset", type=float, default=2.35)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--raw-filename", default="smoothing_gain_raw.csv")
    parser.add_argument("--summary-filename", default="smoothing_gain_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_smoothing_gain_evaluation(
        n_trials=args.n_trials,
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        prior_concentration=args.prior_concentration,
        noise_concentration=args.noise_concentration,
        likelihood_concentration=args.likelihood_concentration,
        outlier_probability=args.outlier_probability,
        outlier_offset=args.outlier_offset,
        seed=args.seed,
    )
    raw_path = write_smoothing_gain_csv(rows, args.output_dir / args.raw_filename)
    summaries = _summarize_gain_rows(
        [row.as_dict() for row in rows],
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed + 1,
    )
    summary_path = _write_summary_csv(summaries, args.output_dir / args.summary_filename)
    print(raw_path)
    print(summary_path)


def _summarize_gain_rows(
    records: list[dict[str, int | float]],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> list[dict[str, int | float | str]]:
    if not records:
        raise ValueError("records must not be empty.")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive.")

    n_trials = max(int(record["trial"]) for record in records) + 1
    time_steps = max(int(record["time_step"]) for record in records) + 1
    filter_errors = np.full((n_trials, time_steps), np.nan, dtype=float)
    smoother_errors = np.full((n_trials, time_steps), np.nan, dtype=float)
    for record in records:
        trial = int(record["trial"])
        time_step = int(record["time_step"])
        filter_errors[trial, time_step] = float(record["filter_error_rad"])
        smoother_errors[trial, time_step] = float(record["smoother_error_rad"])
    if not np.all(np.isfinite(filter_errors)) or not np.all(np.isfinite(smoother_errors)):
        raise ValueError("records must contain one finite row for every trial and time step.")

    final_exclusive = time_steps - 1
    split = (final_exclusive + 1) // 2
    horizons = [
        ("all", np.arange(0, final_exclusive)),
        ("early", np.arange(0, split)),
        ("late", np.arange(split, final_exclusive)),
    ]
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n_trials, size=(bootstrap_repetitions, n_trials))

    summaries: list[dict[str, int | float | str]] = []
    for name, indices in horizons:
        trial_filter_mae = np.mean(filter_errors[:, indices], axis=1)
        trial_smoother_mae = np.mean(smoother_errors[:, indices], axis=1)
        filter_mae = float(np.mean(trial_filter_mae))
        smoother_mae = float(np.mean(trial_smoother_mae))
        bootstrap_filter = np.mean(trial_filter_mae[bootstrap_indices], axis=1)
        bootstrap_smoother = np.mean(trial_smoother_mae[bootstrap_indices], axis=1)
        bootstrap_reduction = 100.0 * (1.0 - bootstrap_smoother / bootstrap_filter)
        summaries.append(
            {
                "horizon": name,
                "n_trials": n_trials,
                "n_time_steps": int(indices.size),
                "filter_mae_rad": filter_mae,
                "smoother_mae_rad": smoother_mae,
                "reduction_percent": 100.0 * (1.0 - smoother_mae / filter_mae),
                "reduction_ci_low_percent": float(np.quantile(bootstrap_reduction, 0.025)),
                "reduction_ci_high_percent": float(np.quantile(bootstrap_reduction, 0.975)),
            }
        )
    return summaries


def _write_summary_csv(rows: list[dict[str, int | float | str]], output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    main()
