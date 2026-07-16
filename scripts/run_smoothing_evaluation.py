#!/usr/bin/env python
"""Generate FIGF/PWC/PF smoothing evaluation CSVs for the paper."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from fourier_smoothing import (
    SmoothingEvaluationRow,
    run_smoothing_evaluation,
    run_smoothing_runtime_evaluation,
    write_smoothing_evaluation_csv,
)


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
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=9)
    parser.add_argument("--likelihood-sharpness", type=float, default=5.0)
    parser.add_argument("--noise-concentration", type=float, default=4.0)
    parser.add_argument("--l1-reference-grid-size", type=int, default=65_535)
    parser.add_argument("--mean-reference-particles", type=int, default=1_000_000)
    parser.add_argument("--mean-reference-repetitions", type=int, default=3)
    parser.add_argument("--particle-kde-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--pwc-quadrature-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--raw-filename", default="smoothing_evaluation_raw.csv")
    parser.add_argument("--summary-filename", default="smoothing_evaluation_summary.csv")
    parser.add_argument("--metadata-filename", default="smoothing_evaluation_metadata.json")
    parser.add_argument(
        "--reuse-error-raw",
        type=Path,
        help="Reuse error metrics from a compatible raw CSV and measure only runtimes on this host.",
    )
    parser.add_argument("--error-source-host", help="Host that generated --reuse-error-raw, for provenance.")
    parser.add_argument("--error-source-git-commit", help="Code revision that generated --reuse-error-raw.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = dt.datetime.now(dt.timezone.utc)
    load_before = _load_average()
    if args.reuse_error_raw is None:
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
            mean_reference_repetitions=args.mean_reference_repetitions,
            particle_kde_bandwidth_scale=args.particle_kde_bandwidth_scale,
            pwc_quadrature_points=args.pwc_quadrature_points,
            seed=args.seed,
        )
    else:
        timing_rows = run_smoothing_runtime_evaluation(
            figf_grid_sizes=args.figf_grid_sizes,
            pwc_grid_sizes=args.pwc_grid_sizes,
            pf_particle_counts=args.pf_particle_counts,
            repetitions=args.repetitions,
            time_steps=args.time_steps,
            likelihood_sharpness=args.likelihood_sharpness,
            noise_concentration=args.noise_concentration,
            particle_likelihood_grid_size=args.l1_reference_grid_size,
            pwc_quadrature_points=args.pwc_quadrature_points,
            seed=args.seed,
        )
        rows = _merge_error_rows_with_timings(args.reuse_error_raw, timing_rows)
    raw_path = write_smoothing_evaluation_csv(rows, args.output_dir / args.raw_filename)
    summary = _summarize_rows([row.as_dict() for row in rows])
    summary_path = _write_summary_csv(summary, args.output_dir / args.summary_filename)
    finished_at = dt.datetime.now(dt.timezone.utc)
    metadata_path = _write_metadata(
        args,
        args.output_dir / args.metadata_filename,
        started_at=started_at,
        finished_at=finished_at,
        load_before=load_before,
        load_after=_load_average(),
    )
    print(raw_path)
    print(summary_path)
    print(metadata_path)


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


def _merge_error_rows_with_timings(error_path: Path, timing_rows) -> list[SmoothingEvaluationRow]:
    errors = {}
    with Path(error_path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            key = (record["method"], int(record["parameter"]), int(record["repetition"]))
            if key in errors:
                raise ValueError(f"Duplicate error row for {key!r}.")
            errors[key] = record

    timings = {}
    for row in timing_rows:
        key = (row.method, int(row.parameter), int(row.repetition))
        if key in timings:
            raise ValueError(f"Duplicate timing row for {key!r}.")
        timings[key] = row

    if errors.keys() != timings.keys():
        missing_timings = sorted(errors.keys() - timings.keys())[:5]
        missing_errors = sorted(timings.keys() - errors.keys())[:5]
        raise ValueError(
            "Error and timing row keys differ; "
            f"missing timings={missing_timings}, missing errors={missing_errors}."
        )

    merged = []
    for key, timing in timings.items():
        record = errors[key]
        merged.append(
            SmoothingEvaluationRow(
                method=timing.method,
                parameter=timing.parameter,
                repetition=timing.repetition,
                runtime_s=timing.runtime_s,
                mean_error_rad=float(record["mean_error_rad"]),
                max_mean_error_rad=float(record["max_mean_error_rad"]),
                l1_error=float(record["l1_error"]),
                max_l1_error=float(record["max_l1_error"]),
                min_evaluated_density=float(record["min_evaluated_density"]),
                max_normalization_error=float(record["max_normalization_error"]),
            )
        )
    return merged


def _fallback_summarize_parameter_sweep_records(records: list[dict[str, str | int | float]]) -> list[dict[str, object]]:
    from collections import defaultdict
    from statistics import mean, pstdev

    import numpy as np

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
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{metric}_q75"] = float(np.quantile(values, 0.75))
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
        *(f"{metric}_median" for metric in METRIC_NAMES),
        *(f"{metric}_q25" for metric in METRIC_NAMES),
        *(f"{metric}_q75" for metric in METRIC_NAMES),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_metadata(
    args: argparse.Namespace,
    output_path: Path,
    *,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    load_before: tuple[float, float, float] | None,
    load_after: tuple[float, float, float] | None,
) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    metadata = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_s": (finished_at - started_at).total_seconds(),
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "load_average_before": load_before,
        "load_average_after": load_after,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "pyrecest_available": importlib.util.find_spec("pyrecest") is not None,
        "git_commit": os.environ.get("FOURIER_SMOOTHING_GIT_COMMIT") or _git_commit(repository_root),
        "source_tree_sha256": _source_tree_hash(repository_root),
        "evaluation_mode": "split_accuracy_timing" if args.reuse_error_raw is not None else "combined",
        "error_source": _error_source_metadata(args),
        "configuration": {
            "figf_grid_sizes": args.figf_grid_sizes,
            "pwc_grid_sizes": args.pwc_grid_sizes,
            "pf_particle_counts": args.pf_particle_counts,
            "repetitions": args.repetitions,
            "time_steps": args.time_steps,
            "likelihood_sharpness": args.likelihood_sharpness,
            "noise_concentration": args.noise_concentration,
            "l1_reference_grid_size": args.l1_reference_grid_size,
            "mean_reference_particles": args.mean_reference_particles,
            "mean_reference_repetitions": args.mean_reference_repetitions,
            "particle_kde_bandwidth_scale": args.particle_kde_bandwidth_scale,
            "pwc_quadrature_points": args.pwc_quadrature_points,
            "seed": args.seed,
        },
        "runtime_scope": {
            "included": ["forward filter", "backward smoother"],
            "excluded": [
                "reference generation",
                "transition-kernel construction",
                "dense FIGF interpolation",
                "PWC densification",
                "particle KDE reconstruction",
            ],
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_average() -> tuple[float, float, float] | None:
    try:
        return tuple(float(value) for value in os.getloadavg())
    except (AttributeError, OSError):
        return None


def _git_commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _source_tree_hash(repository_root: Path) -> str:
    digest = hashlib.sha256()
    source_paths = sorted((repository_root / "src").rglob("*.py"))
    source_paths.extend(sorted((repository_root / "scripts").glob("*.py")))
    source_paths.extend(path for path in (repository_root / "pyproject.toml",) if path.exists())
    for path in source_paths:
        relative_path = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(4, byteorder="big"))
        digest.update(relative_path)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _error_source_metadata(args: argparse.Namespace) -> dict[str, str | None] | None:
    if args.reuse_error_raw is None:
        return None
    return {
        "path": str(args.reuse_error_raw),
        "sha256": _file_sha256(args.reuse_error_raw),
        "host": args.error_source_host,
        "git_commit": args.error_source_git_commit,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
