#!/usr/bin/env python3
"""Assemble the public paper-evidence snapshot from regenerated results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper_evidence"
GENERATED = ROOT / "generated_final_evidence"


def main() -> None:
    EVIDENCE.mkdir(exist_ok=True)

    with (GENERATED / "evaluation/smoothing_evaluation_raw.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        raw_rows = list(csv.DictReader(handle))
    accuracy_fields = [
        "method",
        "parameter",
        "repetition",
        "mean_error_rad",
        "max_mean_error_rad",
        "l1_error",
        "max_l1_error",
        "min_evaluated_density",
        "max_normalization_error",
    ]
    with (EVIDENCE / "smoothing_accuracy_raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=accuracy_fields, lineterminator="\n")
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({field: row[field] for field in accuracy_fields})

    with (GENERATED / "evaluation/smoothing_evaluation_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    accuracy_summary_fields = [
        field for field in summary_rows[0] if not field.startswith("runtime_s_")
    ]
    with (EVIDENCE / "smoothing_accuracy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=accuracy_summary_fields, lineterminator="\n"
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row[field] for field in accuracy_summary_fields})

    summary_by_key = {
        (row["method"], int(row["parameter"])): row for row in summary_rows
    }
    slugs = {"FIGFAN": "figfan", "FIGFDN": "figfdn", "PF": "pf", "PWC": "pwc"}
    old_header = [
        "n",
        "runtime_mean_ms",
        "runtime_median_ms",
        "runtime_q25_ms",
        "runtime_q75_ms",
        "runtime_err_low_ms",
        "runtime_err_high_ms",
        "mean_error",
        "l1_error",
    ]
    new_header = [
        "n",
        "runtime_mean_ms",
        "runtime_median_ms",
        "runtime_q25_ms",
        "runtime_q75_ms",
        "runtime_err_low_ms",
        "runtime_err_high_ms",
        "mean_error",
        "mean_error_q25",
        "mean_error_q75",
        "mean_error_err_low",
        "mean_error_err_high",
        "l1_error",
        "l1_error_q25",
        "l1_error_q75",
        "l1_error_err_low",
        "l1_error_err_high",
    ]
    for method, slug in slugs.items():
        path = EVIDENCE / f"smoothing_evaluation_{slug}.dat"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
            if reader.fieldnames != old_header:
                raise RuntimeError(
                    f"unexpected controlled timing data schema in {path}: {reader.fieldnames}"
                )
            controlled = list(reader)
        lines = [" ".join(new_header)]
        for row in controlled:
            summary = summary_by_key[(method, int(row["n"]))]
            mean_error = float(summary["mean_error_rad_mean"])
            mean_q25 = float(summary["mean_error_rad_q25"])
            mean_q75 = float(summary["mean_error_rad_q75"])
            l1_error = float(summary["l1_error_mean"])
            l1_q25 = float(summary["l1_error_q25"])
            l1_q75 = float(summary["l1_error_q75"])
            values = [
                row["n"],
                row["runtime_mean_ms"],
                row["runtime_median_ms"],
                row["runtime_q25_ms"],
                row["runtime_q75_ms"],
                row["runtime_err_low_ms"],
                row["runtime_err_high_ms"],
                f"{mean_error:.12g}",
                f"{mean_q25:.12g}",
                f"{mean_q75:.12g}",
                f"{mean_error - mean_q25:.12g}",
                f"{mean_q75 - mean_error:.12g}",
                f"{l1_error:.12g}",
                f"{l1_q25:.12g}",
                f"{l1_q75:.12g}",
                f"{l1_error - l1_q25:.12g}",
                f"{l1_q75 - l1_error:.12g}",
            ]
            lines.append(" ".join(values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for source, target in (
        (
            GENERATED / "gain/smoothing_gain_raw.csv",
            EVIDENCE / "smoothing_gain_raw.csv",
        ),
        (
            GENERATED / "gain/smoothing_gain_summary.csv",
            EVIDENCE / "smoothing_gain_summary.csv",
        ),
        (
            GENERATED / "diagnostics/reference_first_moments.csv",
            EVIDENCE / "reference_first_moments.csv",
        ),
        (
            GENERATED / "diagnostics/reference_stability.json",
            EVIDENCE / "reference_stability.json",
        ),
        (
            GENERATED / "diagnostics/adjoint_validation.csv",
            EVIDENCE / "adjoint_validation.csv",
        ),
        (
            GENERATED / "evaluation/smoothing_evaluation_metadata.json",
            EVIDENCE / "accuracy_regeneration_environment.json",
        ),
    ):
        target.write_bytes(source.read_bytes())

    metadata_path = EVIDENCE / "smoothing_evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 3
    metadata["public_evidence"].update(
        {
            "accuracy_raw": "paper_evidence/smoothing_accuracy_raw.csv",
            "accuracy_summary": "paper_evidence/smoothing_accuracy_summary.csv",
            "accuracy_regeneration_environment": "paper_evidence/accuracy_regeneration_environment.json",
            "gain_raw": "paper_evidence/smoothing_gain_raw.csv",
            "gain_summary": "paper_evidence/smoothing_gain_summary.csv",
            "reference_first_moments": "paper_evidence/reference_first_moments.csv",
            "reference_stability": "paper_evidence/reference_stability.json",
            "adjoint_validation": "paper_evidence/adjoint_validation.csv",
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readme = """# Public evidence snapshot for the FIGF smoothing paper

This directory is the public numerical-evidence snapshot referenced by the manuscript *Fixed-Interval Smoothing with the Fourier-Interpreted Grid Filter*.

It contains:

- one generated PGFPlots data file per evaluated method, including controlled runtime mean, median, quartiles and IQR errors, plus mean-direction and $L^1$ accuracy means and quartiles;
- `smoothing_accuracy_raw.csv`, the complete 30-repetition accuracy rows with runtime removed to avoid mixing hosted-runner timings with the controlled timing study;
- `smoothing_accuracy_summary.csv` and `accuracy_regeneration_environment.json`;
- complete raw and summarized 500-sequence filtering-versus-smoothing state-error results;
- the three high-sample reference first-moment sequences and a 65,535-versus-131,071-cell reference-refinement diagnostic;
- weighted-adjoint validation for two-dimensional additive and one-dimensional nonadditive transitions;
- split provenance metadata for the corrected errors and retained controlled timings.

The exact per-repetition controlled timing rows were produced on the designated timing host and remain identified by their Git blob in the manuscript repository. Every timing value used in the paper—mean, median, quartiles and IQR errors—is reproduced in the public method data files. Accuracy and state-error raw rows are fully public here.

Accuracy values are arithmetic means over repetitions. Runtime plots use the median with the first and third quartiles because PF/FFBSi timings are right-skewed. PF accuracy intervals use the corresponding accuracy quartiles.
"""
    (EVIDENCE / "README.md").write_text(readme, encoding="utf-8")

    manifest_lines = []
    for path in sorted(EVIDENCE.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            manifest_lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            )
    (EVIDENCE / "SHA256SUMS").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
