#!/usr/bin/env python
"""Publish a final-code controlled timing run into ``paper_evidence``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


METHOD_SLUGS = ("figfan", "figfdn", "pf", "pwc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--plot-data-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=Path("paper_evidence"))
    parser.add_argument("--implementation-sha", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = args.evidence_dir
    evidence.mkdir(parents=True, exist_ok=True)

    combined_source = args.results_dir / "smoothing_evaluation_raw.csv"
    combined_target = evidence / "smoothing_evaluation_raw.csv"
    summary_target = evidence / "smoothing_evaluation_summary.csv"
    shutil.copy2(combined_source, combined_target)
    shutil.copy2(args.results_dir / "smoothing_evaluation_summary.csv", summary_target)

    with combined_source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timing_fields = ["method", "parameter", "repetition", "runtime_s"]
    timing_target = evidence / "smoothing_runtime_raw.csv"
    with timing_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in timing_fields})

    for slug in METHOD_SLUGS:
        shutil.copy2(
            args.plot_data_dir / f"smoothing_evaluation_{slug}.dat",
            evidence / f"smoothing_evaluation_{slug}.dat",
        )

    metadata = json.loads(
        (args.results_dir / "smoothing_evaluation_metadata.json").read_text(encoding="utf-8")
    )
    metadata["schema_version"] = 4
    metadata["evaluation_mode"] = "final_release_split_accuracy_timing"
    metadata["error_generation"]["implementation_merged_commit"] = args.implementation_sha
    metadata["timing_generation"]["implementation_git_commit"] = args.implementation_sha
    metadata["timing_generation"]["raw_path"] = str(timing_target.as_posix())
    metadata["timing_generation"]["raw_sha256"] = _sha256(timing_target)
    metadata["combined_raw_data"] = {
        "path": str(combined_target.as_posix()),
        "sha256": _sha256(combined_target),
        "join_key": ["method", "parameter", "repetition"],
        "assembly": (
            "Published accuracy columns and final-release timing columns were joined "
            "after exact key equality."
        ),
    }
    metadata["public_evidence"] = {
        "repository": "FlorianPfaff/FourierSmoothing",
        "path": str(evidence.as_posix()),
        "accuracy_raw": str((evidence / "smoothing_accuracy_raw.csv").as_posix()),
        "timing_raw": str(timing_target.as_posix()),
        "combined_raw": str(combined_target.as_posix()),
        "summary": str(summary_target.as_posix()),
        "gain_raw": str((evidence / "smoothing_gain_raw.csv").as_posix()),
        "gain_summary": str((evidence / "smoothing_gain_summary.csv").as_posix()),
        "reference_first_moments": str((evidence / "reference_first_moments.csv").as_posix()),
        "reference_stability": str((evidence / "reference_stability.json").as_posix()),
        "adjoint_validation": str((evidence / "adjoint_validation.csv").as_posix()),
    }
    (evidence / "smoothing_evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (evidence / "README.md").write_text(
        f"""# Public evidence snapshot for the FIGF smoothing paper

This directory is the public numerical-evidence snapshot referenced by the manuscript *Fixed-Interval Smoothing with the Fourier-Interpreted Grid Filter*.

It contains:

- one generated PGFPlots data file per evaluated method, including final-release controlled runtime mean, median, quartiles and IQR errors, plus mean-direction and $L^1$ accuracy means and quartiles;
- `smoothing_accuracy_raw.csv`, the complete 30-repetition accuracy rows;
- `smoothing_runtime_raw.csv`, the complete 30-repetition controlled timing rows measured on implementation commit `{args.implementation_sha}`;
- `smoothing_evaluation_raw.csv`, the exact keywise combination used for the paper figures and summary;
- complete raw and summarized 500-sequence filtering-versus-smoothing state-error results;
- the three high-sample reference first-moment sequences and the 65,535-versus-131,071-cell reference-refinement diagnostic;
- weighted-adjoint validation for two-dimensional additive and one-dimensional nonadditive transitions;
- split provenance metadata for accuracy and final-release timing generation.

Accuracy values are arithmetic means over repetitions. Runtime plots use the median with the first and third quartiles because PF/FFBSi timings are right-skewed. PF accuracy intervals use the corresponding accuracy quartiles.
""",
        encoding="utf-8",
    )

    manifest = []
    for path in sorted(evidence.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            manifest.append(f"{_sha256(path)}  {path.name}")
    (evidence / "SHA256SUMS").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
