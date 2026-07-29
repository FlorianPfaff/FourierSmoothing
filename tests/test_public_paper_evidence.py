import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper_evidence"
HEADER = [
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


@pytest.mark.parametrize(
    ("slug", "expected_rows"),
    [("figfan", 9), ("figfdn", 9), ("pf", 5), ("pwc", 9)],
)
def test_public_paper_evidence_intervals(slug, expected_rows):
    path = EVIDENCE / f"smoothing_evaluation_{slug}.dat"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        assert reader.fieldnames == HEADER
        rows = list(reader)
    assert len(rows) == expected_rows

    for row in rows:
        median = float(row["runtime_median_ms"])
        runtime_q25 = float(row["runtime_q25_ms"])
        runtime_q75 = float(row["runtime_q75_ms"])
        assert runtime_q25 <= median <= runtime_q75
        assert float(row["runtime_err_low_ms"]) == pytest.approx(median - runtime_q25)
        assert float(row["runtime_err_high_ms"]) == pytest.approx(runtime_q75 - median)
        assert float(row["runtime_mean_ms"]) > 0.0

        for prefix in ("mean_error", "l1_error"):
            mean = float(row[prefix])
            q25 = float(row[f"{prefix}_q25"])
            q75 = float(row[f"{prefix}_q75"])
            assert q25 <= mean <= q75
            assert float(row[f"{prefix}_err_low"]) == pytest.approx(mean - q25)
            assert float(row[f"{prefix}_err_high"]) == pytest.approx(q75 - mean)
            assert mean >= 0.0


def test_public_paper_evidence_raw_rows_and_diagnostics():
    with (EVIDENCE / "smoothing_accuracy_raw.csv").open(newline="", encoding="utf-8") as handle:
        accuracy_rows = list(csv.DictReader(handle))
    assert len(accuracy_rows) == 960
    assert "runtime_s" not in accuracy_rows[0]

    with (EVIDENCE / "smoothing_gain_raw.csv").open(newline="", encoding="utf-8") as handle:
        gain_rows = list(csv.DictReader(handle))
    assert len(gain_rows) == 10_000

    with (EVIDENCE / "reference_first_moments.csv").open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    assert len(reference_rows) == 27

    reference = json.loads((EVIDENCE / "reference_stability.json").read_text(encoding="utf-8"))
    assert reference["pwc_reference_refinement_mean_l1"] >= 0.0
    assert reference["max_between_run_mean_direction_deviation_rad"] >= 0.0

    with (EVIDENCE / "adjoint_validation.csv").open(newline="", encoding="utf-8") as handle:
        adjoint_rows = list(csv.DictReader(handle))
    assert len(adjoint_rows) == 3
    assert max(float(row["absolute_error"]) for row in adjoint_rows) < 1.0e-10


def test_public_paper_evidence_provenance_gain_and_checksums():
    metadata = json.loads(
        (EVIDENCE / "smoothing_evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == 3
    assert metadata["evaluation_mode"] == "corrected_errors_with_reused_controlled_timings"
    assert metadata["combined_raw_data"]["join_key"] == ["method", "parameter", "repetition"]
    assert metadata["timing_generation"]["generated_columns"] == ["runtime_s"]
    public = metadata["public_evidence"]
    assert public["repository"] == "FlorianPfaff/FourierSmoothing"
    assert public["path"] == "paper_evidence"
    for key in (
        "accuracy_raw",
        "accuracy_summary",
        "accuracy_regeneration_environment",
        "gain_raw",
        "gain_summary",
        "reference_first_moments",
        "reference_stability",
        "adjoint_validation",
    ):
        assert (ROOT / public[key]).exists()

    with (EVIDENCE / "smoothing_gain_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["horizon"] for row in rows] == ["all", "early", "late"]
    assert [int(row["n_time_steps"]) for row in rows] == [19, 10, 9]

    manifest = {}
    for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    for path in EVIDENCE.iterdir():
        if path.is_file() and path.name != "SHA256SUMS":
            assert manifest[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
