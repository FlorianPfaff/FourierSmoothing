import csv
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
    "l1_error",
]


@pytest.mark.parametrize(
    ("slug", "expected_rows"),
    [("figfan", 9), ("figfdn", 9), ("pf", 5), ("pwc", 9)],
)
def test_public_paper_evidence_runtime_intervals(slug, expected_rows):
    path = EVIDENCE / f"smoothing_evaluation_{slug}.dat"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        assert reader.fieldnames == HEADER
        rows = list(reader)
    assert len(rows) == expected_rows

    for row in rows:
        median = float(row["runtime_median_ms"])
        q25 = float(row["runtime_q25_ms"])
        q75 = float(row["runtime_q75_ms"])
        assert q25 <= median <= q75
        assert float(row["runtime_err_low_ms"]) == pytest.approx(median - q25)
        assert float(row["runtime_err_high_ms"]) == pytest.approx(q75 - median)
        assert float(row["runtime_mean_ms"]) > 0.0
        assert float(row["mean_error"]) >= 0.0
        assert float(row["l1_error"]) >= 0.0


def test_public_paper_evidence_provenance_and_gain_summary():
    metadata = json.loads(
        (EVIDENCE / "smoothing_evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == 2
    assert metadata["evaluation_mode"] == "corrected_errors_with_reused_controlled_timings"
    assert metadata["combined_raw_data"]["join_key"] == ["method", "parameter", "repetition"]
    assert metadata["timing_generation"]["generated_columns"] == ["runtime_s"]
    assert metadata["public_evidence"] == {
        "repository": "FlorianPfaff/FourierSmoothing",
        "path": "paper_evidence",
    }

    with (EVIDENCE / "smoothing_gain_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["horizon"] for row in rows] == ["all", "early", "late"]
    assert [int(row["n_time_steps"]) for row in rows] == [19, 10, 9]
