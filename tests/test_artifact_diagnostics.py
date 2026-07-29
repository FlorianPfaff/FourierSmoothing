import csv
import json
import subprocess
import sys
from pathlib import Path


def test_adjoint_validation_script(tmp_path):
    subprocess.run(
        [sys.executable, "scripts/run_adjoint_validation.py", "--output-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    with (tmp_path / "adjoint_validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert max(float(row["absolute_error"]) for row in rows) < 1.0e-11


def test_reference_diagnostics_smoke(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/run_reference_diagnostics.py",
            "--output-dir",
            str(tmp_path),
            "--time-steps",
            "3",
            "--low-grid-size",
            "31",
            "--high-grid-size",
            "63",
            "--reference-particles",
            "200",
            "--reference-repetitions",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    summary = json.loads((tmp_path / "reference_stability.json").read_text(encoding="utf-8"))
    assert summary["max_between_run_mean_direction_deviation_rad"] >= 0.0
    assert summary["pwc_reference_refinement_mean_l1"] >= 0.0
