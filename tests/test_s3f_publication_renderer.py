"""Regression checks for the canonical publication-only renderer."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("matplotlib")
ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_size_rendering_preserves_frozen_evidence(tmp_path: Path) -> None:
    evidence = ROOT / "paper_evidence"
    before = {str(p.relative_to(evidence)): sha(p) for p in evidence.rglob("*") if p.is_file()}
    subprocess.run([sys.executable, str(ROOT / "scripts/render_publication_figures.py"),
                    "--results-dir", str(evidence), "--figures-dir", str(tmp_path)],
                   check=True, timeout=120)
    after = {str(p.relative_to(evidence)): sha(p) for p in evidence.rglob("*") if p.is_file()}
    assert before == after
    manifest = json.loads((tmp_path / "publication_manifest.json").read_text())
    assert len(manifest["plotted_rows"]) == 32
    assert len(manifest["pdf_sha256"]) == 8
    for name, checksum in manifest["pdf_sha256"].items():
        path = tmp_path / name
        assert sha(path) == checksum
        assert path.with_suffix(".png").is_file()
        with fitz.open(path) as document:
            assert len(document) == 1
            page = document[0]
            wide = name in {"smoothing_accuracy_by_parameter.pdf", "smoothing_runtime_accuracy_summary.pdf", "smoothing_space_time.pdf"}
            expected_width = (516.0 if wide else 252.0) / 72.27 * 72
            assert page.rect.width == pytest.approx(expected_width, abs=0.02)
            assert all(font[2] != "Type3" for font in page.get_fonts(full=True))
            # No labels silently shrink when the panels are assembled.
            spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                     for line in block["lines"] for span in line["spans"] if span["text"].strip()]
            assert spans
            assert min(span["size"] for span in spans) >= 4.8  # Math superscripts are smaller.


def test_shared_style_keeps_overlapping_methods_distinguishable() -> None:
    spec = importlib.util.spec_from_file_location("publication_style", ROOT / "scripts/paper_plot_style.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first, second = module.line_style("FIGFAN"), module.line_style("FIGFDN")
    assert first["marker"] != second["marker"]
    assert first["linestyle"] != second["linestyle"]
    assert first["markersize"] < second["markersize"]
    assert second["markerfacecolor"] == "white"
    assert module.DRAW_ORDER.index("FIGFDN") < module.DRAW_ORDER.index("FIGFAN")
