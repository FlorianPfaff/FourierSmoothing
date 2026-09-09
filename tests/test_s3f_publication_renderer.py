"""Regression checks for the canonical publication-only renderer."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
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
    assert len(manifest["pdf_sha256"]) == 11
    profile = manifest["space_time_subfigures"]
    assert profile["case"] == dict(grid_size=255, time_steps=9, likelihood_sharpness=7.0,
                                   noise_concentration=2.2, prior_concentration=0.45)
    for name, checksum in profile["sources_sha256"].items():
        assert sha(ROOT / "scripts" / name) == checksum
    for name, checksum in manifest["pdf_sha256"].items():
        path = tmp_path / name
        assert sha(path) == checksum
        assert path.with_suffix(".png").is_file()
        with fitz.open(path) as document:
            assert len(document) == 1
            page = document[0]
            wide = name in {"smoothing_accuracy_by_parameter.pdf", "smoothing_runtime_accuracy_summary.pdf", "smoothing_space_time.pdf"}
            space_time_panel = name.startswith("smoothing_space_time_")
            width_pt = 0.32 * 516.0 if space_time_panel else (516.0 if wide else 252.0)
            expected_width = width_pt / 72.27 * 72
            assert page.rect.width == pytest.approx(expected_width, abs=0.02)
            assert all(font[2] != "Type3" for font in page.get_fonts(full=True))
            # No labels silently shrink when the panels are assembled.
            spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                     for line in block["lines"] for span in line["spans"] if span["text"].strip()]
            assert spans
            assert min(span["size"] for span in spans) >= 4.8  # Math superscripts are smaller.
            if space_time_panel:
                assert all(page.rect.contains(fitz.Rect(span["bbox"])) for span in spans)
                assert page.rect.height == pytest.approx(profile["panel_height_pdf_pt"])
                labels = [s for s in spans if s["text"].strip() == "relative value"]
                assert len(labels) == 1
                heatmap = max(page.get_image_info(), key=lambda info: fitz.Rect(info["bbox"]).get_area())
                assert labels[0]["bbox"][3] < heatmap["bbox"][1]
                assert not any(title in page.get_text() for title in ("(a)", "(b)", "(c)", "density", "information"))

    # A targeted refresh must preserve paper-side runtime overrides and provenance.
    manifest["runtime_subfigures"] = {"preserved": True}
    manifest_path = tmp_path / "publication_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    untouched = {p.name: sha(p) for p in tmp_path.iterdir()
                 if p.suffix in {".pdf", ".png"} and not p.name.startswith("smoothing_space_time")}
    command = [sys.executable, str(ROOT / "scripts/render_publication_figures.py"),
               "--results-dir", str(evidence), "--figures-dir", str(tmp_path), "--only-space-time"]
    subprocess.run(command, check=True, timeout=60)
    refreshed = json.loads(manifest_path.read_text())
    for key in manifest.keys() - {"pdf_sha256", "space_time_subfigures"}:
        assert refreshed[key] == manifest[key]
    for name, checksum in untouched.items():
        assert sha(tmp_path / name) == checksum
    for name, checksum in refreshed["pdf_sha256"].items():
        assert sha(tmp_path / name) == checksum
    assert before == {str(p.relative_to(evidence)): sha(p) for p in evidence.rglob("*") if p.is_file()}

    refreshed["inputs_sha256"]["reference_stability.json"] = "outdated"
    manifest_path.write_text(json.dumps(refreshed), encoding="utf-8")
    before_failure = {p.name: sha(p) for p in tmp_path.iterdir() if p.is_file()}
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "does not match the frozen inputs" in result.stderr
    assert before_failure == {p.name: sha(p) for p in tmp_path.iterdir() if p.is_file()}


def test_space_time_panels_preserve_fields_and_mean_traces(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import render_publication_figures as renderer
    from plot_smoothing_hero import _make_smoothing_case, _column_normalize

    fields = _make_smoothing_case(**renderer.SPACE_TIME_CASE)
    renderer.configure()
    original_save = renderer.save
    captured = []

    def checked_save(fig, path):
        index = len(captured)
        ax, cax = fig.axes
        np.testing.assert_array_equal(ax.images[0].get_array(), _column_normalize(fields[index]).T)
        assert ax.images[0].get_clim() == (0, 1)
        np.testing.assert_allclose(ax.get_ylim(), [0, 2 * np.pi])
        assert not ax.get_title()
        assert len(ax.lines) == (0 if index == 1 else 1)
        if ax.lines:
            times, means = ax.lines[0].get_data()
            finite = np.isfinite(means)
            np.testing.assert_array_equal(times[finite], np.arange(9))
            angles = np.arange(255) * (2 * np.pi / 255)
            expected = np.mod(np.angle(fields[index] @ np.exp(1j * angles)), 2 * np.pi)
            np.testing.assert_allclose(means[finite], expected)
        assert cax.get_position().y0 > ax.get_position().y1
        assert cax.get_xlabel() == "relative value"
        assert cax.xaxis.get_label_position() == cax.xaxis.get_ticks_position() == "top"
        captured.append(path)
        original_save(fig, path)

    monkeypatch.setattr(renderer, "save", checked_save)
    paths = renderer.hero(tmp_path / "smoothing_space_time.pdf")
    assert len(captured) == 3
    assert len(paths) == 4


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
