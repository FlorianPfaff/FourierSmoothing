"""Numerical and selection checks for the representative tracking figure script."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'plot_tracking_example.py'
spec = importlib.util.spec_from_file_location('plot_tracking_example', MODULE_PATH)
plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot
spec.loader.exec_module(plot)


@pytest.fixture(scope='module')
def reproduced(tmp_path_factory):
    path = tmp_path_factory.mktemp('tracking-example')
    meta = plot.prepare_data(plot.Config(), path)
    with np.load(path / 'tracking_example_data.npz') as saved:
        data = {name: saved[name] for name in saved.files}
    return meta, data, path


def test_reproduces_published_500_trial_maes_and_bootstrap(reproduced):
    meta, _, _ = reproduced
    results = meta['all_trials']
    np.testing.assert_allclose(
        [results['filter_mae_rad'], results['smoother_mae_rad']],
        plot.EXPECTED_PUBLISHED_MAES, rtol=0, atol=1e-12)
    np.testing.assert_allclose(results['reduction_ci95_percent'],
        [32.56318862023946, 38.01716147281345], rtol=0, atol=1e-10)


def test_representative_selection_is_reproducible(reproduced):
    meta, data, _ = reproduced
    maes = data['all_trial_errors'][:, :-1].mean(axis=1)
    chosen, scores = plot.representative_trial(maes)
    assert chosen == meta['selected_trial_index'] == 59
    assert scores[chosen] == scores.min()
    assert meta['selected_trial']['filter_mae_percentile'] == 51.6
    assert meta['selected_trial']['smoother_mae_percentile'] == 51.6


def test_density_snapshot_selection_is_explicit(reproduced):
    meta, data, _ = reproduced
    chosen, ratios = plot.choose_snapshot(data)
    assert chosen == meta['selected_time_step'] == 4
    assert ratios[chosen] == ratios.max()
    assert ratios[chosen] > 0.6


def test_density_normalization_and_terminal_identity(reproduced):
    _, data, _ = reproduced
    for key, grid in [('filtered', 'grid'), ('smoothed', 'grid'),
                      ('dense_filtered', 'dense_grid'), ('dense_smoothed', 'dense_grid')]:
        p = data[key]
        assert np.all(np.isfinite(p)) and np.min(p) >= 0
        np.testing.assert_allclose(p.sum(axis=1) * (2*np.pi/data[grid].size),
                                   1, atol=1e-12)
    np.testing.assert_allclose(data['filtered'][-1], data['smoothed'][-1], atol=1e-12)
    np.testing.assert_allclose(data['backward'][-1], 1, atol=0)


def test_grid_reconstruction_and_mean_directions_are_stable(reproduced):
    meta, data, _ = reproduced
    assert meta['max_abs_figfan_figfdn_difference'] < 1e-11
    for key in ('filtered', 'smoothed'):
        moments = (data[key] * np.exp(1j * data['grid'])).sum(axis=1) * (2*np.pi/data['grid'].size)
        assert np.min(np.abs(moments)) > 1e-6


def test_replay_uses_same_rng_draw_order(reproduced):
    _, data, _ = reproduced
    c = plot.Config()
    rng = np.random.default_rng(c.seed)
    for i in range(60):
        obs = plot.simulate_trial(rng, c)
    np.testing.assert_array_equal(obs['states'], data['states'])
    np.testing.assert_array_equal(obs['measurements'], data['measurements'])
    np.testing.assert_array_equal(obs['shifted_components'], data['shifted_components'])


def test_adjoint_fft_matches_dense_nonsymmetric_transition():
    n = 31
    omega = 2*np.pi/n
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    noise = plot.normalize_grid_density(np.exp(4*np.cos(x - 0.6)), omega)
    message = np.exp(np.cos(x) + 0.3*np.cos(2*x))
    transition = plot.TorusAdditiveGridTransition.for_grid_shape(noise, (n,))
    offsets = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    dense = noise[offsets]
    np.testing.assert_allclose(transition(message, 0), omega * dense.T @ message,
                               rtol=1e-12, atol=1e-12)


def test_wrap_plot_has_no_long_seam_connections():
    times, angles = plot.break_at_wrap(np.arange(4), np.array([6.1, 0.1, 0.3, 6.2]))
    assert np.isnan(times).sum() == np.isnan(angles).sum() == 2
    finite_differences = np.diff(angles)
    assert np.nanmax(np.abs(finite_differences)) <= np.pi


def test_snapshot_detects_mode_across_angular_seam():
    x = np.linspace(0, 2*np.pi, 511, endpoint=False)
    p = np.exp(8*np.cos(x)) + 0.5*np.exp(8*np.cos(x - np.pi))
    ratio = plot.separated_mode_ratio(p, x)
    assert 0.49 < ratio < 0.51


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        plot.Config(grid_size=64).validate()
    with pytest.raises(ValueError):
        plot.Config(n_trials=0).validate()
    with pytest.raises(ValueError):
        plot.Config(outlier_probability=1.5).validate()
    with pytest.raises(ValueError):
        plot.representative_trial(np.zeros((3, 3)))


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError, match="finite n-by-2"):
        plot.representative_trial(np.empty((0, 2)))


def test_invalid_overrides_are_rejected_before_generation(tmp_path):
    with pytest.raises(ValueError, match="trial index"):
        plot.prepare_data(plot.Config(n_trials=2), tmp_path, trial=2)
    with pytest.raises(ValueError, match="terminal"):
        plot.prepare_data(plot.Config(n_trials=2), tmp_path, snapshot=19)
    assert not list(tmp_path.iterdir())


def test_explicit_overrides_are_recorded_even_when_automatic(tmp_path):
    c = plot.Config(n_trials=1, grid_size=63, time_steps=3)
    meta = plot.prepare_data(c, tmp_path, trial=0, snapshot=0)
    assert meta["trial_override"] is True
    assert meta["requested_trial_index"] == 0
    assert meta["snapshot_override"] is True
    assert meta["requested_time_step"] == 0


def test_plotting_exports_all_formats_with_shared_heatmap_scale(reproduced, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import json

    _, _, output = reproduced
    paths = plot.make_figures(output, figures_dir=tmp_path)
    assert len(paths) == 15
    assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"}
    assert all(p.is_file() and p.stat().st_size > 100 for p in paths)
    for path in paths:
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-")
        elif path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert "<svg" in path.read_text(encoding="utf-8")
    meta = json.loads((output / "tracking_example_metadata.json").read_text())
    assert meta["heatmap_scale"]["type"] == "linear"
    assert meta["heatmap_scale"]["per_column_normalization"] is False
