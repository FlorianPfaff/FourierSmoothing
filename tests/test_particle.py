import numpy as np

from fourier_smoothing import make_identity_likelihoods, make_von_mises_like_noise
from fourier_smoothing.particle import (
    bootstrap_particle_filter_1d,
    circular_mean,
    ffbsi_particle_smoother_1d,
    periodic_linear_interpolate_1d,
    systematic_resample,
)


def test_periodic_linear_interpolate_matches_grid_points():
    values = np.array([0.0, 1.0, 0.0, -1.0])
    xs = 2.0 * np.pi * np.arange(4) / 4
    np.testing.assert_allclose(periodic_linear_interpolate_1d(values, xs), values)


def test_systematic_resample_returns_valid_indices():
    rng = np.random.default_rng(1)
    indices = systematic_resample(np.array([0.1, 0.2, 0.7]), rng)
    assert indices.shape == (3,)
    assert np.all(indices >= 0)
    assert np.all(indices < 3)


def test_bootstrap_particle_filter_normalizes_weights():
    rng = np.random.default_rng(2)
    likelihoods = make_identity_likelihoods((31,), 4)
    noise = make_von_mises_like_noise((31,), 3.0)
    result = bootstrap_particle_filter_1d(likelihoods, noise, 200, rng=rng)

    assert result.particles.shape == (4, 200)
    assert result.weights.shape == (4, 200)
    np.testing.assert_allclose(np.sum(result.weights, axis=1), np.ones(4), rtol=1e-12, atol=1e-12)


def test_ffbsi_particle_smoother_shapes_and_ranges():
    rng = np.random.default_rng(3)
    likelihoods = make_identity_likelihoods((31,), 4)
    noise = make_von_mises_like_noise((31,), 3.0)
    filtered = bootstrap_particle_filter_1d(likelihoods, noise, 250, rng=rng)
    smoothed = ffbsi_particle_smoother_1d(filtered, noise, 40, rng=rng)

    assert smoothed.trajectories.shape == (40, 4)
    assert smoothed.mean_directions.shape == (4,)
    assert np.all(smoothed.trajectories >= 0.0)
    assert np.all(smoothed.trajectories < 2.0 * np.pi)
    assert np.all(smoothed.mean_directions >= 0.0)
    assert np.all(smoothed.mean_directions < 2.0 * np.pi)


def test_circular_mean_for_nearby_samples_is_finite():
    values = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.25]])
    mean = circular_mean(values, axis=0)
    assert mean.shape == (2,)
    assert np.all(np.isfinite(mean))
