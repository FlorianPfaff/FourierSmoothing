import numpy as np

from fourier_smoothing import make_identity_likelihoods, make_von_mises_like_noise
from fourier_smoothing.particle import (
    bootstrap_particle_filter_1d,
    bootstrap_von_mises_particle_filter_1d,
    circular_mean,
    ffbsi_particle_smoother_1d,
    ffbsi_von_mises_particle_smoother_1d,
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


def test_von_mises_particle_filter_and_rejection_ffbsi_are_reproducible():
    grid_size = 65
    angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    likelihoods = np.stack(
        [np.exp(3.0 * np.cos(angles - phase)) for phase in (0.2, 0.5, 0.9)],
        axis=0,
    )

    first_filter = bootstrap_von_mises_particle_filter_1d(likelihoods, 4.0, 200, rng=5)
    second_filter = bootstrap_von_mises_particle_filter_1d(likelihoods, 4.0, 200, rng=5)
    np.testing.assert_allclose(first_filter.particles, second_filter.particles)
    np.testing.assert_allclose(first_filter.weights, second_filter.weights)

    first = ffbsi_von_mises_particle_smoother_1d(first_filter, 4.0, 150, rng=7)
    second = ffbsi_von_mises_particle_smoother_1d(second_filter, 4.0, 150, rng=7)
    assert first.trajectories.shape == (150, 3)
    np.testing.assert_allclose(first.trajectories, second.trajectories)
    np.testing.assert_allclose(first.mean_directions, second.mean_directions)
    assert np.all((first.trajectories >= 0.0) & (first.trajectories < 2.0 * np.pi))


def test_circular_mean_for_nearby_samples_is_finite():
    values = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.25]])
    mean = circular_mean(values, axis=0)
    assert mean.shape == (2,)
    assert np.all(np.isfinite(mean))
