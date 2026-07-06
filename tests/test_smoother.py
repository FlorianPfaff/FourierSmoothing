import numpy as np

from fourier_smoothing import (
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    fourier_identity_smoother,
    fourier_to_grid,
    grid_backward_information_smoother,
    grid_to_fourier,
    normalize_grid_density,
    reverse_frequencies,
    torus_grid,
    torus_identity_backward_predict_fourier,
)


def _normalized_cumulative_likelihoods(likelihoods, cell_volume):
    filtered = []
    cumulative = np.ones_like(likelihoods[0])
    for likelihood in likelihoods:
        cumulative = cumulative * likelihood
        filtered.append(normalize_grid_density(cumulative, cell_volume))
    return np.stack(filtered, axis=0)


def test_grid_smoother_collapses_to_all_likelihoods_for_identity_transition():
    grid_shape = (15,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods = np.stack(
        [
            1.2 + 0.3 * np.cos(x - 0.1),
            1.1 + 0.2 * np.cos(2.0 * x + 0.3),
            1.3 + 0.4 * np.sin(x - 1.0),
            1.4 + 0.1 * np.cos(3.0 * x),
        ],
        axis=0,
    )
    filtered = _normalized_cumulative_likelihoods(likelihoods, cell_volume)

    noise = np.zeros(grid_shape)
    noise[0] = 1.0 / cell_volume
    transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)

    result = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)

    expected = normalize_grid_density(np.prod(likelihoods, axis=0), cell_volume)
    for smoothed_t in result.smoothed:
        np.testing.assert_allclose(smoothed_t, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.sum(smoothed_t) * cell_volume, 1.0, rtol=1e-12, atol=1e-12)


def test_fourier_identity_smoother_matches_grid_identity_case():
    grid_shape = (17,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods_grid = np.stack(
        [
            1.0 + 0.2 * np.cos(x - 0.2),
            1.1 + 0.3 * np.cos(2.0 * x - 0.7),
            1.2 + 0.2 * np.sin(x + 0.4),
        ],
        axis=0,
    )
    filtered_grid = _normalized_cumulative_likelihoods(likelihoods_grid, cell_volume)

    noise_grid = np.zeros(grid_shape)
    noise_grid[0] = 1.0 / cell_volume

    filtered_coeffs = np.stack([grid_to_fourier(values) for values in filtered_grid], axis=0)
    likelihood_coeffs = np.stack([grid_to_fourier(values) for values in likelihoods_grid], axis=0)
    noise_coeffs = grid_to_fourier(noise_grid)

    result = fourier_identity_smoother(filtered_coeffs, likelihood_coeffs, noise_coeffs)
    expected = normalize_grid_density(np.prod(likelihoods_grid, axis=0), cell_volume)

    for coeffs in result.smoothed_coefficients:
        smoothed_grid = fourier_to_grid(coeffs)
        np.testing.assert_allclose(smoothed_grid, expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.sum(smoothed_grid) * cell_volume, 1.0, rtol=1e-12, atol=1e-12)


def test_fourier_backward_prediction_matches_grid_correlation():
    grid_shape = (19,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)

    message = 1.0 + 0.2 * np.cos(x - 0.6) + 0.1 * np.sin(2.0 * x)
    noise = np.exp(2.0 * np.cos(x))
    noise = normalize_grid_density(noise, cell_volume)

    transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
    beta_grid = transition(message, 0)

    beta_coeffs = torus_identity_backward_predict_fourier(grid_to_fourier(message), grid_to_fourier(noise))
    np.testing.assert_allclose(fourier_to_grid(beta_coeffs), beta_grid, rtol=1e-12, atol=1e-12)


def test_reverse_frequencies_negates_frequency_indices():
    coeffs = np.arange(25, dtype=float).reshape(5, 5) + 1j
    reversed_coeffs = reverse_frequencies(coeffs)
    for i in range(5):
        for j in range(5):
            assert reversed_coeffs[i, j] == coeffs[(-i) % 5, (-j) % 5]
