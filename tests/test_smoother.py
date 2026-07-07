import numpy as np

from fourier_smoothing import (
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    fourier_identity_smoother,
    fourier_to_grid,
    grid_backward_information_smoother,
    grid_to_fourier,
    multiply_fourier_truncated,
    multiply_fourier_via_grid,
    normalize_grid_density,
    resize_fourier_coefficients,
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


def _brute_force_additive_backward_predict(noise, message, cell_volume):
    beta = np.zeros_like(message, dtype=float)
    shape = noise.shape
    for current_index in np.ndindex(shape):
        total = 0.0
        for next_index in np.ndindex(shape):
            offset_index = tuple((next_index[axis] - current_index[axis]) % shape[axis] for axis in range(len(shape)))
            total += noise[offset_index] * message[next_index]
        beta[current_index] = cell_volume * total
    return beta


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


def test_fourier_smoother_matches_grid_smoother_with_diffusive_noise_in_grid_mode():
    grid_shape = (11, 13)
    x, y = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods_grid = np.stack(
        [
            1.2 + 0.15 * np.cos(x - 0.1) + 0.10 * np.sin(y + 0.2),
            1.1 + 0.20 * np.cos(2.0 * x - 0.4) + 0.12 * np.cos(y - 0.8),
            1.3 + 0.08 * np.sin(x + y),
        ],
        axis=0,
    )
    filtered_grid = _normalized_cumulative_likelihoods(likelihoods_grid, cell_volume)
    noise_grid = np.exp(1.5 * np.cos(x) + 0.8 * np.cos(y))
    noise_grid = normalize_grid_density(noise_grid, cell_volume)

    transition = TorusAdditiveGridTransition.for_grid_shape(noise_grid, grid_shape)
    grid_result = grid_backward_information_smoother(filtered_grid, likelihoods_grid, transition, cell_volume=cell_volume)

    filtered_coeffs = np.stack([grid_to_fourier(values) for values in filtered_grid], axis=0)
    likelihood_coeffs = np.stack([grid_to_fourier(values) for values in likelihoods_grid], axis=0)
    noise_coeffs = grid_to_fourier(noise_grid)
    fourier_result = fourier_identity_smoother(
        filtered_coeffs,
        likelihood_coeffs,
        noise_coeffs,
        multiplication="grid",
    )

    for coeffs, smoothed_grid in zip(fourier_result.smoothed_coefficients, grid_result.smoothed):
        np.testing.assert_allclose(fourier_to_grid(coeffs), smoothed_grid, rtol=1e-10, atol=1e-11)


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


def test_two_dimensional_grid_backward_prediction_matches_brute_force_reference():
    grid_shape = (5, 7)
    x, y = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)

    message = 1.0 + 0.1 * np.cos(x - 0.4) + 0.2 * np.sin(y + 0.7)
    noise = np.exp(1.1 * np.cos(x) + 0.7 * np.cos(y))
    noise = normalize_grid_density(noise, cell_volume)

    transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
    np.testing.assert_allclose(
        transition(message, 0),
        _brute_force_additive_backward_predict(noise, message, cell_volume),
        rtol=1e-12,
        atol=1e-12,
    )


def test_multiply_fourier_truncated_avoids_same_grid_aliasing():
    coeffs = np.zeros((5,), dtype=complex)
    coeffs[2] = 1.0  # highest represented positive frequency k=2

    truncated_product = multiply_fourier_truncated(coeffs, coeffs)
    grid_product = multiply_fourier_via_grid(coeffs, coeffs)

    assert np.max(np.abs(truncated_product)) < 1e-12
    assert np.max(np.abs(grid_product)) >= 0.49


def test_multiply_fourier_truncated_matches_grid_product_when_band_limited_product_fits():
    grid_shape = (7,)
    (x,) = torus_grid(grid_shape)
    f = 1.0 + 0.2 * np.cos(x - 0.3)
    g = 1.0 + 0.1 * np.sin(x + 0.4)

    product_coeffs = multiply_fourier_truncated(grid_to_fourier(f), grid_to_fourier(g))
    np.testing.assert_allclose(fourier_to_grid(product_coeffs), f * g, rtol=1e-12, atol=1e-12)


def test_fourier_coefficients_can_be_resized_for_dense_evaluation():
    coarse_shape = (5,)
    dense_shape = (31,)
    (x_coarse,) = torus_grid(coarse_shape)
    (x_dense,) = torus_grid(dense_shape)
    values = 1.0 + 0.2 * np.cos(2.0 * x_coarse - 0.3)
    expected_dense = 1.0 + 0.2 * np.cos(2.0 * x_dense - 0.3)

    coeffs = grid_to_fourier(values)
    dense_values = fourier_to_grid(coeffs, grid_shape=dense_shape)
    resized = resize_fourier_coefficients(coeffs, dense_shape)

    np.testing.assert_allclose(dense_values, expected_dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fourier_to_grid(resized), expected_dense, rtol=1e-12, atol=1e-12)


def test_reverse_frequencies_negates_frequency_indices():
    coeffs = np.arange(25, dtype=float).reshape(5, 5) + 1j
    reversed_coeffs = reverse_frequencies(coeffs)
    for i in range(5):
        for j in range(5):
            assert reversed_coeffs[i, j] == coeffs[(-i) % 5, (-j) % 5]
