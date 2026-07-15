import numpy as np

from fourier_smoothing import (
    DenseGridTransition,
    additive_noise_density_m_step,
    average_increment_density,
    cell_volume_for_grid,
    grid_backward_information_smoother,
    grid_pairwise_smoothed_marginals,
    normalize_grid_density,
    torus_additive_transition_density_matrix,
    torus_grid,
    torus_increment_densities_from_smoother,
    torus_increment_density_from_messages,
    torus_increment_density_from_pairwise,
)


def _forward_filter(likelihoods, transition, cell_volume):
    prior = normalize_grid_density(np.ones_like(likelihoods[0]), cell_volume)
    filtered = [normalize_grid_density(prior * likelihoods[0], cell_volume)]
    for t in range(1, likelihoods.shape[0]):
        predicted = transition.forward_predict(filtered[-1], t - 1)
        filtered.append(normalize_grid_density(predicted * likelihoods[t], cell_volume))
    return np.stack(filtered, axis=0)


def test_two_dimensional_increment_projection_recovers_additive_noise():
    grid_shape = (3, 4)
    x, y = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    noise = normalize_grid_density(
        np.exp(1.1 * np.cos(x - 0.2) + 0.6 * np.cos(y + 0.4)),
        cell_volume,
    )
    transition = torus_additive_transition_density_matrix(noise, cell_volume=cell_volume)

    uniform_current = np.full(int(np.prod(grid_shape)), 1.0 / (2.0 * np.pi) ** len(grid_shape))
    pairwise = uniform_current[:, None] * transition.T
    np.testing.assert_allclose(np.sum(pairwise) * cell_volume**2, 1.0, rtol=1e-12, atol=1e-12)

    recovered = torus_increment_density_from_pairwise(pairwise, grid_shape, cell_volume)
    np.testing.assert_allclose(recovered, noise, rtol=1e-12, atol=1e-12)


def test_fft_increment_density_matches_dense_pairwise_projection():
    grid_shape = (5, 7)
    x, y = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    filtered = normalize_grid_density(
        1.2 + 0.2 * np.cos(x - 0.3) + 0.1 * np.sin(y + 0.5),
        cell_volume,
    )
    future_information = 1.1 + 0.3 * np.cos(2.0 * x - y + 0.2)
    noise = normalize_grid_density(
        np.exp(1.0 * np.cos(x - 0.1) + 0.7 * np.cos(y + 0.4)),
        cell_volume,
    )
    transition = torus_additive_transition_density_matrix(noise, cell_volume=cell_volume)

    dense_unnormalized = filtered.reshape(-1)[:, None] * transition.T * future_information.reshape(-1)[None, :]
    dense_pairwise = dense_unnormalized / (np.sum(dense_unnormalized) * cell_volume**2)
    expected = torus_increment_density_from_pairwise(dense_pairwise, grid_shape, cell_volume)
    actual = torus_increment_density_from_messages(
        filtered,
        future_information,
        noise,
        cell_volume=cell_volume,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.sum(actual) * cell_volume, 1.0, rtol=1e-12, atol=1e-12)


def test_message_increment_sequence_matches_pairwise_sequence_and_m_step():
    grid_shape = (13,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods = np.stack(
        [
            1.2 + 0.20 * np.cos(x - 0.1),
            1.1 + 0.25 * np.cos(2.0 * x - 0.7),
            1.3 + 0.15 * np.sin(x + 0.4),
            1.2 + 0.10 * np.cos(3.0 * x + 0.2),
        ],
        axis=0,
    )
    noise = normalize_grid_density(np.exp(1.7 * np.cos(x)), cell_volume)
    matrix = torus_additive_transition_density_matrix(noise, cell_volume=cell_volume)
    transition = DenseGridTransition.for_grid_shape(matrix, grid_shape, cell_volume=cell_volume)
    filtered = _forward_filter(likelihoods, transition, cell_volume)
    smoothed = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)
    pairwise = grid_pairwise_smoothed_marginals(
        filtered,
        likelihoods,
        smoothed.backward_messages,
        matrix,
        cell_volume=cell_volume,
    )

    expected = torus_increment_density_from_pairwise(pairwise.joint, grid_shape, cell_volume)
    actual = torus_increment_densities_from_smoother(
        filtered,
        likelihoods,
        smoothed.backward_messages,
        noise,
        cell_volume=cell_volume,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    expected_m_step = average_increment_density(expected, cell_volume)
    actual_m_step = additive_noise_density_m_step(
        filtered,
        likelihoods,
        smoothed.backward_messages,
        noise,
        cell_volume=cell_volume,
    )
    np.testing.assert_allclose(actual_m_step, expected_m_step, rtol=1e-11, atol=1e-12)


def test_average_increment_density_is_normalized():
    grid_shape = (9,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    first = normalize_grid_density(np.exp(1.0 * np.cos(x)), cell_volume)
    second = normalize_grid_density(np.exp(2.0 * np.cos(x - 0.5)), cell_volume)

    averaged = average_increment_density(np.stack([first, second]), cell_volume)
    expected = normalize_grid_density(0.5 * (first + second), cell_volume)
    np.testing.assert_allclose(averaged, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(averaged) * cell_volume, 1.0, rtol=1e-12, atol=1e-12)
