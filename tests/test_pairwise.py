import numpy as np

from fourier_smoothing import (
    DenseGridTransition,
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    grid_backward_information_smoother,
    normalize_grid_density,
    torus_grid,
)
from fourier_smoothing.pairwise import (
    grid_pairwise_smoothed_marginals,
    pairwise_current_marginal,
    pairwise_next_marginal,
    torus_additive_transition_density_matrix,
)


def _forward_filter(likelihoods, transition, cell_volume):
    prior = normalize_grid_density(np.ones_like(likelihoods[0]), cell_volume)
    filtered = [normalize_grid_density(prior * likelihoods[0], cell_volume)]
    for t in range(1, likelihoods.shape[0]):
        predicted = transition.forward_predict(filtered[-1], t - 1)
        filtered.append(normalize_grid_density(predicted * likelihoods[t], cell_volume))
    return np.stack(filtered, axis=0)


def test_additive_transition_matrix_matches_fft_backward_prediction():
    grid_shape = (17,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    noise = normalize_grid_density(np.exp(2.0 * np.cos(x)), cell_volume)
    message = 1.1 + 0.2 * np.cos(2.0 * x - 0.4)

    matrix = torus_additive_transition_density_matrix(noise, cell_volume=cell_volume)
    dense_backward = cell_volume * matrix.T @ message
    fft_backward = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)(message, 0)

    np.testing.assert_allclose(dense_backward, fft_backward, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(matrix, axis=0) * cell_volume, np.ones(grid_shape[0]), rtol=1e-12, atol=1e-12)


def test_pairwise_marginals_recover_one_time_smoothed_densities():
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
    smoothed = grid_backward_information_smoother(
        filtered,
        likelihoods,
        transition,
        cell_volume=cell_volume,
    )
    pairwise = grid_pairwise_smoothed_marginals(
        filtered,
        likelihoods,
        smoothed.backward_messages,
        matrix,
        cell_volume=cell_volume,
    )

    assert pairwise.joint.shape == (likelihoods.shape[0] - 1, grid_shape[0], grid_shape[0])
    for t, joint_t in enumerate(pairwise.joint):
        np.testing.assert_allclose(
            np.sum(joint_t) * cell_volume**2,
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            pairwise_current_marginal(joint_t, cell_volume),
            smoothed.smoothed[t].reshape(-1),
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            pairwise_next_marginal(joint_t, cell_volume),
            smoothed.smoothed[t + 1].reshape(-1),
            rtol=1e-11,
            atol=1e-12,
        )


def test_pairwise_supports_time_varying_transition_matrices():
    grid_shape = (7,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods = np.ones((3, grid_shape[0]))

    noise_a = normalize_grid_density(np.exp(1.0 * np.cos(x)), cell_volume)
    noise_b = normalize_grid_density(np.exp(2.0 * np.cos(x - 0.3)), cell_volume)
    matrices = np.stack(
        [
            torus_additive_transition_density_matrix(noise_a, cell_volume=cell_volume),
            torus_additive_transition_density_matrix(noise_b, cell_volume=cell_volume),
        ],
        axis=0,
    )
    transition = DenseGridTransition.for_grid_shape(matrices, grid_shape, cell_volume=cell_volume)
    filtered = _forward_filter(likelihoods, transition, cell_volume)
    smoothed = grid_backward_information_smoother(filtered, likelihoods, transition, cell_volume=cell_volume)

    result = grid_pairwise_smoothed_marginals(
        filtered,
        likelihoods,
        smoothed.backward_messages,
        matrices,
        cell_volume=cell_volume,
    )
    assert result.joint.shape == (2, 7, 7)
    assert np.all(np.isfinite(result.joint))
