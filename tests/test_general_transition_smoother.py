import numpy as np

from fourier_smoothing import (
    DenseGridTransition,
    cell_volume_for_grid,
    fourier_general_transition_smoother,
    fourier_to_grid,
    grid_backward_information_smoother,
    grid_to_fourier,
    normalize_grid_density,
    torus_grid,
    transition_grid_to_fourier,
)


def _transition_matrix(grid, cell_volume, phase_shift):
    next_angle = grid[:, None]
    current_angle = grid[None, :]
    center = current_angle + 0.3 * np.sin(current_angle + phase_shift)
    values = (
        np.exp(1.5 * np.cos(next_angle - center - phase_shift))
        + 0.20 * np.exp(0.7 * np.cos(2.0 * next_angle + current_angle + 0.4))
    )
    return values / (np.sum(values, axis=0, keepdims=True) * cell_volume)


def _forward_filter(likelihoods, transition, cell_volume):
    prior = normalize_grid_density(np.ones_like(likelihoods[0]), cell_volume)
    filtered = [normalize_grid_density(prior * likelihoods[0], cell_volume)]
    for t in range(1, likelihoods.shape[0]):
        predicted = transition.forward_predict(filtered[-1], t - 1)
        filtered.append(normalize_grid_density(predicted * likelihoods[t], cell_volume))
    return np.stack(filtered, axis=0)


def test_general_transition_fourier_smoother_matches_grid_smoother():
    grid_shape = (13,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    likelihoods = np.stack(
        [
            1.15 + 0.20 * np.cos(x - 0.1),
            1.20 + 0.25 * np.cos(2.0 * x - 0.6),
            1.10 + 0.15 * np.sin(3.0 * x + 0.3),
            1.25 + 0.18 * np.cos(x + 1.1),
        ],
        axis=0,
    )
    transition_matrices = np.stack(
        [
            _transition_matrix(x, cell_volume, 0.0),
            _transition_matrix(x, cell_volume, 0.25),
            _transition_matrix(x, cell_volume, -0.20),
        ],
        axis=0,
    )
    grid_transition = DenseGridTransition.for_grid_shape(
        transition_matrices,
        grid_shape,
        cell_volume=cell_volume,
    )
    filtered = _forward_filter(likelihoods, grid_transition, cell_volume)
    grid_result = grid_backward_information_smoother(
        filtered,
        likelihoods,
        grid_transition,
        cell_volume=cell_volume,
    )

    filtered_coefficients = np.stack([grid_to_fourier(values) for values in filtered])
    likelihood_coefficients = np.stack([grid_to_fourier(values) for values in likelihoods])
    transition_coefficients = np.stack(
        [transition_grid_to_fourier(values, next_dimension=1) for values in transition_matrices]
    )
    fourier_result = fourier_general_transition_smoother(
        filtered_coefficients,
        likelihood_coefficients,
        transition_coefficients,
        multiplication="grid",
    )

    for coefficients, expected in zip(fourier_result.smoothed_coefficients, grid_result.smoothed):
        np.testing.assert_allclose(
            fourier_to_grid(coefficients),
            expected,
            rtol=1e-10,
            atol=1e-11,
        )


def test_general_transition_fourier_smoother_single_step_needs_no_transition():
    grid_shape = (9,)
    (x,) = torus_grid(grid_shape)
    cell_volume = cell_volume_for_grid(grid_shape)
    filtered = normalize_grid_density(1.1 + 0.2 * np.cos(x - 0.4), cell_volume)
    likelihood = 1.2 + 0.1 * np.sin(x)

    result = fourier_general_transition_smoother(
        np.stack([grid_to_fourier(filtered)]),
        np.stack([grid_to_fourier(likelihood)]),
        None,
    )
    np.testing.assert_allclose(
        fourier_to_grid(result.smoothed_coefficients[0]),
        filtered,
        rtol=1e-12,
        atol=1e-12,
    )
