import numpy as np

from fourier_smoothing import (
    cell_volume_for_grid,
    fourier_general_backward_predict,
    fourier_general_forward_predict,
    fourier_to_grid,
    grid_to_fourier,
    normalize_grid_density,
    torus_grid,
    transition_grid_to_fourier,
)


def test_two_dimensional_general_transition_matches_dense_quadrature():
    next_shape = (3, 4)
    current_shape = (4, 3)
    next_y1, next_y2 = torus_grid(next_shape)
    current_x1, current_x2 = torus_grid(current_shape)
    next_cell_volume = cell_volume_for_grid(next_shape)
    current_cell_volume = cell_volume_for_grid(current_shape)

    y1 = next_y1[:, :, None, None]
    y2 = next_y2[:, :, None, None]
    x1 = current_x1[None, None, :, :]
    x2 = current_x2[None, None, :, :]
    center1 = x1 + 0.25 * np.sin(x2)
    center2 = x2 + 0.20 * np.sin(x1)
    transition = (
        np.exp(1.2 * np.cos(y1 - center1) + 0.8 * np.cos(y2 - center2))
        + 0.15 * np.exp(0.6 * np.cos(y1 + y2 - x1 + 0.3))
    )
    transition /= np.sum(transition, axis=(0, 1), keepdims=True) * next_cell_volume

    transition_coefficients = transition_grid_to_fourier(transition, next_dimension=2)
    future_message = 1.2 + 0.2 * np.cos(next_y1 - 0.3) + 0.15 * np.sin(2.0 * next_y2 + 0.1)
    backward = fourier_to_grid(
        fourier_general_backward_predict(
            grid_to_fourier(future_message),
            transition_coefficients,
        )
    )
    transition_matrix = transition.reshape(np.prod(next_shape), np.prod(current_shape))
    expected_backward = (
        next_cell_volume * transition_matrix.T @ future_message.reshape(-1)
    ).reshape(current_shape)
    np.testing.assert_allclose(backward, expected_backward, rtol=1e-11, atol=1e-12)

    current_density = normalize_grid_density(
        1.1 + 0.2 * np.cos(current_x1 - 0.4) + 0.1 * np.sin(current_x2 + 0.2),
        current_cell_volume,
    )
    predicted = fourier_to_grid(
        fourier_general_forward_predict(
            grid_to_fourier(current_density),
            transition_coefficients,
            next_dimension=2,
        )
    )
    expected_predicted = (
        current_cell_volume * transition_matrix @ current_density.reshape(-1)
    ).reshape(next_shape)
    np.testing.assert_allclose(predicted, expected_predicted, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(predicted) * next_cell_volume,
        1.0,
        rtol=1e-11,
        atol=1e-12,
    )
