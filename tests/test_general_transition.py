import numpy as np

from fourier_smoothing import (
    cell_volume_for_grid,
    fourier_to_grid,
    grid_to_fourier,
    normalize_grid_density,
    torus_grid,
)
from fourier_smoothing.general_transition import (
    fourier_general_backward_predict,
    fourier_general_forward_predict,
    transition_grid_to_fourier,
)


def _normalized_transition(next_grid, current_grid, next_cell_volume):
    next_angle = next_grid[:, None]
    current_angle = current_grid[None, :]
    center = current_angle + 0.35 * np.sin(current_angle)
    values = (
        np.exp(1.7 * np.cos(next_angle - center))
        + 0.25 * np.exp(0.8 * np.cos(next_angle + 2.0 * current_angle - 0.4))
    )
    return values / (np.sum(values, axis=0, keepdims=True) * next_cell_volume)


def test_general_fourier_backward_prediction_matches_dense_grid_quadrature():
    next_shape = (7,)
    current_shape = (9,)
    (next_grid,) = torus_grid(next_shape)
    (current_grid,) = torus_grid(current_shape)
    next_cell_volume = cell_volume_for_grid(next_shape)

    transition = _normalized_transition(next_grid, current_grid, next_cell_volume)
    future_message = 1.2 + 0.3 * np.cos(2.0 * next_grid - 0.5) + 0.1 * np.sin(3.0 * next_grid)

    transition_coefficients = transition_grid_to_fourier(transition, next_dimension=1)
    message_coefficients = grid_to_fourier(future_message)
    backward_coefficients = fourier_general_backward_predict(
        message_coefficients,
        transition_coefficients,
    )

    expected = next_cell_volume * transition.T @ future_message
    actual = fourier_to_grid(backward_coefficients)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_general_fourier_forward_prediction_matches_dense_grid_quadrature():
    next_shape = (7,)
    current_shape = (9,)
    (next_grid,) = torus_grid(next_shape)
    (current_grid,) = torus_grid(current_shape)
    next_cell_volume = cell_volume_for_grid(next_shape)
    current_cell_volume = cell_volume_for_grid(current_shape)

    transition = _normalized_transition(next_grid, current_grid, next_cell_volume)
    current_density = normalize_grid_density(
        1.1 + 0.25 * np.cos(current_grid - 0.2) + 0.15 * np.sin(2.0 * current_grid),
        current_cell_volume,
    )

    transition_coefficients = transition_grid_to_fourier(transition, next_dimension=1)
    current_coefficients = grid_to_fourier(current_density)
    predicted_coefficients = fourier_general_forward_predict(
        current_coefficients,
        transition_coefficients,
        next_dimension=1,
    )

    expected = current_cell_volume * transition @ current_density
    actual = fourier_to_grid(predicted_coefficients)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.sum(actual) * next_cell_volume, 1.0, rtol=1e-11, atol=1e-12)


def test_general_fourier_forward_and_backward_are_discrete_adjoint_pair():
    next_shape = (8,)
    current_shape = (11,)
    (next_grid,) = torus_grid(next_shape)
    (current_grid,) = torus_grid(current_shape)
    next_cell_volume = cell_volume_for_grid(next_shape)
    current_cell_volume = cell_volume_for_grid(current_shape)

    transition = _normalized_transition(next_grid, current_grid, next_cell_volume)
    transition_coefficients = transition_grid_to_fourier(transition, next_dimension=1)
    current_function = 1.0 + 0.2 * np.cos(3.0 * current_grid + 0.1)
    next_function = 1.1 + 0.3 * np.sin(2.0 * next_grid - 0.4)

    forward = fourier_to_grid(
        fourier_general_forward_predict(
            grid_to_fourier(current_function),
            transition_coefficients,
            next_dimension=1,
        )
    )
    backward = fourier_to_grid(
        fourier_general_backward_predict(
            grid_to_fourier(next_function),
            transition_coefficients,
        )
    )

    left = np.sum(next_function * forward) * next_cell_volume
    right = np.sum(backward * current_function) * current_cell_volume
    np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)
