import numpy as np

from fourier_smoothing import (
    average_increment_density,
    cell_volume_for_grid,
    normalize_grid_density,
    torus_additive_transition_density_matrix,
    torus_grid,
    torus_increment_density_from_pairwise,
)


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
