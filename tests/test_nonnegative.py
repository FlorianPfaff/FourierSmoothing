import numpy as np
import pytest

from fourier_smoothing import (
    DenseGridTransition,
    cell_volume_for_grid,
    clip_roundoff_nonnegative,
    grid_backward_information_smoother,
    normalize_grid_density,
)


def test_roundoff_scale_negative_values_are_clipped():
    values = clip_roundoff_nonnegative([1.0, -2.0e-14], "test values")
    np.testing.assert_array_equal(values, [1.0, 0.0])


def test_material_negative_values_raise_in_density_normalization():
    with pytest.raises(ValueError, match="materially negative"):
        normalize_grid_density([1.0, -1.0e-4], 1.0)


def test_dense_transition_rejects_materially_negative_entries():
    grid_shape = (2,)
    cell_volume = cell_volume_for_grid(grid_shape)
    matrix = np.array([[1.0, -1.0e-3], [0.0, 1.0]])
    with pytest.raises(ValueError, match="materially negative"):
        DenseGridTransition.for_grid_shape(matrix, grid_shape, cell_volume=cell_volume)


def test_smoother_rejects_materially_negative_backward_prediction():
    filtered = np.full((2, 2), 1.0 / (2.0 * np.pi))
    likelihoods = np.ones((2, 2))

    def invalid_backward(_message, _time):
        return np.array([1.0, -1.0e-3])

    with pytest.raises(ValueError, match="materially negative"):
        grid_backward_information_smoother(filtered, likelihoods, invalid_backward)


def test_smoother_accepts_roundoff_negative_backward_prediction():
    filtered = np.full((2, 2), 1.0 / (2.0 * np.pi))
    likelihoods = np.ones((2, 2))

    def roundoff_backward(_message, _time):
        return np.array([1.0, -1.0e-14])

    result = grid_backward_information_smoother(filtered, likelihoods, roundoff_backward)
    assert np.min(result.backward_messages) == 0.0
    assert np.min(result.smoothed) >= 0.0
