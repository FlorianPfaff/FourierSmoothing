import csv

import numpy as np

from fourier_smoothing import (
    cell_volume_for_grid,
    filtered_from_likelihoods,
    make_identity_likelihoods,
    make_sharp_multimodal_likelihoods,
    run_identity_torus_benchmark,
    run_truncation_negativity_diagnostic,
    write_benchmark_csv,
    write_negativity_csv,
)


def test_make_identity_likelihoods_shape_and_positivity():
    likelihoods = make_identity_likelihoods((9, 7), 3)
    assert likelihoods.shape == (3, 9, 7)
    assert np.all(likelihoods > 0.0)


def test_make_sharp_multimodal_likelihoods_shape_and_normalization():
    cell_volume = cell_volume_for_grid((33,))
    likelihoods = make_sharp_multimodal_likelihoods((33,), 3, sharpness=7.0)
    assert likelihoods.shape == (3, 33)
    assert np.all(likelihoods > 0.0)
    np.testing.assert_allclose(np.sum(likelihoods, axis=1) * cell_volume, np.ones(3), rtol=1e-12, atol=1e-12)


def test_filtered_from_likelihoods_normalizes_each_step():
    cell_volume = cell_volume_for_grid((11,))
    likelihoods = make_identity_likelihoods((11,), 4)
    filtered = filtered_from_likelihoods(likelihoods, cell_volume)
    integrals = np.sum(filtered, axis=1) * cell_volume
    np.testing.assert_allclose(integrals, np.ones(4), rtol=1e-12, atol=1e-12)


def test_identity_torus_benchmark_writes_csv(tmp_path):
    rows = run_identity_torus_benchmark([9], repetitions=2, time_steps=3, fourier_multiplication="grid")
    assert len(rows) == 4
    assert {row.method for row in rows} == {"grid", "fourier_identity_grid"}
    assert max(row.max_abs_difference_to_grid for row in rows) < 1e-10
    assert max(row.max_normalization_error for row in rows) < 1e-10

    output_path = write_benchmark_csv(rows, tmp_path / "identity_torus_benchmark.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["method"] in {"grid", "fourier_identity_grid"}


def test_identity_torus_benchmark_default_uses_truncated_convolution():
    rows = run_identity_torus_benchmark([9], repetitions=1, time_steps=3)
    assert {row.method for row in rows} == {"grid", "fourier_identity_truncated_convolution"}
    assert max(row.max_normalization_error for row in rows) < 1e-10


def test_truncation_negativity_diagnostic_writes_csv(tmp_path):
    rows = run_truncation_negativity_diagnostic(
        [1, 3],
        sharpness_values=[9.0],
        evaluation_grid_size=65,
        time_steps=3,
    )
    assert len(rows) == 2
    assert all(row.negative_mass >= 0.0 for row in rows)
    assert all(row.max_negative_undershoot >= 0.0 for row in rows)
    assert all(row.l1_error_to_dense_grid >= 0.0 for row in rows)
    assert all(np.isfinite(row.min_value) for row in rows)
    assert all(np.isfinite(row.negative_mass) for row in rows)
    assert all(np.isfinite(row.max_negative_undershoot) for row in rows)
    assert all(np.isfinite(row.l1_error_to_dense_grid) for row in rows)
    assert all(row.max_normalization_error < 1e-8 for row in rows)

    output_path = write_negativity_csv(rows, tmp_path / "truncation_negativity_diagnostic.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["k_max"] in {"1", "3"}
