import csv

import numpy as np

from fourier_smoothing import (
    cell_volume_for_grid,
    filtered_from_likelihoods,
    make_identity_likelihoods,
    run_identity_torus_benchmark,
    write_benchmark_csv,
)


def test_make_identity_likelihoods_shape_and_positivity():
    likelihoods = make_identity_likelihoods((9, 7), 3)
    assert likelihoods.shape == (3, 9, 7)
    assert np.all(likelihoods > 0.0)


def test_filtered_from_likelihoods_normalizes_each_step():
    cell_volume = cell_volume_for_grid((11,))
    likelihoods = make_identity_likelihoods((11,), 4)
    filtered = filtered_from_likelihoods(likelihoods, cell_volume)
    integrals = np.sum(filtered, axis=1) * cell_volume
    np.testing.assert_allclose(integrals, np.ones(4), rtol=1e-12, atol=1e-12)


def test_identity_torus_benchmark_writes_csv(tmp_path):
    rows = run_identity_torus_benchmark([9], repetitions=2, time_steps=3)
    assert len(rows) == 4
    assert {row.method for row in rows} == {"grid", "fourier_identity"}
    assert max(row.max_abs_difference_to_grid for row in rows) < 1e-10
    assert max(row.max_normalization_error for row in rows) < 1e-10

    output_path = write_benchmark_csv(rows, tmp_path / "identity_torus_benchmark.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["method"] in {"grid", "fourier_identity"}
