import csv

import numpy as np

from fourier_smoothing import (
    cell_volume_for_grid,
    filtered_from_likelihoods,
    make_identity_likelihoods,
    make_pwc_additive_transition_density_matrix_1d,
    make_pwc_additive_transition_kernel_1d,
    make_sharp_multimodal_likelihoods,
    run_identity_torus_benchmark,
    run_figf_pwc_benchmark,
    run_smoothing_gain_evaluation,
    run_smoothing_evaluation,
    run_smoothing_runtime_evaluation,
    run_truncation_negativity_diagnostic,
    write_benchmark_csv,
    write_figf_pwc_csv,
    write_negativity_csv,
    write_smoothing_gain_csv,
    write_smoothing_evaluation_csv,
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


def test_figf_pwc_benchmark_writes_csv(tmp_path):
    rows = run_figf_pwc_benchmark(
        [9],
        repetitions=1,
        time_steps=3,
        reference_grid_size=65,
        pwc_quadrature_points=3,
    )
    assert len(rows) == 3
    assert {row.method for row in rows} == {"FIGFAN", "FIGFDN", "PWC"}
    assert all(row.mean_l1_error_to_reference >= 0.0 for row in rows)
    assert all(row.max_l1_error_to_reference >= 0.0 for row in rows)
    assert all(np.isfinite(row.min_evaluated_density) for row in rows)
    assert all(row.max_normalization_error < 1e-8 for row in rows)

    output_path = write_figf_pwc_csv(rows, tmp_path / "figf_pwc_benchmark.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["method"] in {"FIGFAN", "FIGFDN", "PWC"}


def test_pwc_transition_matrix_columns_are_normalized():
    grid_size = 11
    cell_volume = cell_volume_for_grid((grid_size,))
    transition = make_pwc_additive_transition_density_matrix_1d(grid_size, 3.0, quadrature_points=3)
    np.testing.assert_allclose(np.sum(transition, axis=0) * cell_volume, np.ones(grid_size), rtol=1e-12, atol=1e-12)


def test_pwc_transition_kernel_is_normalized():
    grid_size = 11
    cell_volume = cell_volume_for_grid((grid_size,))
    kernel = make_pwc_additive_transition_kernel_1d(grid_size, 3.0, quadrature_points=3)
    np.testing.assert_allclose(np.sum(kernel) * cell_volume, 1.0, rtol=1e-12, atol=1e-12)


def test_smoothing_evaluation_writes_csv(tmp_path):
    rows = run_smoothing_evaluation(
        figf_grid_sizes=[9],
        pwc_grid_sizes=[9],
        pf_particle_counts=[30],
        repetitions=1,
        time_steps=3,
        l1_reference_grid_size=65,
        mean_reference_particles=200,
        mean_reference_repetitions=1,
        pwc_quadrature_points=3,
        seed=7,
    )
    assert len(rows) == 4
    assert {row.method for row in rows} == {"FIGFAN", "FIGFDN", "PWC", "PF"}
    assert all(row.runtime_s >= 0.0 for row in rows)
    assert all(row.mean_error_rad >= 0.0 for row in rows)
    assert all(row.l1_error >= 0.0 for row in rows)
    assert all(row.max_normalization_error < 1e-8 for row in rows)

    output_path = write_smoothing_evaluation_csv(rows, tmp_path / "smoothing_evaluation_raw.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["method"] in {"FIGFAN", "FIGFDN", "PWC", "PF"}


def test_smoothing_runtime_evaluation_matches_error_row_keys():
    rows = run_smoothing_runtime_evaluation(
        figf_grid_sizes=[9],
        pwc_grid_sizes=[9],
        pf_particle_counts=[30],
        repetitions=2,
        time_steps=3,
        particle_likelihood_grid_size=65,
        pwc_quadrature_points=3,
        seed=7,
    )

    assert len(rows) == 8
    assert {row.method for row in rows} == {"FIGFAN", "FIGFDN", "PWC", "PF"}
    assert all(row.runtime_s >= 0.0 for row in rows)
    assert {(row.method, row.parameter, row.repetition) for row in rows} == {
        (method, parameter, repetition)
        for method, parameter in (("FIGFAN", 9), ("FIGFDN", 9), ("PWC", 9), ("PF", 30))
        for repetition in range(2)
    }
    for repetition in range(2):
        figf_rows = [row for row in rows if row.repetition == repetition and row.method.startswith("FIGF")]
        assert figf_rows[0].runtime_s == figf_rows[1].runtime_s


def test_smoothing_gain_evaluation_writes_csv(tmp_path):
    rows = run_smoothing_gain_evaluation(
        n_trials=2,
        grid_size=33,
        time_steps=3,
        seed=7,
    )
    assert len(rows) == 6
    assert {row.trial for row in rows} == {0, 1}
    assert {row.time_step for row in rows} == {0, 1, 2}
    assert all(np.isfinite(row.filter_error_rad) for row in rows)
    assert all(np.isfinite(row.smoother_error_rad) for row in rows)
    assert all(row.filter_error_rad >= 0.0 for row in rows)
    assert all(row.smoother_error_rad >= 0.0 for row in rows)
    final_rows = [row for row in rows if row.time_step == 2]
    np.testing.assert_allclose(
        [row.filter_error_rad for row in final_rows],
        [row.smoother_error_rad for row in final_rows],
    )

    output_path = write_smoothing_gain_csv(rows, tmp_path / "smoothing_gain_raw.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["trial"] == "0"


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
