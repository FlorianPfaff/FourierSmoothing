import csv

import numpy as np

from fourier_smoothing import circular_abs_difference, run_particle_baseline_benchmark, write_particle_baseline_csv


def test_circular_abs_difference_wraps_correctly():
    left = np.array([0.0, 2.0 * np.pi - 0.1])
    right = np.array([0.1, 0.0])
    np.testing.assert_allclose(circular_abs_difference(left, right), np.array([0.1, 0.1]), atol=1e-12)


def test_particle_baseline_benchmark_writes_csv(tmp_path):
    rows = run_particle_baseline_benchmark(
        [50],
        n_trajectories=20,
        repetitions=2,
        grid_size=41,
        time_steps=3,
        seed=7,
    )
    assert len(rows) == 2
    assert all(row.n_particles == 50 for row in rows)
    assert all(row.n_trajectories == 20 for row in rows)
    assert all(row.runtime_s >= 0.0 for row in rows)
    assert all(row.mean_abs_circular_error_to_grid >= 0.0 for row in rows)
    assert all(row.max_abs_circular_error_to_grid >= 0.0 for row in rows)

    output_path = write_particle_baseline_csv(rows, tmp_path / "particle_smoother_baseline.csv")
    with output_path.open(newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == len(rows)
    assert loaded_rows[0]["n_particles"] == "50"
