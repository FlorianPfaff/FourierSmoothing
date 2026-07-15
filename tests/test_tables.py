import csv

from fourier_smoothing.tables import TABLE_FILENAMES, write_latex_tables


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_write_latex_tables_from_known_result_csvs(tmp_path):
    results_dir = tmp_path / "results"
    tables_dir = tmp_path / "tables"
    results_dir.mkdir()

    _write_csv(
        results_dir / "identity_torus_benchmark.csv",
        ["method", "grid_size", "repetition", "runtime_s", "max_abs_difference_to_grid", "max_normalization_error"],
        [
            {
                "method": "grid",
                "grid_size": "9",
                "repetition": "0",
                "runtime_s": "0.01",
                "max_abs_difference_to_grid": "0.0",
                "max_normalization_error": "0.0",
            },
            {
                "method": "fourier_identity_truncated_convolution",
                "grid_size": "9",
                "repetition": "0",
                "runtime_s": "0.02",
                "max_abs_difference_to_grid": "1e-5",
                "max_normalization_error": "1e-14",
            },
        ],
    )
    _write_csv(
        results_dir / "truncation_negativity_diagnostic.csv",
        [
            "k_max",
            "n_coefficients",
            "sharpness",
            "time_steps",
            "min_value",
            "negative_mass",
            "max_negative_undershoot",
            "max_normalization_error",
            "l1_error_to_dense_grid",
        ],
        [
            {
                "k_max": "1",
                "n_coefficients": "3",
                "sharpness": "9.0",
                "time_steps": "3",
                "min_value": "-0.01",
                "negative_mass": "0.02",
                "max_negative_undershoot": "0.01",
                "max_normalization_error": "0.0",
                "l1_error_to_dense_grid": "0.3",
            }
        ],
    )
    _write_csv(
        results_dir / "particle_smoother_baseline.csv",
        [
            "n_particles",
            "n_trajectories",
            "repetition",
            "runtime_s",
            "mean_abs_circular_error_to_grid",
            "max_abs_circular_error_to_grid",
        ],
        [
            {
                "n_particles": "50",
                "n_trajectories": "20",
                "repetition": "0",
                "runtime_s": "0.03",
                "mean_abs_circular_error_to_grid": "0.1",
                "max_abs_circular_error_to_grid": "0.2",
            }
        ],
    )
    _write_csv(
        results_dir / "smoothing_evaluation_summary.csv",
        ["method", "parameter", "n_repetitions", "runtime_s_mean", "mean_error_rad_mean", "l1_error_mean"],
        [
            {
                "method": "FIGFAN",
                "parameter": "17",
                "n_repetitions": "1",
                "runtime_s_mean": "0.004",
                "mean_error_rad_mean": "0.02",
                "l1_error_mean": "0.05",
            }
        ],
    )
    # If both are present, the pre-aggregated summary must take precedence.
    _write_csv(
        results_dir / "smoothing_evaluation_raw.csv",
        ["method", "parameter", "repetition", "runtime_s", "mean_error_rad", "l1_error"],
        [
            {
                "method": "FIGFAN",
                "parameter": "17",
                "repetition": "0",
                "runtime_s": "9.9",
                "mean_error_rad": "9.9",
                "l1_error": "9.9",
            }
        ],
    )

    written = write_latex_tables(results_dir, tables_dir)

    assert len(written) == 4
    for filename in [
        TABLE_FILENAMES["identity"],
        TABLE_FILENAMES["negativity"],
        TABLE_FILENAMES["particle"],
        TABLE_FILENAMES["smoothing"],
    ]:
        content = (tables_dir / filename).read_text(encoding="utf-8")
        assert "\\begin{tabular}" in content
        assert "\\toprule" in content
        assert "\\bottomrule" in content
    assert "fourier\\_identity\\_truncated\\_convolution" in (tables_dir / TABLE_FILENAMES["identity"]).read_text(
        encoding="utf-8"
    )
    smoothing_content = (tables_dir / TABLE_FILENAMES["smoothing"]).read_text(encoding="utf-8")
    assert "0.004" in smoothing_content
    assert "9.9" not in smoothing_content
