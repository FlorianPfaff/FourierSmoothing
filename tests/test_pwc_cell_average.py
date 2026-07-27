import numpy as np
import pytest

import fourier_smoothing.experiments as historical_experiments
from fourier_smoothing import (
    make_pwc_cell_averaged_likelihoods_1d,
    make_sharp_multimodal_likelihoods,
    run_figf_pwc_benchmark,
    run_smoothing_evaluation,
    run_smoothing_runtime_evaluation,
)


def test_pwc_likelihood_projection_is_normalized_and_converged():
    projected = make_pwc_cell_averaged_likelihoods_1d(
        (15,),
        3,
        sharpness=5.0,
        quadrature_points=8,
    )
    reference = make_pwc_cell_averaged_likelihoods_1d(
        (15,),
        3,
        sharpness=5.0,
        quadrature_points=4096,
    )

    cell_width = 2.0 * np.pi / 15
    np.testing.assert_allclose(np.sum(projected, axis=1) * cell_width, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(projected, reference, rtol=6.0e-4, atol=2.0e-6)


def test_pwc_cell_average_is_not_midpoint_collocation_on_coarse_grid():
    projected = make_pwc_cell_averaged_likelihoods_1d(
        (15,),
        1,
        sharpness=5.0,
    )
    midpoint = make_sharp_multimodal_likelihoods(
        (15,),
        1,
        sharpness=5.0,
        grid_offset=0.5,
    )

    assert np.max(np.abs(projected - midpoint)) > 1.0e-2


def test_paper_evaluation_responds_to_pwc_cell_quadrature():
    common = {
        "figf_grid_sizes": (9,),
        "pwc_grid_sizes": (9,),
        "pf_particle_counts": (20,),
        "repetitions": 1,
        "time_steps": 3,
        "l1_reference_grid_size": 257,
        "mean_reference_particles": 200,
        "mean_reference_repetitions": 1,
        "seed": 4,
    }
    midpoint_rows = run_smoothing_evaluation(**common, pwc_quadrature_points=1)
    averaged_rows = run_smoothing_evaluation(**common, pwc_quadrature_points=8)

    midpoint_pwc = next(row for row in midpoint_rows if row.method == "PWC")
    averaged_pwc = next(row for row in averaged_rows if row.method == "PWC")
    assert abs(midpoint_pwc.mean_error_rad - averaged_pwc.mean_error_rad) > 1.0e-5
    assert abs(midpoint_pwc.l1_error - averaged_pwc.l1_error) > 1.0e-5


def test_historical_experiment_module_routes_to_corrected_runners():
    assert historical_experiments.run_figf_pwc_benchmark is run_figf_pwc_benchmark
    assert historical_experiments.run_smoothing_evaluation is run_smoothing_evaluation
    assert historical_experiments.run_smoothing_runtime_evaluation is run_smoothing_runtime_evaluation


@pytest.mark.parametrize("quadrature_points", [0, -1])
def test_pwc_likelihood_projection_rejects_invalid_quadrature(quadrature_points):
    with pytest.raises(ValueError, match="quadrature_points"):
        make_pwc_cell_averaged_likelihoods_1d(
            (15,),
            1,
            sharpness=5.0,
            quadrature_points=quadrature_points,
        )
