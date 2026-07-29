#!/usr/bin/env python3
"""Apply the final implementation and artifact hardening pass."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    content = read(path)
    actual = content.count(old)
    if actual < count:
        raise RuntimeError(f"{path}: expected at least {count} occurrences, found {actual}: {old!r}")
    write(path, content.replace(old, new, count))


def replace_all(path: str, old: str, new: str) -> None:
    content = read(path)
    if old not in content:
        raise RuntimeError(f"{path}: missing replacement text: {old!r}")
    write(path, content.replace(old, new))


write(
    "src/fourier_smoothing/nonnegative.py",
    '''"""Tolerance-aware validation of numerically nonnegative arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_NEGATIVITY_RTOL = 1.0e-12
DEFAULT_NEGATIVITY_ATOL = 1.0e-15


def clip_roundoff_nonnegative(
    values: ArrayLike,
    name: str = "values",
    *,
    rtol: float = DEFAULT_NEGATIVITY_RTOL,
    atol: float = DEFAULT_NEGATIVITY_ATOL,
) -> NDArray[np.float64]:
    """Validate nonnegativity and clip only roundoff-scale undershoots.

    Values below ``-(atol + rtol * max(1, max(abs(values))))`` are treated as
    materially negative and raise ``ValueError``. Values inside that tolerance
    are clipped to zero. This prevents invalid transition or density inputs from
    being silently repaired while retaining protection against FFT roundoff.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("nonnegativity tolerances must be finite and nonnegative")

    scale = max(1.0, float(np.max(np.abs(array))))
    tolerance = float(atol + rtol * scale)
    minimum = float(np.min(array))
    if minimum < -tolerance:
        raise ValueError(
            f"{name} contains materially negative values: minimum={minimum:.6e}, "
            f"allowed_roundoff={tolerance:.6e}"
        )
    return np.maximum(array, 0.0)
''',
)

# Grid smoother: reject material negativity and clip only numerical roundoff.
replace(
    "src/fourier_smoothing/smoother.py",
    "from numpy.typing import ArrayLike, NDArray\n",
    "from numpy.typing import ArrayLike, NDArray\n\nfrom .nonnegative import clip_roundoff_nonnegative\n",
)
replace(
    "src/fourier_smoothing/smoother.py",
    'noise_density=_as_real(noise_density, "noise_density"),',
    'noise_density=clip_roundoff_nonnegative(noise_density, "noise_density"),',
)
replace(
    "src/fourier_smoothing/smoother.py",
    'message = _as_real(next_message, "next_message")',
    'message = clip_roundoff_nonnegative(next_message, "next_message")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    "        return np.maximum(beta, 0.0)",
    '        return clip_roundoff_nonnegative(beta, "additive backward prediction")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    'matrix = _as_real(transition_density, "transition_density")',
    'matrix = clip_roundoff_nonnegative(transition_density, "transition_density")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    'message = _as_real(next_message, "next_message")',
    'message = clip_roundoff_nonnegative(next_message, "next_message")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    "        return np.maximum(values.reshape(self.grid_shape), 0.0)",
    '        return clip_roundoff_nonnegative(values.reshape(self.grid_shape), "dense backward prediction")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    'density = _as_real(current_density, "current_density")',
    'density = clip_roundoff_nonnegative(current_density, "current_density")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    "        return np.maximum(values.reshape(self.grid_shape), 0.0)",
    '        return clip_roundoff_nonnegative(values.reshape(self.grid_shape), "dense forward prediction")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    "        if self.normalize_columns:\n",
    '        matrix = clip_roundoff_nonnegative(matrix, "transition_density")\n        if self.normalize_columns:\n',
)
replace(
    "src/fourier_smoothing/smoother.py",
    '    arr = np.maximum(_as_real(values, "values"), 0.0)',
    '    arr = clip_roundoff_nonnegative(values, "values")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    '    ``filtered[t]`` is p(x_t | z_1, ..., z_t). ``likelihoods[t]`` is p(z_t|x_t).',
    '    ``filtered[t]`` is p(x_t | z_0, ..., z_t). ``likelihoods[t]`` is p(z_t|x_t).',
)
replace(
    "src/fourier_smoothing/smoother.py",
    '    f = np.maximum(_as_real(filtered, "filtered"), 0.0)\n    ell = np.maximum(_as_real(likelihoods, "likelihoods"), 0.0)',
    '    f = clip_roundoff_nonnegative(filtered, "filtered")\n    ell = clip_roundoff_nonnegative(likelihoods, "likelihoods")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    '        beta_t = np.maximum(_as_real(backward_predict(ell[t + 1] * beta[t + 1], t), "beta_t"), 0.0)',
    '        future = clip_roundoff_nonnegative(ell[t + 1] * beta[t + 1], "future information product")\n        beta_t = clip_roundoff_nonnegative(backward_predict(future, t), "beta_t")',
)
replace(
    "src/fourier_smoothing/smoother.py",
    "    product = np.maximum(a * b, 0.0)",
    '    product = clip_roundoff_nonnegative(a * b, "smoothing product")',
)

# Re-export the tolerance-aware helper.
replace(
    "src/fourier_smoothing/__init__.py",
    "from .particle_experiments import (",
    "from .nonnegative import (\n    DEFAULT_NEGATIVITY_ATOL,\n    DEFAULT_NEGATIVITY_RTOL,\n    clip_roundoff_nonnegative,\n)\nfrom .particle_experiments import (",
)
replace(
    "src/fourier_smoothing/__init__.py",
    '    "DenseGridTransition",\n',
    '    "DEFAULT_NEGATIVITY_ATOL",\n    "DEFAULT_NEGATIVITY_RTOL",\n    "DenseGridTransition",\n',
)
replace(
    "src/fourier_smoothing/__init__.py",
    '    "circular_mean",\n',
    '    "circular_mean",\n    "clip_roundoff_nonnegative",\n',
)

# Apply the same validation policy in pairwise and EM diagnostics.
replace(
    "src/fourier_smoothing/em.py",
    "from .smoother import cell_volume_for_grid, normalize_grid_density",
    "from .nonnegative import clip_roundoff_nonnegative\nfrom .smoother import cell_volume_for_grid, normalize_grid_density",
)
replace(
    "src/fourier_smoothing/em.py",
    "    correlation = np.maximum(correlation * volume, 0.0)",
    '    correlation = clip_roundoff_nonnegative(correlation * volume, "increment correlation")',
)
old_em_helper = '''def _finite_nonnegative_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array
'''
replace(
    "src/fourier_smoothing/em.py",
    old_em_helper,
    '''def _finite_nonnegative_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    return clip_roundoff_nonnegative(values, name)
''',
)
replace(
    "src/fourier_smoothing/pairwise.py",
    "from .smoother import cell_volume_for_grid, normalize_grid_density",
    "from .nonnegative import clip_roundoff_nonnegative\nfrom .smoother import cell_volume_for_grid, normalize_grid_density",
)
replace(
    "src/fourier_smoothing/pairwise.py",
    old_em_helper,
    '''def _finite_nonnegative_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    return clip_roundoff_nonnegative(values, name)
''',
)

# Particle baseline: reject materially negative likelihood/noise/weight inputs.
replace(
    "src/fourier_smoothing/particle.py",
    "from numpy.typing import ArrayLike, NDArray\n",
    "from numpy.typing import ArrayLike, NDArray\n\nfrom .nonnegative import clip_roundoff_nonnegative\n",
)
replace(
    "src/fourier_smoothing/particle.py",
    "    probabilities = _normalize_weights(np.maximum(density, 0.0))",
    '    probabilities = _normalize_weights(clip_roundoff_nonnegative(density, "density_values"))',
)
replace(
    "src/fourier_smoothing/particle.py",
    "    likelihood_array = np.asarray(likelihoods, dtype=float)",
    '    likelihood_array = clip_roundoff_nonnegative(likelihoods, "likelihoods")',
    count=1,
)
replace(
    "src/fourier_smoothing/particle.py",
    "    noise = np.asarray(noise_density, dtype=float)",
    '    noise = clip_roundoff_nonnegative(noise_density, "noise_density")',
    count=1,
)
replace(
    "src/fourier_smoothing/particle.py",
    "    likelihood_array = np.asarray(likelihoods, dtype=float)",
    '    likelihood_array = clip_roundoff_nonnegative(likelihoods, "likelihoods")',
    count=1,
)
replace(
    "src/fourier_smoothing/particle.py",
    "    noise = np.asarray(noise_density, dtype=float)",
    '    noise = clip_roundoff_nonnegative(noise_density, "noise_density")',
    count=1,
)
replace(
    "src/fourier_smoothing/particle.py",
    "    w = np.maximum(np.asarray(weights, dtype=float), 0.0)",
    '    w = clip_roundoff_nonnegative(weights, "weights")',
)

# FFT outputs in experiments are now tolerance-checked instead of unconditionally clipped.
replace(
    "src/fourier_smoothing/experiments.py",
    "from .particle import (",
    "from .nonnegative import clip_roundoff_nonnegative\nfrom .particle import (",
)
replace(
    "src/fourier_smoothing/experiments.py",
    "    return normalize_grid_density(np.maximum(predicted * cell_volume, 0.0), cell_volume)",
    '    return normalize_grid_density(clip_roundoff_nonnegative(predicted * cell_volume, "PWC forward prediction"), cell_volume)',
    count=1,
)
replace(
    "src/fourier_smoothing/experiments.py",
    "    return np.maximum(predicted * cell_volume, 0.0)",
    '    return clip_roundoff_nonnegative(predicted * cell_volume, "PWC backward prediction")',
)
replace(
    "src/fourier_smoothing/experiments.py",
    "    return normalize_grid_density(np.maximum(predicted * cell_volume, 0.0), cell_volume)",
    '    return normalize_grid_density(clip_roundoff_nonnegative(predicted * cell_volume, "FIGF forward prediction"), cell_volume)',
    count=1,
)
replace(
    "src/fourier_smoothing/experiments.py",
    "        sqrt_values = np.sqrt(np.maximum(values, 0.0))",
    '        sqrt_values = np.sqrt(clip_roundoff_nonnegative(values, "smoothed grid density"))',
)
replace(
    "src/fourier_smoothing/experiments.py",
    "        densities[t] = normalize_grid_density(np.maximum(density, 0.0), cell_volume)",
    '        densities[t] = normalize_grid_density(clip_roundoff_nonnegative(density, "particle KDE density"), cell_volume)',
)

# Unit tests for the safety gate.
replace(
    "tests/test_smoother.py",
    "import numpy as np\n",
    "import numpy as np\nimport pytest\n",
)
replace(
    "tests/test_smoother.py",
    "    cell_volume_for_grid,\n",
    "    cell_volume_for_grid,\n    clip_roundoff_nonnegative,\n",
)
write(
    "tests/test_nonnegative.py",
    '''import numpy as np
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
''',
)

# Extend plot data with PF accuracy uncertainty as well as runtime uncertainty.
plot_path = "scripts/plot_runtime_accuracy_column.py"
replace(
    plot_path,
    '                    "mean_error_rad": float(row["mean_error_rad_mean"]),\n                    "l1_error": float(row["l1_error_mean"]),',
    '                    "mean_error_rad": float(row["mean_error_rad_mean"]),\n                    "mean_error_rad_q25": float(row["mean_error_rad_q25"]),\n                    "mean_error_rad_q75": float(row["mean_error_rad_q75"]),\n                    "l1_error": float(row["l1_error_mean"]),\n                    "l1_error_q25": float(row["l1_error_q25"]),\n                    "l1_error_q75": float(row["l1_error_q75"]),',
)
replace(
    plot_path,
    '        "runtime_err_low_ms runtime_err_high_ms mean_error l1_error"',
    '        "runtime_err_low_ms runtime_err_high_ms mean_error mean_error_q25 "\n        "mean_error_q75 mean_error_err_low mean_error_err_high l1_error l1_error_q25 "\n        "l1_error_q75 l1_error_err_low l1_error_err_high"',
)
replace(
    plot_path,
    '                        _format_data_value(float(row["mean_error_rad"])),\n                        _format_data_value(float(row["l1_error"])),',
    '                        _format_data_value(float(row["mean_error_rad"])),\n                        _format_data_value(float(row["mean_error_rad_q25"])),\n                        _format_data_value(float(row["mean_error_rad_q75"])),\n                        _format_data_value(float(row["mean_error_rad"]) - float(row["mean_error_rad_q25"])),\n                        _format_data_value(float(row["mean_error_rad_q75"]) - float(row["mean_error_rad"])),\n                        _format_data_value(float(row["l1_error"])),\n                        _format_data_value(float(row["l1_error_q25"])),\n                        _format_data_value(float(row["l1_error_q75"])),\n                        _format_data_value(float(row["l1_error"]) - float(row["l1_error_q25"])),\n                        _format_data_value(float(row["l1_error_q75"]) - float(row["l1_error"])),',
)
replace(
    plot_path,
    '''        values = []
        for row in method_rows:
            median_ms, _, _, err_low_ms, err_high_ms = _runtime_ms_and_iqr(row)
            medians.append(median_ms)
            lower_errors.append(err_low_ms)
            upper_errors.append(err_high_ms)
            values.append(float(row[metric]))
        ax.errorbar(
            medians,
            values,
            xerr=[lower_errors, upper_errors],
            label=method,
            capsize=1.5,
            elinewidth=0.6,
            **_method_plot_style(method),
        )
''',
    '''        values = []
        value_lower_errors = []
        value_upper_errors = []
        for row in method_rows:
            median_ms, _, _, err_low_ms, err_high_ms = _runtime_ms_and_iqr(row)
            medians.append(median_ms)
            lower_errors.append(err_low_ms)
            upper_errors.append(err_high_ms)
            value = float(row[metric])
            values.append(value)
            value_lower_errors.append(value - float(row[f"{metric}_q25"]))
            value_upper_errors.append(float(row[f"{metric}_q75"]) - value)
        yerr = [value_lower_errors, value_upper_errors] if method == "PF" else None
        ax.errorbar(
            medians,
            values,
            xerr=[lower_errors, upper_errors],
            yerr=yerr,
            label=method,
            capsize=1.5,
            elinewidth=0.6,
            **_method_plot_style(method),
        )
''',
)

# Update plotting regression test for the expanded generated schema.
test_plot = "tests/test_runtime_accuracy_column_plot.py"
replace(
    test_plot,
    '                "mean_error_rad_mean",\n                "l1_error_mean",',
    '                "mean_error_rad_mean",\n                "mean_error_rad_q25",\n                "mean_error_rad_q75",\n                "l1_error_mean",\n                "l1_error_q25",\n                "l1_error_q75",',
)
replace(
    test_plot,
    '                    "mean_error_rad_mean": mean_error,\n                    "l1_error_mean": l1_error,',
    '                    "mean_error_rad_mean": mean_error,\n                    "mean_error_rad_q25": 0.9 * mean_error,\n                    "mean_error_rad_q75": 1.1 * mean_error,\n                    "l1_error_mean": l1_error,\n                    "l1_error_q25": 0.9 * l1_error,\n                    "l1_error_q75": 1.1 * l1_error,',
)
replace(
    test_plot,
    '''        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error l1_error\n"
''',
    '''        "n runtime_mean_ms runtime_median_ms runtime_q25_ms runtime_q75_ms "
        "runtime_err_low_ms runtime_err_high_ms mean_error mean_error_q25 "
        "mean_error_q75 mean_error_err_low mean_error_err_high l1_error l1_error_q25 "
        "l1_error_q75 l1_error_err_low l1_error_err_high\n"
''',
)
replace(
    test_plot,
    '"9 1 0.9 0.8 1.1 0.1 0.2 0.03 0.02\\n"',
    '"9 1 0.9 0.8 1.1 0.1 0.2 0.03 0.027 0.033 0.003 0.003 0.02 0.018 0.022 0.002 0.002\\n"',
)
replace(
    test_plot,
    '"17 2 1.8 1.6 2.2 0.2 0.4 0.02 0.01\\n"',
    '"17 2 1.8 1.6 2.2 0.2 0.4 0.02 0.018 0.022 0.002 0.002 0.01 0.009 0.011 0.001 0.001\\n"',
)
replace(
    test_plot,
    '"9 1 0.9 0.8 1.1 0.1 0.2 0.025 0.015\\n"',
    '"9 1 0.9 0.8 1.1 0.1 0.2 0.025 0.0225 0.0275 0.0025 0.0025 0.015 0.0135 0.0165 0.0015 0.0015\\n"',
)
replace(
    test_plot,
    '"17 2 1.8 1.6 2.2 0.2 0.4 0.02 0.01\\n"',
    '"17 2 1.8 1.6 2.2 0.2 0.4 0.02 0.018 0.022 0.002 0.002 0.01 0.009 0.011 0.001 0.001\\n"',
    count=1,
)
replace(
    test_plot,
    '"50 10 8 6 12 2 4 0.08 0.1\\n"',
    '"50 10 8 6 12 2 4 0.08 0.072 0.088 0.008 0.008 0.1 0.09 0.11 0.01 0.01\\n"',
)
replace(
    test_plot,
    '"100 20 16 13 24 3 8 0.06 0.08\\n"',
    '"100 20 16 13 24 3 8 0.06 0.054 0.066 0.006 0.006 0.08 0.072 0.088 0.008 0.008\\n"',
)
replace(
    test_plot,
    '"9 0.8 0.7 0.6 0.9 0.1 0.2 0.05 0.08\\n"',
    '"9 0.8 0.7 0.6 0.9 0.1 0.2 0.05 0.045 0.055 0.005 0.005 0.08 0.072 0.088 0.008 0.008\\n"',
)
replace(
    test_plot,
    '"17 1.5 1.3 1.1 1.6 0.2 0.3 0.04 0.05\\n"',
    '"17 1.5 1.3 1.1 1.6 0.2 0.3 0.04 0.036 0.044 0.004 0.004 0.05 0.045 0.055 0.005 0.005\\n"',
)

# Provenance generator: schema-v2 split source metadata and optional source environment.
replace(
    "scripts/run_smoothing_evaluation.py",
    '    parser.add_argument("--error-source-git-commit", help="Code revision that generated --reuse-error-raw.")\n',
    '    parser.add_argument("--error-source-git-commit", help="Code revision that generated --reuse-error-raw.")\n    parser.add_argument(\n        "--error-source-metadata",\n        type=Path,\n        help="Optional JSON metadata describing the environment that generated --reuse-error-raw.",\n    )\n',
)
metadata_file = read("scripts/run_smoothing_evaluation.py")
start = metadata_file.index("def _write_metadata(")
end = metadata_file.index("\ndef _load_average", start)
new_metadata = '''def _write_metadata(
    args: argparse.Namespace,
    output_path: Path,
    *,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    load_before: tuple[float, float, float] | None,
    load_after: tuple[float, float, float] | None,
) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    current_environment = _environment_metadata(
        repository_root,
        started_at=started_at,
        finished_at=finished_at,
        load_before=load_before,
        load_after=load_after,
    )
    if args.reuse_error_raw is None:
        evaluation_mode = "combined"
        error_generation = {
            **current_environment,
            "generated_columns": [
                "mean_error_rad",
                "max_mean_error_rad",
                "l1_error",
                "max_l1_error",
                "min_evaluated_density",
                "max_normalization_error",
            ],
        }
    else:
        evaluation_mode = "split_accuracy_timing"
        error_generation = _error_source_metadata(args)

    metadata = {
        "schema_version": 2,
        "evaluation_mode": evaluation_mode,
        "configuration": {
            "figf_grid_sizes": args.figf_grid_sizes,
            "pwc_grid_sizes": args.pwc_grid_sizes,
            "pf_particle_counts": args.pf_particle_counts,
            "repetitions": args.repetitions,
            "time_steps": args.time_steps,
            "likelihood_sharpness": args.likelihood_sharpness,
            "noise_concentration": args.noise_concentration,
            "l1_reference_grid_size": args.l1_reference_grid_size,
            "mean_reference_particles": args.mean_reference_particles,
            "mean_reference_repetitions": args.mean_reference_repetitions,
            "particle_kde_bandwidth_scale": args.particle_kde_bandwidth_scale,
            "pwc_quadrature_points": args.pwc_quadrature_points,
            "seed": args.seed,
        },
        "error_generation": error_generation,
        "timing_generation": {
            **current_environment,
            "generated_columns": ["runtime_s"],
        },
        "runtime_scope": {
            "included": ["forward filter", "backward smoother"],
            "excluded": [
                "reference generation",
                "transition-kernel construction",
                "likelihood cell projection",
                "dense FIGF interpolation",
                "PWC densification",
                "particle KDE reconstruction",
            ],
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return path


def _environment_metadata(
    repository_root: Path,
    *,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    load_before: tuple[float, float, float] | None,
    load_after: tuple[float, float, float] | None,
) -> dict[str, object]:
    return {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_s": (finished_at - started_at).total_seconds(),
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "load_average_before": load_before,
        "load_average_after": load_after,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "pyrecest_available": importlib.util.find_spec("pyrecest") is not None,
        "git_commit": os.environ.get("FOURIER_SMOOTHING_GIT_COMMIT") or _git_commit(repository_root),
        "source_tree_sha256": _source_tree_hash(repository_root),
    }

'''
write("scripts/run_smoothing_evaluation.py", metadata_file[:start] + new_metadata + metadata_file[end + 1 :])
replace(
    "scripts/run_smoothing_evaluation.py",
    '''    return {
        "path": str(args.reuse_error_raw),
        "sha256": _file_sha256(args.reuse_error_raw),
        "host": args.error_source_host,
        "git_commit": args.error_source_git_commit,
    }
''',
    '''    metadata: dict[str, object] = {
        "path": str(args.reuse_error_raw),
        "sha256": _file_sha256(args.reuse_error_raw),
        "host": args.error_source_host,
        "git_commit": args.error_source_git_commit,
        "generated_columns": [
            "mean_error_rad",
            "max_mean_error_rad",
            "l1_error",
            "max_l1_error",
            "min_evaluated_density",
            "max_normalization_error",
        ],
    }
    if args.error_source_metadata is not None:
        metadata["source_environment"] = json.loads(
            args.error_source_metadata.read_text(encoding="utf-8")
        )
    return metadata
''',
)

# Reference-stability and adjoint validation artifact generators.
write(
    "scripts/run_reference_diagnostics.py",
    '''#!/usr/bin/env python3
"""Quantify numerical-reference stability for the paper benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fourier_smoothing import cell_volume_for_grid, make_pwc_cell_averaged_likelihoods_1d
from fourier_smoothing.experiments import (
    _evaluate_pwc_1d,
    _run_pwc_smoother_1d,
    _run_von_mises_ffbsi_1d,
    make_sharp_multimodal_likelihoods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--time-steps", type=int, default=9)
    parser.add_argument("--likelihood-sharpness", type=float, default=5.0)
    parser.add_argument("--noise-concentration", type=float, default=4.0)
    parser.add_argument("--low-grid-size", type=int, default=65_535)
    parser.add_argument("--high-grid-size", type=int, default=131_071)
    parser.add_argument("--reference-particles", type=int, default=1_000_000)
    parser.add_argument("--reference-repetitions", type=int, default=3)
    parser.add_argument("--pwc-quadrature-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.low_grid_size >= args.high_grid_size:
        raise ValueError("low-grid-size must be smaller than high-grid-size")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    likelihoods = make_sharp_multimodal_likelihoods(
        (args.low_grid_size,),
        args.time_steps,
        sharpness=args.likelihood_sharpness,
    )
    seed_sequence = np.random.SeedSequence(args.seed)
    moments = []
    rows = []
    for run, child_seed in enumerate(seed_sequence.spawn(args.reference_repetitions)):
        run_seed = int(child_seed.generate_state(1)[0])
        _, smoother = _run_von_mises_ffbsi_1d(
            likelihoods,
            args.noise_concentration,
            args.reference_particles,
            seed=run_seed,
        )
        run_moments = np.mean(np.exp(1j * smoother.trajectories), axis=0)
        moments.append(run_moments)
        for time_step, moment in enumerate(run_moments):
            rows.append(
                {
                    "run": run,
                    "seed": run_seed,
                    "time_step": time_step,
                    "moment_real": float(moment.real),
                    "moment_imag": float(moment.imag),
                    "moment_magnitude": float(abs(moment)),
                    "mean_direction_rad": float(np.mod(np.angle(moment), 2.0 * np.pi)),
                }
            )

    moment_array = np.stack(moments, axis=0)
    aggregate = np.mean(moment_array, axis=0)
    aggregate_angles = np.angle(aggregate)
    deviations = np.abs(np.angle(np.exp(1j * (np.angle(moment_array) - aggregate_angles[None, :]))))

    low_likelihoods = make_pwc_cell_averaged_likelihoods_1d(
        (args.low_grid_size,),
        args.time_steps,
        sharpness=args.likelihood_sharpness,
        quadrature_points=args.pwc_quadrature_points,
    )
    high_likelihoods = make_pwc_cell_averaged_likelihoods_1d(
        (args.high_grid_size,),
        args.time_steps,
        sharpness=args.likelihood_sharpness,
        quadrature_points=args.pwc_quadrature_points,
    )
    low_density = _run_pwc_smoother_1d(
        low_likelihoods,
        args.noise_concentration,
        quadrature_points=args.pwc_quadrature_points,
    )
    high_density = _run_pwc_smoother_1d(
        high_likelihoods,
        args.noise_concentration,
        quadrature_points=args.pwc_quadrature_points,
    )
    low_on_high = _evaluate_pwc_1d(low_density, args.high_grid_size)
    high_volume = cell_volume_for_grid((args.high_grid_size,))
    l1_by_time = np.sum(np.abs(low_on_high - high_density), axis=1) * high_volume

    csv_path = args.output_dir / "reference_first_moments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "max_between_run_mean_direction_deviation_rad": float(np.max(deviations)),
        "mean_between_run_mean_direction_deviation_rad": float(np.mean(deviations)),
        "pwc_reference_refinement_mean_l1": float(np.mean(l1_by_time)),
        "pwc_reference_refinement_max_l1": float(np.max(l1_by_time)),
        "pwc_reference_refinement_l1_by_time": [float(value) for value in l1_by_time],
        "aggregate_mean_directions_rad": [
            float(np.mod(value, 2.0 * np.pi)) for value in aggregate_angles
        ],
    }
    (args.output_dir / "reference_stability.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(csv_path)
    print(args.output_dir / "reference_stability.json")


if __name__ == "__main__":
    main()
''',
)
write(
    "scripts/run_adjoint_validation.py",
    '''#!/usr/bin/env python3
"""Validate weighted adjoint identities for additive and dense transitions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fourier_smoothing import (
    DenseGridTransition,
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    normalize_grid_density,
    torus_additive_transition_density_matrix,
    torus_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def weighted_inner(left, right, cell_volume):
    return float(np.sum(left * right) * cell_volume)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    grid_shape = (7, 9)
    x, y = torus_grid(grid_shape)
    volume = cell_volume_for_grid(grid_shape)
    noise = normalize_grid_density(np.exp(1.1 * np.cos(x) + 0.7 * np.cos(y)), volume)
    density = normalize_grid_density(0.5 + rng.random(grid_shape), volume)
    message = 0.5 + rng.random(grid_shape)
    fft_adjoint = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)
    matrix = torus_additive_transition_density_matrix(noise, cell_volume=volume)
    dense = DenseGridTransition.for_grid_shape(matrix, grid_shape, cell_volume=volume)
    forward = dense.forward_predict(density, 0)
    rows.append(
        {
            "case": "2d_additive_weighted_adjoint",
            "absolute_error": abs(weighted_inner(forward, message, volume) - weighted_inner(density, fft_adjoint(message, 0), volume)),
        }
    )
    rows.append(
        {
            "case": "2d_additive_fft_vs_dense_adjoint",
            "absolute_error": float(np.max(np.abs(fft_adjoint(message, 0) - dense(message, 0)))),
        }
    )

    grid_shape = (13,)
    volume = cell_volume_for_grid(grid_shape)
    raw_matrix = 0.1 + rng.random((13, 13))
    transition = DenseGridTransition.for_grid_shape(raw_matrix, grid_shape, cell_volume=volume)
    density = normalize_grid_density(0.5 + rng.random(grid_shape), volume)
    message = 0.5 + rng.random(grid_shape)
    rows.append(
        {
            "case": "1d_nonadditive_weighted_adjoint",
            "absolute_error": abs(
                weighted_inner(transition.forward_predict(density, 0), message, volume)
                - weighted_inner(density, transition(message, 0), volume)
            ),
        }
    )

    output = args.output_dir / "adjoint_validation.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "absolute_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
''',
)
write(
    "tests/test_artifact_diagnostics.py",
    '''import csv
import json
import subprocess
import sys
from pathlib import Path


def test_adjoint_validation_script(tmp_path):
    subprocess.run(
        [sys.executable, "scripts/run_adjoint_validation.py", "--output-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    with (tmp_path / "adjoint_validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert max(float(row["absolute_error"]) for row in rows) < 1.0e-11


def test_reference_diagnostics_smoke(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/run_reference_diagnostics.py",
            "--output-dir",
            str(tmp_path),
            "--time-steps",
            "3",
            "--low-grid-size",
            "31",
            "--high-grid-size",
            "63",
            "--reference-particles",
            "200",
            "--reference-repetitions",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    summary = json.loads((tmp_path / "reference_stability.json").read_text(encoding="utf-8"))
    assert summary["max_between_run_mean_direction_deviation_rad"] >= 0.0
    assert summary["pwc_reference_refinement_mean_l1"] >= 0.0
''',
)

# Explicit license and citation metadata.
write(
    "LICENSE",
    '''MIT License

Copyright (c) 2026 Florian Pfaff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
''',
)
write(
    "CITATION.cff",
    '''cff-version: 1.2.0
message: "Please cite the accompanying paper and this software release."
title: "FourierSmoothing"
type: software
authors:
  - family-names: Pfaff
    given-names: Florian
repository-code: "https://github.com/FlorianPfaff/FourierSmoothing"
version: 0.1.0
date-released: 2026-07-27
license: MIT
preferred-citation:
  type: article
  title: "Fixed-Interval Smoothing with the Fourier-Interpreted Grid Filter"
  authors:
    - family-names: Pfaff
      given-names: Florian
  year: 2026
''',
)
replace(
    "pyproject.toml",
    'authors = [{ name = "Florian Pfaff" }]\n',
    'authors = [{ name = "Florian Pfaff" }]\nlicense = { file = "LICENSE" }\nkeywords = ["Bayesian smoothing", "Fourier filter", "grid filter", "hypertorus"]\n',
)

# README additions.
replace(
    "README.md",
    "## Scope\n",
    '''## Numerical nonnegativity policy

Grid densities, likelihoods, and sampled transitions are validated as nonnegative. Only undershoots within a scale-aware floating-point tolerance are clipped to zero; materially negative values raise `ValueError`. This prevents an invalid signed transition approximation from being silently repaired.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The implementation is released under the MIT License; see `LICENSE`.

## Scope
''',
)

# Remove the one-shot patch mechanism from the resulting branch.
(ROOT / ".github/workflows/apply-final-hardening.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
