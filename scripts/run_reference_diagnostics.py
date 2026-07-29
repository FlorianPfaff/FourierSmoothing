#!/usr/bin/env python3
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
        json.dumps(summary, indent=2, sort_keys=True) + "
",
        encoding="utf-8",
    )
    print(csv_path)
    print(args.output_dir / "reference_stability.json")


if __name__ == "__main__":
    main()
