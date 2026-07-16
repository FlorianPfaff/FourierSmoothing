# FourierSmoothing

Experimental reference implementation of fixed-interval smoothing for Fourier/grid filters on hypertori.

This repository is for **code**: Python package code, experiment scripts, tests, and reusable numerical utilities. Manuscript files, committed figures, and generated result tables belong in the separate `FlorianPfaff/2026-07-FourierSmoothing-Paper` repository.

The current implementation focuses on the backward-information smoother

```math
\beta_t(x_t)=\int p(x_{t+1}\mid x_t)\,\ell_{t+1}(x_{t+1})\,\beta_{t+1}(x_{t+1})\,dx_{t+1}
```

with the smoothed density

```math
p(x_t\mid z_{0:T}) \propto p(x_t\mid z_{0:t})\,\beta_t(x_t).
```

It includes:

- a generic grid-based backward-information smoother,
- a dense transition-matrix path for histogram/PWC forward-backward smoothing,
- pairwise two-time smoothed marginals for EM-style transition/noise estimation,
- FFT-based additive-noise increment posteriors and a nonparametric M-step without materializing pairwise matrices,
- a periodic-grid transition for the additive identity model on `T^d`,
- dense general-transition forward and adjoint contractions in Fourier coefficients,
- FFT helpers for complex Fourier coefficients in NumPy order,
- aliasing-free Fourier multiplication by linear coefficient convolution and truncation,
- dense-grid evaluation of truncated Fourier coefficient tensors,
- a first Fourier identity smoother for additive torus dynamics,
- a one-dimensional torus bootstrap particle-filter / FFBSi particle-smoother diagnostic baseline,
- reproducible FIGF/PWC benchmark, particle-baseline, and truncation-negativity diagnostic writers for paper result CSVs,
- plotting and LaTeX-table scripts that write figures/tables to the paper repository,
- tests for identity and diffusive dynamics, general transitions, pairwise marginals, multidimensional EM statistics, aliasing, particle baselines, and truncation diagnostics.

## Install

```bash
python -m pip install -e .[test]
```

For paper figures, install the plotting extra:

```bash
python -m pip install -e .[paper]
```

## Run tests

```bash
pytest
```

## One-command paper artifact pipeline

For a smoke run that generates CSVs, figures, and LaTeX tables in a local artifact directory:

```bash
python scripts/run_paper_artifact_pipeline.py --profile smoke --output-root generated_paper_artifacts
```

The smoke profile includes a compact FIGFAN/FIGFDN/PWC/PF evaluation so the complete result-to-figure-to-table path is exercised in CI.

For a paper-scale diagnostic run, use:

```bash
python scripts/run_paper_artifact_pipeline.py \
  --profile paper \
  --results-dir ../2026-07-FourierSmoothing-Paper/results \
  --figures-dir ../2026-07-FourierSmoothing-Paper/figures \
  --tables-dir ../2026-07-FourierSmoothing-Paper/tex/tables
```

The paper profile does not rerun the main timing evaluation by default, because its final runtimes should be measured on the designated server. Add `--include-smoothing-evaluation` to run it as part of the pipeline, or `--no-include-smoothing-evaluation` to skip it explicitly. The GitHub Actions workflow `Paper benchmark on gpuserver6000` runs the controlled benchmark and uploads generated CSV, figure, table, and environment artifacts.

## Generate paper results

Run experiment code from this repository, but write generated outputs to the paper repository:

```bash
ssh gpuserver6000
cd /path/to/FourierSmoothing
export PYTHONPATH=$PWD/src:/path/to/PyRecEst/src
python scripts/run_smoothing_evaluation.py --output-dir ../2026-07-FourierSmoothing-Paper/results
python scripts/run_smoothing_error_reduction.py --output-dir ../2026-07-FourierSmoothing-Paper/results
```

The main benchmark writes `smoothing_evaluation_raw.csv`, `smoothing_evaluation_summary.csv`, and `smoothing_evaluation_metadata.json`. It compares FIGFAN, FIGFDN, PWC, and a bootstrap-PF/FFBSi smoother by mean-direction error, $L^1$ density error, runtime, and error over runtime over 30 repetitions. The mean reference aggregates three independent FFBSi runs with one million particles and trajectories each. The density reference is a PWC smoother with 65,535 cells. Particle marginals are converted to continuous densities with a wrapped-normal KDE using bandwidth $N^{-1/5}$. The summary uses `pyrecest.evaluation.summarize_parameter_sweep_records` when PyRecEst is on `PYTHONPATH`; the repository contains an equivalent fallback so its smoke pipeline remains self-contained.

Accuracy generation can be separated from controlled timing. Given a compatible raw CSV whose `(method, parameter, repetition)` keys match the requested sweep, `--reuse-error-raw` skips references, interpolation, densification, and KDE reconstruction; it reruns only the forward filters and backward smoothers and replaces every runtime. For example, errors may be generated on `gpuserver4090` and timed on an idle `gpuserver6000` via:

```bash
python scripts/run_smoothing_evaluation.py \
  --output-dir ../2026-07-FourierSmoothing-Paper/results \
  --reuse-error-raw /path/to/gpuserver4090/smoothing_evaluation_raw.csv \
  --error-source-host gpuserver4090 \
  --error-source-git-commit <revision>
```

The smoothing-gain command writes `smoothing_gain_raw.csv` and `smoothing_gain_summary.csv`. It compares filtered and smoothed circular means with simulated latent states over 500 sequences and reports a trial-bootstrap confidence interval. It deliberately makes no filter-to-smoother $L^1$ reduction claim because filtering and smoothing target different posterior densities.

Runtime covers the forward filter and backward smoother. Dense FIGF interpolation, PWC evaluation, PF KDE reconstruction, transition-kernel construction, and reference generation are excluded. Thus FIGFAN and FIGFDN share one runtime curve. Paper timing values must be measured on `gpuserver6000` while the server is idle. The metadata sidecar records the host, load averages, software versions, git revision, source-tree hash, full configuration, and timing scope. A staged source tree without `.git` can supply the revision through `FOURIER_SMOOTHING_GIT_COMMIT`.

Additional diagnostics are still available:

```bash
python scripts/run_identity_torus_experiment.py --output-dir ../2026-07-FourierSmoothing-Paper/results
python scripts/run_truncation_negativity_diagnostic.py --output-dir ../2026-07-FourierSmoothing-Paper/results
python scripts/run_particle_baseline_experiment.py --output-dir ../2026-07-FourierSmoothing-Paper/results
```

## Generate paper figures and tables

After generating CSV results, create figures and tables in the paper repository via:

```bash
python scripts/plot_paper_results.py --results-dir ../2026-07-FourierSmoothing-Paper/results --figures-dir ../2026-07-FourierSmoothing-Paper/figures
python scripts/write_latex_tables.py --results-dir ../2026-07-FourierSmoothing-Paper/results --tables-dir ../2026-07-FourierSmoothing-Paper/tex/tables
python scripts/plot_smoothing_hero.py --figures-dir ../2026-07-FourierSmoothing-Paper/figures
```

## Minimal example

```python
import numpy as np
from fourier_smoothing import (
    TorusAdditiveGridTransition,
    cell_volume_for_grid,
    grid_backward_information_smoother,
    normalize_grid_density,
    torus_grid,
)

grid_shape = (64,)
(x,) = torus_grid(grid_shape)
cell_volume = cell_volume_for_grid(grid_shape)

likelihoods = np.stack([
    1.0 + 0.2 * np.cos(x),
    1.0 + 0.3 * np.cos(x - 0.7),
    1.0 + 0.2 * np.sin(2.0 * x),
])

noise = np.exp(3.0 * np.cos(x))
noise = normalize_grid_density(noise, cell_volume)
transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)

filtered = [normalize_grid_density(likelihoods[0], cell_volume)]
for likelihood in likelihoods[1:]:
    predicted = np.fft.ifft(np.fft.fft(noise) * np.fft.fft(filtered[-1])).real
    predicted *= cell_volume
    filtered.append(normalize_grid_density(predicted * likelihood, cell_volume))
filtered = np.stack(filtered)

result = grid_backward_information_smoother(filtered, likelihoods, transition)
smoothed = result.smoothed
```

## Scope

The efficient special case is additive torus dynamics, where forward and backward propagation reduce to FFT-based cyclic convolution/correlation. Dense grid and Fourier transition operators are also provided for general models; scalable sparse or low-rank general-transition representations remain future work. The coefficient-only identity smoother retains both aliasing-free truncated-convolution and same-grid diagnostic multiplication paths.
