# FourierSmoothing

Experimental reference implementation of fixed-interval smoothing for Fourier/grid filters on hypertori.

This repository is for **code**: Python package code, experiment scripts, tests, and reusable numerical utilities. Manuscript files, committed figures, and generated result tables belong in the separate `FlorianPfaff/2026-07-FourierSmoothing-Paper` repository.

The current implementation focuses on the backward-information smoother

```math
\beta_t(x_t)=\int p(x_{t+1}\mid x_t)\,\ell_{t+1}(x_{t+1})\,\beta_{t+1}(x_{t+1})\,dx_{t+1}
```

with the smoothed density

```math
p(x_t\mid z_{1:T}) \propto p(x_t\mid z_{1:t})\,\beta_t(x_t).
```

It includes:

- a generic grid-based backward-information smoother,
- a dense transition-matrix path for histogram/PWC forward-backward smoothing,
- pairwise two-time smoothed marginals for EM-style transition/noise estimation,
- a periodic-grid transition for the additive identity model on `T^d`,
- FFT helpers for complex Fourier coefficients in NumPy order,
- aliasing-free Fourier multiplication by linear coefficient convolution and truncation,
- dense-grid evaluation of truncated Fourier coefficient tensors,
- a first Fourier identity smoother for additive torus dynamics,
- a one-dimensional torus bootstrap particle-filter / FFBSi particle-smoother diagnostic baseline,
- reproducible FIGF/PWC benchmark, particle-baseline, and truncation-negativity diagnostic writers for paper result CSVs,
- plotting and LaTeX-table scripts that write figures/tables to the paper repository,
- tests for identity, diffusive transition, pairwise marginals, aliasing, particle-baseline, and truncation-diagnostic cases.

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

The paper profile does not rerun the main timing evaluation by default, because its final runtimes should be measured on the designated server. Add `--include-smoothing-evaluation` to run it as part of the pipeline, or `--no-include-smoothing-evaluation` to skip it explicitly. The GitHub Actions workflow `Paper artifacts` runs the smoke pipeline and uploads generated CSV, figure, and table artifacts.

## Generate paper results

Run experiment code from this repository, but write generated outputs to the paper repository:

```bash
ssh gpuserver6000
cd /path/to/FourierSmoothing
export PYTHONPATH=$PWD/src:/path/to/PyRecEst/src
python scripts/run_smoothing_evaluation.py --output-dir ../2026-07-FourierSmoothing-Paper/results
```

The main paper benchmark writes `smoothing_evaluation_raw.csv` and `smoothing_evaluation_summary.csv`. It compares FIGFAN, FIGFDN, PWC, and PF by mean error, L1 density error, runtime, and error over runtime. The mean reference is a path-space PF smoother with 100,000 particles by default; the L1 reference is a high-resolution PWC smoother. Runtime values used in the paper should be measured on `gpuserver6000` while the server is idle.

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

filtered = []
cumulative = np.ones_like(x)
for likelihood in likelihoods:
    cumulative *= likelihood
    filtered.append(normalize_grid_density(cumulative, cell_volume))
filtered = np.stack(filtered)

noise = np.exp(3.0 * np.cos(x))
noise = normalize_grid_density(noise, cell_volume)
transition = TorusAdditiveGridTransition.for_grid_shape(noise, grid_shape)

result = grid_backward_information_smoother(filtered, likelihoods, transition)
smoothed = result.smoothed
```

## Scope

This is deliberately a first implementation. The Fourier smoother now has both an aliasing-free truncated coefficient-convolution path and a grid-transform path. The grid-transform path remains useful for diagnostics and equivalence checks against grid smoothers.
