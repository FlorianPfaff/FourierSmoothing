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
- a periodic-grid transition for the additive identity model on `T^d`,
- FFT helpers for complex Fourier coefficients in NumPy order,
- aliasing-free Fourier multiplication by linear coefficient convolution and truncation,
- dense-grid evaluation of truncated Fourier coefficient tensors,
- a first Fourier identity smoother for additive torus dynamics,
- a one-dimensional torus bootstrap particle-filter / FFBSi particle-smoother baseline,
- reproducible benchmark, particle-baseline, and truncation-negativity diagnostic writers for paper result CSVs,
- plotting scripts that write figures to the paper repository,
- tests for identity, diffusive transition, aliasing, particle-baseline, and truncation-diagnostic cases.

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

## Generate paper results

Run experiment code from this repository, but write generated outputs to the paper repository:

```bash
python scripts/run_identity_torus_experiment.py --output-dir ../2026-07-FourierSmoothing-Paper/results --grid-sizes 15 31 63 127 --repetitions 5

python scripts/run_truncation_negativity_diagnostic.py --output-dir ../2026-07-FourierSmoothing-Paper/results --k-max-values 1 2 3 5 8 12 16 --sharpness-values 2 5 9 13

python scripts/run_particle_baseline_experiment.py --output-dir ../2026-07-FourierSmoothing-Paper/results --n-particles-values 100 300 1000 --n-trajectories 200 --repetitions 5
```

The identity benchmark writes `identity_torus_benchmark.csv`. The truncation diagnostic writes `truncation_negativity_diagnostic.csv`. The particle baseline writes `particle_smoother_baseline.csv`.

The default Fourier multiplication mode is `truncated_convolution`, which performs an aliasing-free coefficient convolution followed by truncation. To reproduce same-grid multiplication exactly in the identity benchmark, add:

```bash
--fourier-multiplication grid
```

## Generate paper figures

After generating CSV results, create figures in the paper repository via:

```bash
python scripts/plot_paper_results.py --results-dir ../2026-07-FourierSmoothing-Paper/results --figures-dir ../2026-07-FourierSmoothing-Paper/figures
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
