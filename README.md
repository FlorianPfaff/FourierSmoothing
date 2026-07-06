# FourierSmoothing

Experimental reference implementation of fixed-interval smoothing for Fourier/grid filters on hypertori.

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
- a first Fourier identity smoother for additive torus dynamics,
- tests for the identity-transition case and the Fourier/grid equivalence of the backward prediction.

## Install

```bash
python -m pip install -e .[test]
```

## Run tests

```bash
pytest
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

This is deliberately a first implementation. The multiplication of Fourier coefficient arrays is currently implemented through grid transforms for clarity. A later optimized version can replace this with zero-padded/truncated coefficient convolutions and add spherical-harmonic support.
