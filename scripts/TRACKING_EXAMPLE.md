# Representative angular-tracking example

This adds a state-truth trajectory and posterior-density views for the paper's
existing 500-sequence smoothing-gain experiment. It does not introduce a new
sensor model, change the numerical smoother, or edit the manuscript.

## Reproduce

From the root of `FlorianPfaff/FourierSmoothing`:

```bash
python -m pip install -e '.[paper]'
python scripts/plot_tracking_example.py --output-dir generated_tracking_example
```

For headless machines, set `MPLBACKEND=Agg`. Each plot is exported as PDF, SVG,
and 240-dpi PNG under `generated_tracking_example/figures/`:

| File stem | Content |
| --- | --- |
| `tracking_trajectory` | True angle, observations, filtering means, smoothing means |
| `tracking_density_t00` | Initial filtering and smoothing densities, true angle, observation |
| `tracking_density_t04` | Most ambiguous nonterminal filtering density in the selected trial |
| `tracking_filter_density` | Filtering density over angle and time |
| `tracking_smoother_density` | Smoothing density over angle and time |

The `t04` suffix is the default result of the rule below, not a hard-coded time.
The initial snapshot and the selected snapshot are deduplicated when both are
at time zero. Line plots have vector geometry; heatmap meshes are rasterized
inside PDF/SVG to keep files compact. Both heatmaps share the same **linear
probability-density scale**, without columnwise peak normalization or temporal
interpolation. Lines break at the angular seam instead of connecting across it.

Generated NPZ, CSV, and JSON files retain the observations, latent states,
posterior grids, reconstructed densities, trial errors, selection scores,
configuration, revision and environment metadata. Figures are generated
artifacts rather than checked-in publication-layout files.

To regenerate only the plots from saved numerical results:

```bash
python scripts/plot_tracking_example.py --output-dir generated_tracking_example --plot-only
```

To write the figures to a separate paper checkout without changing its prose:

```bash
python scripts/plot_tracking_example.py \
  --output-dir generated_tracking_example \
  --figures-dir ../2026-07-FourierSmoothing-Paper/figures
```

A small nonpublication smoke run is available with
`--n-trials 3 --grid-size 63 --time-steps 5`. Optional `--trial` and `--snapshot`
arguments use **zero-based indices**; explicit overrides are recorded in the
metadata. No override is needed for the default figures.

## Model and selection

The model and RNG draw order follow `run_smoothing_gain_evaluation` at source
revision `4f210310306daaee115f7abedf2f435fca34cefb`: a von Mises circular random
walk with prior concentration 1, process concentration 8, and observation-noise
concentration 12. The observation noise is a mixture with probability 0.3 of
a component shifted by 2.35 rad. The default is 500 trials, 20 time steps,
1,023 grid points, and seed 21. The shifted-noise component is latent simulation
diagnostic information, **not an input to the filter**.

The existing repository functions perform forward filtering, backward-information
smoothing, circular means, and FIGFDN reconstruction. Only the simulator's draw
sequence is repeated here so that truth, measurements, and full densities remain
available; the production smoother is not copied or replaced.

The representative trial minimizes the squared distance from the componentwise
medians of the filtering and smoothing trial MAEs, with each coordinate divided
by its interquartile range. MAEs exclude the terminal time, at which filtering
and smoothing coincide. Ties use the earliest trial. This criterion uses the
simulation errors, but **does not maximize the smoothing gain**.

Within that trial, the additional density snapshot maximizes the filtering
secondary-to-primary local-peak-height ratio over nonterminal times, requiring
at least 0.75 rad circular separation. This is a **peak-height ratio**, not
posterior mode probability. The initial-time density is also always shown.
Neither snapshot selection uses the improvement at that time.

## Default results

The median-nearest trial is zero-based index **59** (trial **60** of 500).
Its filtering and smoothing MAEs are both at the **51.6th empirical percentile**.

| Quantity | Filtering | Smoothing |
| --- | ---: | ---: |
| Selected trial MAE, rad | 0.2532224381 | 0.1776645919 |
| All 500 trials MAE, rad | 0.2972943386 | 0.1921700633 |

The selected-trial reduction is **29.8385%**. The ensemble reduction is
**35.3603%**, with trial-bootstrap 95% interval **[32.5632%, 38.0172%]**,
reproducing the retained paper evidence. The automatic density time is **t = 4**;
the filtered secondary-to-primary peak-height ratio is **0.6782**.

At t = 0, future information suppresses a secondary mode. At t = 4, the density
view also illustrates that a sharper posterior does **not** imply that its mean
is closer to the realized true state at every time. These are illustrations of
the existing simulated experiment, not a real-data application or a new
aggregate performance claim. At this grid resolution, the plotted FIGFDN
reconstruction agrees with FIGFAN to floating-point precision.

## Validation and CI artifacts

```bash
PYTHONPATH=src pytest -q tests/test_tracking_example.py
```

Tests cover reproduction of published MAEs and bootstrap bounds, deterministic
selection, RNG replay, nonnegativity and normalization, terminal identity,
nonsymmetric-transition adjoint validation, seam handling, invalid inputs,
metadata overrides, and all plot exports.

The **Tracking example figures** workflow generates the full default example
on relevant pull requests and can also be run manually. Its
`fourier-smoothing-tracking-example` artifact contains all figures and numeric
outputs. The existing Tests workflow runs the new tests with the complete
repository suite. No manuscript files are modified.
