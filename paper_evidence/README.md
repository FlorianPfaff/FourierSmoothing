# Public evidence snapshot for the FIGF smoothing paper

This directory is the public numerical-evidence snapshot referenced by the manuscript *Fixed-Interval Smoothing with the Fourier-Interpreted Grid Filter*.

It contains:

- one generated PGFPlots data file per evaluated method, including representation size, arithmetic mean runtime, median runtime, runtime quartiles, asymmetric interquartile errors, mean-direction error, and mean $L^1$ density error;
- `smoothing_gain_summary.csv`, containing the 500-sequence filtering-versus-smoothing state-error results and trial-bootstrap confidence intervals;
- `smoothing_evaluation_metadata.json`, separating corrected error generation from the retained controlled timing generation.

The full experiment code and tests live in this repository. The paper's complete raw 30-repetition CSV is intentionally not duplicated here; the public data files expose every plotted point and all runtime uncertainty values, while the metadata records the exact assembly key and the Git blob identifier of the full raw file in the manuscript repository.

Regenerate the plot data from a compatible `smoothing_evaluation_summary.csv` with:

```bash
python scripts/plot_runtime_accuracy_column.py \
  --results-dir /path/to/results \
  --figures-dir /path/to/figures
```

Accuracy values are arithmetic means over repetitions. Runtime plots use the median with the first and third quartiles as an interquartile interval because the PF/FFBSi timings are right-skewed. The arithmetic runtime mean remains in each data file for audit.
