# Public evidence snapshot for the FIGF smoothing paper

This directory is the public numerical-evidence snapshot referenced by the manuscript *Fixed-Interval Smoothing with the Fourier-Interpreted Grid Filter*.

It contains:

- one generated PGFPlots data file per evaluated method, including controlled runtime mean, median, quartiles and IQR errors, plus mean-direction and $L^1$ accuracy means and quartiles;
- `smoothing_accuracy_raw.csv`, the complete 30-repetition accuracy rows with runtime removed to avoid mixing hosted-runner timings with the controlled timing study;
- `smoothing_accuracy_summary.csv` and `accuracy_regeneration_environment.json`;
- complete raw and summarized 500-sequence filtering-versus-smoothing state-error results;
- the three high-sample reference first-moment sequences and a 65,535-versus-131,071-cell reference-refinement diagnostic;
- weighted-adjoint validation for two-dimensional additive and one-dimensional nonadditive transitions;
- split provenance metadata for the corrected errors and retained controlled timings.

The exact per-repetition controlled timing rows were produced on the designated timing host and remain identified by their Git blob in the manuscript repository. Every timing value used in the paper—mean, median, quartiles and IQR errors—is reproduced in the public method data files. Accuracy and state-error raw rows are fully public here.

Accuracy values are arithmetic means over repetitions. Runtime plots use the median with the first and third quartiles because PF/FFBSi timings are right-skewed. PF accuracy intervals use the corresponding accuracy quartiles.
