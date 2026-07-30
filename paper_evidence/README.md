# Public evidence snapshot for the FIGF smoothing paper

This directory is the public numerical-evidence snapshot referenced by the manuscript *Fixed-Interval Smoothing with the Fourier-Interpreted Grid Filter*.

It contains:

- one generated PGFPlots data file per evaluated method, including final-release controlled runtime mean, median, quartiles and IQR errors, plus mean-direction and $L^1$ accuracy means and quartiles;
- `smoothing_accuracy_raw.csv`, the complete 30-repetition accuracy rows;
- `smoothing_runtime_raw.csv`, the complete 30-repetition controlled timing rows measured on implementation commit `194d1f9b4bb49180831a9162358302f189141633`;
- `smoothing_evaluation_raw.csv`, the exact keywise combination used for the paper figures and summary;
- complete raw and summarized 500-sequence filtering-versus-smoothing state-error results;
- the three high-sample reference first-moment sequences and the 65,535-versus-131,071-cell reference-refinement diagnostic;
- weighted-adjoint validation for two-dimensional additive and one-dimensional nonadditive transitions;
- split provenance metadata for accuracy and final-release timing generation.

Accuracy values are arithmetic means over repetitions. Runtime plots use the median with the first and third quartiles because PF/FFBSi timings are right-skewed. PF accuracy intervals use the corresponding accuracy quartiles.
