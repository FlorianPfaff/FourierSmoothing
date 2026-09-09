# Publication figures

`render_publication_figures.py` is the canonical renderer for the Fourier
smoothing manuscript. The shared visual specification is in
`paper_plot_style.py`, following the final-size typography, line weights,
markers, colors, and restrained grids of `FlorianPfaff/S3F-Improved-Paper`.

## Restyle without rerunning experiments

From the code repository:

```bash
python -m pip install -e '.[paper]'
python scripts/render_publication_figures.py \
  --results-dir ../2026-07-FourierSmoothing-Paper/results \
  --figures-dir ../2026-07-FourierSmoothing-Paper/figures
```

For an independent preview from the public evidence snapshot:

```bash
python scripts/render_publication_figures.py \
  --results-dir paper_evidence --figures-dir publication_preview
```

This reads the frozen evaluation summary and reference diagnostics. It does
not rerun experiments, estimate new runtimes, change a statistic, or write
numerical results. The space-time illustration reuses the existing illustrative
case and its original parameters; it is not a new benchmark.

To update only the space-time illustration while preserving the accuracy and
runtime PDFs and their manifest metadata, add `--only-space-time` to the command
above. This requires an existing manifest whose input and PDF hashes still match.

Eleven PDFs and matching 300-dpi PNG previews are generated. The space-time
panels are `smoothing_space_time_filtering.pdf`,
`smoothing_space_time_backward.pdf`, and `smoothing_space_time_smoothing.pdf`.
Each is rendered at `0.32\textwidth`, with a horizontal colorbar labeled
"relative value" above the heatmap. Panel captions and labels belong to LaTeX;
no panel titles are embedded in the images. `smoothing_space_time.pdf` remains
as a combined preview. The angular range remains zero to two pi, and white
circular-mean traces remain on the filtering and smoothing panels only.

The remaining outputs are five column-sized comparison plots and two combined
previews. The manuscript includes the two accuracy panels at column width and
the three runtime panels at `0.32\textwidth`. After a full rendering pass, run
`python scripts/render_runtime_subfigures.py` in the paper repository to restore
its final-size runtime fonts. The space-time-only mode preserves this override.

Accuracy coordinates remain arithmetic means with PF IQRs. Runtimes remain
medians with IQRs. Interval endpoints are drawn directly, including the valid
case of a mean outside its IQR. Lines connect increasing grid/particle counts,
not runtime-sorted points. Coincident FIGFAN/FIGFDN points use nested markers
and drawing order, never numerical offsets. The runtime-scaling panel shows
only the shared FIGF recursion. Gray reference lines are diagnostics, not
bounds on intrinsic method error.

`publication_manifest.json` records the renderer revision, numerical-input
hashes, generated-PDF hashes, and all plotted summary rows. The paper repository
validates those rows independently against the existing numerical `.dat`
tables. The `space_time_subfigures` profile also records source-file hashes,
panel dimensions, and the unchanged illustrative-case parameters. Figure
regeneration and validation are manual; manuscript compilation is in Overleaf.

The older `plot_paper_results.py`, `plot_runtime_accuracy_column.py`, and
`plot_smoothing_hero.py` entry points are retained for historical reproduction.
They are not the final styling pass. In particular, the old TikZ figure templates
are archival alternatives, not the manuscript's active rendering path.
