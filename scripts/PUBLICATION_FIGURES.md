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

Eight PDFs and matching 300-dpi PNG previews are generated. The manuscript
uses `smoothing_space_time.pdf`, `smoothing_accuracy_by_parameter.pdf`, and
`smoothing_runtime_accuracy_summary.pdf` at full text width. Five individual
charts are also generated at column width. Each panel is rendered at its actual
printed dimensions before vector-PDF assembly; do not shrink the three-panel
runtime comparison into a single column.

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
tables. Its workflow pins the reviewed renderer commit and the rendering
package versions, compiles and validates the manuscript, then commits the PDFs,
previews, and manifest on successful pushes to `main`.

The older `plot_paper_results.py`, `plot_runtime_accuracy_column.py`, and
`plot_smoothing_hero.py` entry points are retained for historical reproduction.
They are not the final styling pass. In particular, the old TikZ figure templates
are archival alternatives, not the manuscript's active rendering path.
