import subprocess
import sys


METHOD_DATA_FILES = (
    "smoothing_evaluation_figfan.dat",
    "smoothing_evaluation_figfdn.dat",
    "smoothing_evaluation_pf.dat",
    "smoothing_evaluation_pwc.dat",
)


def test_write_publication_plot_sources(tmp_path):
    figures_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "scripts/write_publication_plot_sources.py",
            "--figures-dir",
            str(figures_dir),
        ],
        check=True,
    )

    accuracy = (figures_dir / "smoothing_accuracy_by_parameter.tex").read_text(encoding="utf-8")
    runtime = (figures_dir / "smoothing_runtime_accuracy_column.tex").read_text(encoding="utf-8")

    for filename in METHOD_DATA_FILES:
        assert filename in accuracy
        assert filename in runtime

    assert "coordinates {" not in accuracy
    assert "coordinates {" not in runtime
    assert "legend to name=fsaccuracylegend" in accuracy
    assert "legend to name=fsaccuracyruntimelegend" in runtime
    assert "runtime_median_ms" in runtime
    assert "runtime_err_low_ms" in runtime
    assert "runtime_err_high_ms" in runtime
    assert "runtime_mean_ms" not in runtime
    assert "error bars" in accuracy
    assert "error bars" in runtime
    assert "HTML}{0072B2}" in accuracy
    assert "HTML}{D55E00}" in runtime
