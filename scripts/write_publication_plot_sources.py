#!/usr/bin/env python
"""Write the publication PGFPlots sources used by the Fourier-smoothing paper.

The numerical coordinates remain in the generated ``figures/data/*.dat`` files.
This script owns only the visual encoding and layout, so rerunning experiments
cannot silently revert the manuscript to older plot styling.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE_NAMES = (
    "smoothing_accuracy_by_parameter.tex",
    "smoothing_runtime_accuracy_column.tex",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../2026-07-FourierSmoothing-Paper/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_dir = Path(__file__).resolve().with_name("figure_templates")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    for name in TEMPLATE_NAMES:
        source = template_dir / name
        destination = args.figures_dir / name
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        print(destination)


if __name__ == "__main__":
    main()
