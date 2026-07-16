#!/usr/bin/env python
"""Write LaTeX summary tables from generated paper result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from fourier_smoothing.tables import write_latex_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("../2026-07-FourierSmoothing-Paper/results"))
    parser.add_argument("--tables-dir", type=Path, default=Path("../2026-07-FourierSmoothing-Paper/tex/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = write_latex_tables(args.results_dir, args.tables_dir)
    if not written:
        raise FileNotFoundError(f"No known result CSVs found in {args.results_dir}.")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
