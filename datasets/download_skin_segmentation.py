#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the Skin Segmentation dataset into this directory.

The dataset (245057 rows) is not redistributed here; it is downloaded on
demand from the UCI Machine Learning Repository and written as
`datasets/Skin_NonSkin.txt`, the tab-separated, header-less file that
`run_benchmarks.py B` expects.

    python datasets/download_skin_segmentation.py

This is a shortcut for `python scripts/fetch_datasets.py --only-skin`, which
does the same work; `scripts/fetch_datasets.py` without options fetches this
file together with the eleven small datasets.

Source
------
Rajen Bhatt and Abhinav Dhall, *Skin Segmentation*, UCI Machine Learning
Repository, 2012. https://doi.org/10.24432/C5T30C -- CC BY 4.0.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

FETCH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_datasets.py"


def main() -> int:
    sys.argv = [str(FETCH), "--only-skin"]
    try:
        runpy.run_path(str(FETCH), run_name="__main__")
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
