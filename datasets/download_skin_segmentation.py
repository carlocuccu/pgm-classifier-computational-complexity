#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the Skin Segmentation dataset into this directory.

The dataset (245057 rows) is not redistributed here; it is downloaded on
demand from the UCI Machine Learning Repository and written as
`datasets/Skin_NonSkin.txt`, the tab-separated, header-less file that
`run_benchmarks.py B` expects.

    python datasets/download_skin_segmentation.py

Source
------
Rajen Bhatt and Abhinav Dhall, *Skin Segmentation*, UCI Machine Learning
Repository, 2012. https://doi.org/10.24432/C5T30C -- CC BY 4.0.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "Skin_NonSkin.txt"

URL = "https://archive.ics.uci.edu/static/public/229/skin+segmentation.zip"
MEMBER = "Skin_NonSkin.txt"
EXPECTED_ROWS = 245057

USER_AGENT = "pgm-complexity-repo/1.0 (dataset download script)"


def main() -> int:
    if TARGET.exists():
        rows = sum(1 for line in TARGET.open() if line.strip())
        print(f"{TARGET.name} already present ({rows} rows); nothing to do.")
        return 0 if rows == EXPECTED_ROWS else 1

    print(f"Downloading {URL} ...")
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        member = MEMBER if MEMBER in names else names[0]
        data = archive.read(member)

    TARGET.write_bytes(data)

    rows = sum(1 for line in TARGET.open() if line.strip())
    print(f"Wrote {TARGET} ({rows} rows).")
    if rows != EXPECTED_ROWS:
        print(f"Warning: expected {EXPECTED_ROWS} rows.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
