"""Complexity analysis and benchmarks of three quantum-inspired PGM classifiers.

The package is deliberately import-light at the top level: nothing here pulls
in numpy, and nothing pulls in torch. The benchmark harness pins the BLAS and
OpenMP thread count into the environment *before* numpy is imported, which only
works if importing this package does not import numpy first.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

# The repository root, four levels up from this file:
#   <root>/src/pgm_complexity/__init__.py
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "results_benchmark"

__all__ = ["DATA_DIR", "OUT_DIR", "ROOT", "__version__"]
