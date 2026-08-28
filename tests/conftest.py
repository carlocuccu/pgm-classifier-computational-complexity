"""Shared fixtures.

The harness and the dataset script live at the top of the repository rather
than in an installed package, so they are loaded here by path. When the code
becomes a package this file shrinks to a plain import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, relative: str):
    """Import a top-level script as a module, without polluting sys.argv."""
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def harness():
    """`run_benchmarks.py`, imported as a module."""
    return _load("run_benchmarks", "run_benchmarks.py")


@pytest.fixture(scope="session")
def datasets_script():
    """`scripts/fetch_datasets.py`, imported as a module."""
    return _load("fetch_datasets", "scripts/fetch_datasets.py")


def pytest_collection_modifyitems(config, items):
    """Skip what the current environment cannot run.

    `torch` marks the tests that need PyTorch and the qunica estimators; they
    are the ones that check the published implementations rather than the
    analysis around them. `network` marks the tests that download from PMLB,
    OpenML or UCI, which are opt-in via `-m network`.
    """
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False

    skip_torch = pytest.mark.skip(reason="PyTorch is not installed")
    skip_network = pytest.mark.skip(reason="needs network; run with -m network")
    selected = config.getoption("-m", default="")

    for item in items:
        if "torch" in item.keywords and not has_torch:
            item.add_marker(skip_torch)
        if "network" in item.keywords and "network" not in selected:
            item.add_marker(skip_network)
