"""Shared fixtures.

`pgm_complexity` is installed, so the modules are imported normally. The two
fixtures exist because most tests want the same handful of entry points and
because naming them keeps the tests readable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def harness():
    """The pieces of the benchmark harness the tests exercise.

    A namespace rather than a module: the harness is several modules now, and
    the tests care about what it does, not about where each function lives.
    """
    from types import SimpleNamespace

    from pgm_complexity import config
    from pgm_complexity.bench import estimators, io, measure, selftest
    from pgm_complexity.thresholds import dsym_of, thresholds

    return SimpleNamespace(
        BASE_TOL=config.BASE_TOL,
        HARMONIZE_TOL=config.HARMONIZE_TOL,
        dsym_of=dsym_of,
        thresholds=thresholds,
        load_dataset=io.load_dataset,
        stratified_split=io.stratified_split,
        stratified_subsample=io.stratified_subsample,
        env_info=io.env_info,
        model_bytes=measure.model_bytes,
        timed=measure.timed,
        get_classifiers=estimators.get_classifiers,
        selftest=selftest.selftest,
        _MockPGM=selftest._MockPGM,
    )


@pytest.fixture(scope="session")
def datasets_script():
    """The dataset specifications and the machinery that rebuilds them."""
    from types import SimpleNamespace

    from pgm_complexity.data import build
    from pgm_complexity.data.specs import CAR_TAIL, SKIN, SOURCES

    return SimpleNamespace(
        SOURCES=SOURCES,
        CAR_TAIL=CAR_TAIL,
        SKIN=SKIN,
        matches=build.matches,
        materialize=build.materialize,
        digests=build.digests,
    )


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
