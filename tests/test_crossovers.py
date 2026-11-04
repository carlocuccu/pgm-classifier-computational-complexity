"""Empirical crossovers of Component B, and the two components end to end.

The crossover is located from the data, not read off a figure: these tests fix
what it means on series whose answer is known, and run both components on a
small synthetic problem to check what they write.
"""

from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pgm_complexity.crossovers import (
    ABOVE,
    BELOW,
    CROSSED,
    crossover_table,
    empirical_crossover,
)

N = [100, 200, 400, 800]


def test_crossover_between_the_bracketing_sizes():
    # k-PGM ~ N^2 / 300, Rc-PGM ~ N: equal at N = 300, between 200 and 400.
    k = [n * n / 300 for n in N]
    r = list(N)
    value, status = empirical_crossover(N, k, r)
    assert status == CROSSED
    assert value == pytest.approx(300, rel=1e-9)


def test_crossover_before_and_beyond_the_grid():
    assert empirical_crossover(N, [2, 2, 2, 2], [1, 1, 1, 1]) == (None, BELOW)
    assert empirical_crossover(N, [1, 1, 1, 1], [2, 2, 2, 2]) == (None, ABOVE)


def test_the_last_crossing_is_the_one_that_counts():
    """Noise around the threshold can cross back; the crossover is where the
    k-PGM becomes the more expensive and stays so."""
    k = [0.5, 2.0, 0.9, 4.0]
    r = [1.0, 1.0, 1.0, 1.0]
    value, status = empirical_crossover(N, k, r)
    assert status == CROSSED
    assert 400 < value < 800


def test_zero_readings_do_not_break_the_logarithm():
    value, status = empirical_crossover(N, [0, 1, 4, 16], [2, 2, 2, 2])
    assert status == CROSSED and 200 < value < 400


def _row(method, n, c, **fields):
    base = {"method": method, "N": n, "c": c}
    base.update(fields)
    return base


def test_table_has_one_record_per_copy_number_and_quantity():
    rows = []
    for c in (2, 3):
        for n in N:
            cost_k = dict.fromkeys(("fit_mean", "fit_rss_peak", "pred_mean"), n * n)
            cost_r = dict.fromkeys(
                ("fit_mean", "fit_rss_peak", "pred_mean"), 40000.0 * c
            )
            rows.append(_row("kpgm", n, c, model_bytes=n, **cost_k))
            rows.append(_row("rcpgm", n, c, model_bytes=300, **cost_r))
            rows.append(_row("kpgm_features", n, c, model_bytes=n, **cost_r))
    thr = {
        c: {
            "dsym": 10 * c,
            "tr_time_thr": 20.0,
            "tr_mem_thr": 100.0 * c,
            "tr_mem_thr_class": 90.0 * c,
            "pred_thr": 250.0,
        }
        for c in (2, 3)
    }
    table = crossover_table(rows, thr)
    assert len(table) == 2 * 4
    time_c2 = next(t for t in table if t["c"] == 2 and t["quantity"] == "training time")
    assert time_c2["empirical"] == pytest.approx(math.sqrt(80000), rel=1e-9)
    assert time_c2["ratio"] == pytest.approx(math.sqrt(80000) / 20.0)
    mem = [t for t in table if t["quantity"].startswith("training memory")]
    assert all("theoretical_class" in t for t in mem)
    model_c3 = next(t for t in table if t["c"] == 3 and t["quantity"] == "stored model")
    assert model_c3["empirical"] == pytest.approx(300, rel=1e-9)


# ---------------------------------------------------------------------------
# The two components on a small synthetic problem
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic(monkeypatch):
    """A two-class dataset in place of every real one."""
    from pgm_complexity.bench import components

    rng = np.random.default_rng(20261104)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1]])
    X = np.vstack([c + 0.5 * rng.normal(size=(150, 3)) for c in centres])
    y = np.repeat([1.0, 2.0], 150)
    monkeypatch.setattr(components, "load_dataset", lambda name: (X.copy(), y.copy()))
    monkeypatch.setattr(components, "SKIN_TEST", 60)
    return X, y


@pytest.mark.torch
def test_component_b_sweeps_every_copy_number_with_three_classifiers(
    synthetic, tmp_path
):
    pytest.importorskip("torch")
    from pgm_complexity.bench.components import component_B

    args = SimpleNamespace(
        seed=1, reps=1, nmax=10_000, sweep=[40, 80, 160], copies=[2, 3],
        outdir=str(tmp_path),
    )  # fmt: skip
    component_B(args)

    with open(tmp_path / "componentB.csv") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads((tmp_path / "componentB_meta.json").read_text())
    with open(tmp_path / "componentB_crossovers.csv") as fh:
        table = list(csv.DictReader(fh))

    assert {(r["method"], r["c"]) for r in rows} == {
        (m, c) for m in ("kpgm", "kpgm_features", "rcpgm") for c in ("2", "3")
    }
    assert meta["copies"] == [2, 3] and set(meta["thresholds"]) == {"2", "3"}
    assert len(table) == 2 * 4
    # The three classifiers are the same classifier.
    for c in ("2", "3"):
        for n in {r["N"] for r in rows}:
            acc = {r["accuracy"] for r in rows if r["c"] == c and r["N"] == n}
            assert len(acc) == 1
    for c in (2, 3):
        assert (tmp_path / f"componentB_crossover_c{c}.png").exists()


@pytest.mark.torch
def test_component_a_reads_the_copy_numbers_of_pgm_select(synthetic, tmp_path):
    pytest.importorskip("torch")
    from pgm_complexity.bench.components import component_A

    selection = tmp_path / "selection.csv"
    selection.write_text("dataset,c,selected\niris,2,True\nhaberman,1,True\n")
    args = SimpleNamespace(
        include_cpgm=False, datasets=["iris", "haberman"], seed=42, reps=1,
        mem_limit_gb=8.0, cpgm_dim_limit=4100, outdir=str(tmp_path / "out"),
        selection=str(selection),
    )  # fmt: skip
    component_A(args)

    with open(tmp_path / "out" / "componentA.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {(r["dataset"], r["c"]) for r in rows} == {("iris", "2"), ("haberman", "1")}
    assert {r["method"] for r in rows} == {"kpgm", "rcpgm"}


def test_component_a_needs_a_selection(tmp_path):
    from pgm_complexity.bench.components import component_A

    args = SimpleNamespace(
        include_cpgm=False, datasets=["iris"], outdir=str(tmp_path),
        selection=str(tmp_path / "missing.csv"),
    )  # fmt: skip
    with pytest.raises(SystemExit, match="pgm select"):
        component_A(args)
