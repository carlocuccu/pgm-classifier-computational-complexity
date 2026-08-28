"""The measurements deposited in results_benchmark/.

These are the numbers Table 8 and Figure 4 report. The files are committed on
purpose, so they can be checked: not the timings, which belong to a machine,
but everything that does not depend on one -- the equivalence of the two
classifiers, the thresholds recorded alongside the run, the sweep grid, and the
protocol the run declares it followed.
"""

from __future__ import annotations

import csv
import json

import pytest

TABLE7_C = {"skin": 5}


@pytest.fixture(scope="module")
def results(repo_root):
    directory = repo_root / "results_benchmark"
    if not (directory / "componentB.csv").exists():
        pytest.skip("no deposited run in results_benchmark/")

    def rows(name):
        with (directory / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    return {
        "dir": directory,
        "a": rows("componentA.csv"),
        "b": rows("componentB.csv"),
        "a_meta": json.loads((directory / "componentA_meta.json").read_text()),
        "b_meta": json.loads((directory / "componentB_meta.json").read_text()),
    }


def by_method(rows, method):
    return sorted(
        (r for r in rows if r["method"] == method), key=lambda r: int(float(r["N"]))
    )


# --------------------------------------------------------------------------
# What must hold regardless of the machine
# --------------------------------------------------------------------------
def test_the_two_classifiers_agree_on_every_accuracy(results):
    """k-PGM and Rc-PGM are the same classifier; only their cost differs.

    This is the claim the whole comparison rests on, and it is the one thing in
    the deposited files that a different machine cannot change.
    """
    for rows in (results["a"], results["b"]):
        k = by_method(rows, "kpgm")
        r = by_method(rows, "rcpgm")
        assert [x["dataset"] for x in k] == [x["dataset"] for x in r]
        for a, b in zip(k, r, strict=True):
            assert a["accuracy"] == b["accuracy"], f"{a['dataset']}: accuracies differ"


def test_component_a_reproduces_the_table7_accuracies(results):
    """Component A fits at the tabulated copy number, on the tabulated split."""
    expected = {
        "balance-scale": (500, 1, 0.896),
        "haberman": (244, 1, 0.742),
        "iris": (120, 2, 0.967),
        "led7": (2560, 4, 0.748),
        "ecoli": (261, 8, 0.909),
        "car": (1381, 9, 0.824),
    }
    measured = {r["dataset"]: r for r in by_method(results["a"], "kpgm")}
    assert set(measured) == set(expected), "Component A ran on a different selection"

    for name, (n, c, accuracy) in expected.items():
        row = measured[name]
        assert int(float(row["N"])) == n, name
        assert int(float(row["c"])) == c, name
        assert round(float(row["accuracy"]), 3) == accuracy, name


def test_component_b_thresholds_match_the_recorded_configuration(results, harness):
    """The thresholds stored beside the run are the ones its parameters imply."""
    meta = results["b_meta"]
    rows = by_method(results["b"], "kpgm")
    d_raw = int(float(rows[0]["d"]))
    c = int(float(rows[0]["c"]))
    l = int(float(rows[0]["l"]))

    assert c == TABLE7_C["skin"]
    n_max = int(float(rows[-1]["N"]))
    recomputed = harness.thresholds(N=n_max, d_raw=d_raw, c=c, l=l)
    assert recomputed["dsym"] == meta["thresholds"]["dsym"] == 56
    assert recomputed["tr_time_thr"] == pytest.approx(meta["thresholds"]["tr_time_thr"])
    assert recomputed["tr_mem_thr"] == meta["thresholds"]["tr_mem_thr"] == 3136


def test_component_b_swept_the_recorded_grid(results):
    meta = results["b_meta"]
    grid = sorted(meta["grid"])
    assert grid == [250, 500, 1000, 1750, 3000, 5000, 8000]
    for method in ("kpgm", "rcpgm"):
        assert [int(float(r["N"])) for r in by_method(results["b"], method)] == grid
    assert meta["test_size"] == 2000


def test_the_rcpgm_model_is_the_same_size_at_every_n(results):
    """Its stored model depends on d_sym and l, not on N: the paper's point."""
    sizes = {int(r["model_bytes"]) for r in by_method(results["b"], "rcpgm")}
    assert len(sizes) == 1, f"the Rc-PGM model should not grow with N, saw {sizes}"


def test_the_kpgm_model_grows_with_n(results):
    sizes = [int(r["model_bytes"]) for r in by_method(results["b"], "kpgm")]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 20 * sizes[0], "the k-PGM model should grow roughly with N"


# --------------------------------------------------------------------------
# The protocol the runs declare
# --------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["a_meta", "b_meta"])
def test_the_run_declares_the_measurement_protocol(results, which, harness):
    env = results[which]["env"]
    assert env["threads"] == 1, "the reported timings are single-threaded"
    assert env["base_tol"] == harness.BASE_TOL
    assert env["harmonize_tol"] is True, (
        "the truncation thresholds of the two estimators must be on the same scale"
    )
    assert env["torch"] is not None, "a deposited run must have run the estimators"


def test_the_figure_is_deposited_next_to_the_numbers(results):
    for name in ("componentB_crossover.png", "componentB_crossover.pdf"):
        assert (results["dir"] / name).exists(), name
