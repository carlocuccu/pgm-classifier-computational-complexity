"""The runs deposited with the code: results_benchmark/, results_selection/
and results_diagnostics/.

These are the numbers Tables 7 and 8, Figure 4 and the appendices report. The
files are committed on purpose, so they can be checked: not the timings, which
belong to a machine, but everything that does not depend on one -- the
equivalence of the classifiers, the copy numbers and where they came from, the
thresholds recorded beside each run, the sweep grid, the crossovers derived
from the measurements, the residuals of the measurement, and the protocol the
runs declare they followed.
"""

from __future__ import annotations

import csv
import json

import pytest

SKIN_COPIES = [2, 3, 4, 5, 6, 7]
SKIN_GRID = [25, 50, 100, 250, 500, 1000, 1750, 3000, 5000, 8000]
TABLE8 = ["balance-scale", "haberman", "iris", "led7", "ecoli", "car"]


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def results(repo_root):
    bench = repo_root / "results_benchmark"
    if not (bench / "componentB.csv").exists():
        pytest.skip("no deposited run in results_benchmark/")
    selection = repo_root / "results_selection"
    diagnostics = repo_root / "results_diagnostics"
    return {
        "dir": bench,
        "a": _rows(bench / "componentA.csv"),
        "b": _rows(bench / "componentB.csv"),
        "crossovers": _rows(bench / "componentB_crossovers.csv"),
        "a_meta": json.loads((bench / "componentA_meta.json").read_text()),
        "b_meta": json.loads((bench / "componentB_meta.json").read_text()),
        "selection": _rows(selection / "selection.csv"),
        "sweep": _rows(selection / "selection_sweep.csv"),
        "s_meta": json.loads((selection / "selection_meta.json").read_text()),
        "povm": _rows(diagnostics / "povm.csv"),
        "p_meta": json.loads((diagnostics / "povm_meta.json").read_text()),
    }


def _n(row) -> int:
    return int(float(row["N"]))


def by_method(rows, method, c=None):
    out = [r for r in rows if r["method"] == method]
    if c is not None:
        out = [r for r in out if int(float(r["c"])) == c]
    return sorted(out, key=_n)


# --------------------------------------------------------------------------
# Table 7 and the copy-number sweep (pgm select)
# --------------------------------------------------------------------------
def test_one_selected_copy_number_per_dataset(results):
    selected = [r for r in results["sweep"] if r["selected"] == "True"]
    assert len(selected) == len({r["dataset"] for r in results["sweep"]}) == 11
    assert {r["dataset"]: r["c"] for r in selected} == {
        r["dataset"]: r["c"] for r in results["selection"]
    }


def test_the_selection_follows_the_one_standard_error_rule(results):
    from pgm_complexity.selection import select_copy_number

    datasets = {r["dataset"] for r in results["sweep"]}
    for name in datasets:
        rows = sorted(
            (r for r in results["sweep"] if r["dataset"] == name),
            key=lambda r: int(r["c"]),
        )
        copies = [int(r["c"]) for r in rows]
        chosen = select_copy_number(
            [float(r["cv_mean"]) for r in rows],
            [float(r["cv_se"]) for r in rows],
            copies,
        )
        assert chosen == next(int(r["c"]) for r in rows if r["selected"] == "True")


def test_the_flags_of_the_sweep_follow_from_its_columns(results, harness):
    for r in results["sweep"]:
        thr = harness.thresholds(
            N=int(r["N"]),
            d_raw=int(r["d_enc"]) - 1,
            c=int(r["c"]),
            l=int(r["l"]),
            r_g=int(r["r_Gc"]),
        )
        assert int(r["dsym"]) == thr["dsym"]
        assert int(r["r_Gc"]) <= min(int(r["N"]), int(r["dsym"]))
        assert (r["win_tr_time"] == "True") is (int(r["N"]) > thr["tr_time_thr"])
        assert (r["win_tr_mem"] == "True") is (int(r["N"]) > thr["tr_mem_thr"])
        assert (r["win_pred"] == "True") is (int(r["N"]) > thr["pred_thr"])


# --------------------------------------------------------------------------
# Component A (Table 8)
# --------------------------------------------------------------------------
def test_component_a_ran_at_the_selected_copy_numbers(results):
    chosen = {r["dataset"]: r for r in results["selection"]}
    measured = {r["dataset"]: r for r in by_method(results["a"], "kpgm")}
    assert set(measured) == set(TABLE8)
    assert results["a_meta"]["copy_numbers_from"] == "results_selection/selection.csv"
    for name, row in measured.items():
        assert int(float(row["c"])) == int(chosen[name]["c"]), name
        assert _n(row) == int(chosen[name]["N"]), name
        # Same split, same model: the test accuracy of Table 7.
        assert float(row["accuracy"]) == pytest.approx(
            float(chosen[name]["test_acc"]), abs=1e-12
        ), name


def test_component_a_classifiers_agree(results):
    k = by_method(results["a"], "kpgm")
    r = by_method(results["a"], "rcpgm")
    assert [x["dataset"] for x in k] == [x["dataset"] for x in r]
    for a, b in zip(k, r, strict=True):
        assert a["accuracy"] == b["accuracy"], a["dataset"]
    for m in results["a_meta"]["meta"]:
        assert m["argmax_agreement"] == 1.0, m["dataset"]


# --------------------------------------------------------------------------
# Component B (Figure 4 and the crossover table)
# --------------------------------------------------------------------------
def test_component_b_swept_the_recorded_grid(results):
    meta = results["b_meta"]
    assert meta["copies"] == SKIN_COPIES
    assert meta["grid"] == SKIN_GRID
    assert meta["test_size"] == 2000
    assert meta["methods"] == ["kpgm", "kpgm_features", "rcpgm"]
    for method in meta["methods"]:
        for c in SKIN_COPIES:
            assert [_n(r) for r in by_method(results["b"], method, c)] == SKIN_GRID


def test_component_b_thresholds_match_the_recorded_configuration(results, harness):
    meta = results["b_meta"]
    for c in SKIN_COPIES:
        row = by_method(results["b"], "kpgm", c)[-1]
        recorded = meta["thresholds"][str(c)]
        recomputed = harness.thresholds(
            N=_n(row),
            d_raw=int(float(row["d"])),
            c=c,
            l=int(float(row["l"])),
            r_g=meta["ranks"][str(c)][str(_n(row))],
        )
        for key in ("dsym", "tr_time_thr", "tr_mem_thr", "pred_thr"):
            assert recomputed[key] == pytest.approx(recorded[key]), (c, key)


def test_the_classifiers_agree_where_n_exceeds_dsym(results, harness):
    """The three realizations are one classifier. Where N < d_sym the spectrum
    has eigenvalues at the threshold and a few test labels may differ; the
    deposited run has two such cells (N = 25 at c = 6 and 7)."""
    differing = []
    for c in SKIN_COPIES:
        ds = harness.dsym_of(3, c)
        for n in SKIN_GRID:
            acc = {
                r["accuracy"]
                for r in results["b"]
                if int(float(r["c"])) == c and _n(r) == n
            }
            if len(acc) > 1:
                differing.append((c, n))
                assert n < ds, (c, n)
    assert differing == [(6, 25), (7, 25)]


def test_the_crossover_table_follows_from_the_measurements(results):
    from pgm_complexity.crossovers import crossover_table

    infos = {int(c): v for c, v in results["b_meta"]["thresholds"].items()}
    table = crossover_table(results["b"], infos)
    deposited = results["crossovers"]
    assert len(table) == len(deposited) == 4 * len(SKIN_COPIES)
    for new, old in zip(table, deposited, strict=True):
        assert str(new["c"]) == old["c"] and new["quantity"] == old["quantity"]
        assert new["status"] == old["status"]
        if new["empirical"] is not None:
            assert new["empirical"] == pytest.approx(float(old["empirical"]))


@pytest.mark.parametrize("c", SKIN_COPIES)
def test_the_rcpgm_model_is_the_same_size_at_every_n(results, c):
    """Its stored model depends on d_sym and l, not on N."""
    sizes = {int(r["model_bytes"]) for r in by_method(results["b"], "rcpgm", c)}
    assert len(sizes) == 1


@pytest.mark.parametrize("c", SKIN_COPIES)
def test_the_kpgm_model_grows_with_n(results, c):
    for method in ("kpgm", "kpgm_features"):
        sizes = [int(r["model_bytes"]) for r in by_method(results["b"], method, c)]
        assert sizes == sorted(sizes)
        assert sizes[-1] > 100 * sizes[0]


@pytest.mark.parametrize("c", SKIN_COPIES)
def test_the_figure_is_deposited_next_to_the_numbers(results, c):
    for ext in ("png", "pdf"):
        assert (results["dir"] / f"componentB_crossover_c{c}.{ext}").exists()


# --------------------------------------------------------------------------
# Residuals of the measurement (pgm diag povm)
# --------------------------------------------------------------------------
def test_the_measurement_is_complete_and_positive(results):
    rows = results["povm"]
    assert len(rows) == 11 * 9 and results["p_meta"]["skipped"] == []
    for r in rows:
        assert float(r["completeness_2"]) < 1e-5, (r["dataset"], r["c"])
        assert float(r["min_eig_F"]) > -1e-8, (r["dataset"], r["c"])
        assert int(r["rank_sigma"]) <= int(r["dim"])


# --------------------------------------------------------------------------
# The protocol the runs declare
# --------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["a_meta", "b_meta", "s_meta", "p_meta"])
def test_the_run_declares_the_measurement_protocol(results, which, harness):
    env = results[which]["env"]
    assert env["threads"] == 1, "the reported timings are single-threaded"
    assert env["base_tol"] == harness.BASE_TOL
    assert "harmonize_tol" not in env
    assert env["torch"] is not None, "a deposited run must have run the estimators"


def test_no_deposited_file_records_the_machine_path(results):
    for directory in ("results_benchmark", "results_selection", "results_diagnostics"):
        for path in (results["dir"].parent / directory).iterdir():
            if path.suffix in (".json", ".csv", ".tex"):
                assert "/home/" not in path.read_text(), path
