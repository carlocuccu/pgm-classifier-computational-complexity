"""Choice of the copy number by cross-validation: `pgm select`.

The selection must depend on the training set only, and the advantage
conditions must be evaluated at every copy number with the rank of that copy
number.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pgm_complexity.selection import argmax_copy_number, select_copy_number


def test_argmax_takes_the_smallest_maximum():
    assert argmax_copy_number([0.5, 0.7, 0.6], [1, 2, 3]) == 2
    assert argmax_copy_number([0.7, 0.6, 0.7, 0.7], [1, 2, 3, 4]) == 1


def test_one_standard_error_rule():
    copies = [1, 2, 3, 4]
    mean = [0.880, 0.910, 0.912, 0.913]
    # Within one SE of the maximum (0.913 - 0.002): c = 2 is not, c = 3 is.
    assert select_copy_number(mean, [0, 0, 0, 0.002], copies) == 3
    # A wider SE reaches further back.
    assert select_copy_number(mean, [0, 0, 0, 0.005], copies) == 2
    # With no spread it reduces to the argmax.
    assert select_copy_number(mean, [0, 0, 0, 0], copies) == 4
    # The SE is the one at the maximum, not at the candidate.
    assert select_copy_number(mean, [0.1, 0.1, 0.1, 0.0], copies) == 4


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A three-class dataset on disk, laid out like the real ones."""
    rng = np.random.default_rng(20261104)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1], [0.2, -0.8, 0.7]])
    X = np.vstack([c + 0.6 * rng.normal(size=(40, 3)) for c in centres])
    y = np.repeat([0.0, 1.0, 2.0], 40)
    return X, y


def _patch_loader(monkeypatch, X, y):
    from pgm_complexity.bench import io

    monkeypatch.setattr(io, "load_dataset", lambda name: (X.copy(), y.copy()))


@pytest.mark.torch
def test_selection_ignores_the_test_set(monkeypatch, toy):
    """Changing the test samples changes no CV accuracy and no selected c.

    The split depends on the labels only, so the test features can be replaced
    without moving any sample between the two parts.
    """
    pytest.importorskip("torch")
    from pgm_complexity.bench.estimators import get_classifiers
    from pgm_complexity.bench.io import stratified_split
    from pgm_complexity.selection import sweep_dataset

    X, y = toy
    factory = get_classifiers()["kpgm"]
    _patch_loader(monkeypatch, X, y)
    rows = sweep_dataset("toy", [1, 2, 3], 3, 2, 42, factory)

    _, te = stratified_split(y, 0.2, 42)
    X_bad = X.copy()
    X_bad[te] = -X_bad[te]
    _patch_loader(monkeypatch, X_bad, y)
    rows_bad = sweep_dataset("toy", [1, 2, 3], 3, 2, 42, factory)

    assert [r["cv_mean"] for r in rows] == [r["cv_mean"] for r in rows_bad]
    assert [r["selected"] for r in rows] == [r["selected"] for r in rows_bad]
    assert [r["cv_argmax"] for r in rows] == [r["cv_argmax"] for r in rows_bad]
    assert [r["test_acc"] for r in rows] != [r["test_acc"] for r in rows_bad]


@pytest.mark.torch
def test_flags_use_the_rank_of_each_copy_number(monkeypatch, toy):
    pytest.importorskip("torch")
    from pgm_complexity.bench.estimators import get_classifiers
    from pgm_complexity.selection import sweep_dataset
    from pgm_complexity.thresholds import thresholds

    X, y = toy
    _patch_loader(monkeypatch, X, y)
    rows = sweep_dataset("toy", [1, 2, 3, 4], 3, 1, 42, get_classifiers()["kpgm"])

    assert sum(r["selected"] for r in rows) == 1
    assert sum(r["cv_argmax"] for r in rows) == 1
    c_sel = next(r["c"] for r in rows if r["selected"])
    c_max = next(r["c"] for r in rows if r["cv_argmax"])
    assert c_sel <= c_max
    for r in rows:
        assert r["r_Gc"] <= min(r["N"], r["dsym"])
        thr = thresholds(r["N"], r["d_enc"] - 1, r["c"], r["l"], r_g=r["r_Gc"])
        assert r["pred_thr"] == thr["pred_thr"]
        assert r["win_pred"] is (r["N"] > thr["pred_thr"])
        assert r["win_tr_time"] is (r["N"] > thr["tr_time_thr"])
        assert r["win_tr_mem"] is (r["N"] > thr["tr_mem_thr"])


@pytest.mark.torch
def test_command_writes_the_tables(monkeypatch, toy, tmp_path):
    pytest.importorskip("torch")
    from pgm_complexity.cli import main

    X, y = toy
    _patch_loader(monkeypatch, X, y)
    rc = main(
        ["select", "--datasets", "iris", "--copies", "1", "2",
         "--folds", "3", "--repeats", "2", "--outdir", str(tmp_path)]
    )  # fmt: skip
    assert rc == 0

    with open(tmp_path / "selection_sweep.csv") as fh:
        sweep = list(csv.DictReader(fh))
    with open(tmp_path / "selection.csv") as fh:
        chosen = list(csv.DictReader(fh))
    meta = json.loads((tmp_path / "selection_meta.json").read_text())

    assert [int(r["c"]) for r in sweep] == [1, 2]
    assert len(chosen) == 1 and chosen[0]["selected"] == "True"
    assert meta["folds"] == 3 and meta["repeats"] == 2
    assert meta["cv_random_states"] == [0, 1]
    assert meta["env"]["base_tol"] == 1e-10


def _fake_sweep():
    """Two datasets, three copy numbers, in the layout `pgm select` writes."""
    rows = []
    for name, sel, n in (("b_set", 2, 100), ("a_set", 1, 50)):
        for c in (1, 2, 3):
            rows.append(
                {
                    "dataset": name, "N": str(n), "d_enc": "4", "c": str(c),
                    "l": "2", "dsym": str(3 * c), "r_Gc": str(2 * c),
                    "cv_mean": f"{0.7 + 0.01 * c}", "cv_std": "0.002",
                    "cv_se": "0.0006", "test_acc": "0.75",
                    "win_tr_time": str(c < 3), "win_tr_mem": str(c < 2),
                    "win_pred": str(c < 3), "selected": str(c == sel),
                    "cv_argmax": str(c == 3),
                }
            )  # fmt: skip
    return rows


def test_table7_rows_take_the_selected_copy_number():
    from pgm_complexity.paper import table7_rows

    lines = table7_rows(_fake_sweep()).splitlines()
    # One row per dataset, in alphabetical order, at the selected c.
    assert [line.split(" & ")[0] for line in lines] == ["a set", "b set"]
    assert lines[0].split(" & ")[3] == "1"
    assert lines[1].split(" & ")[3] == "2"
    assert "0.720 $\\pm$ 0.002" in lines[1]
    assert lines[1].endswith("True & False & True \\\\")


def test_sweep_rows_mark_the_selected_copy_number():
    from pgm_complexity.paper import sweep_accuracy_rows, sweep_flag_rows

    acc = sweep_accuracy_rows(_fake_sweep()).splitlines()
    assert acc[1] == "b set & 0.710 & \\textbf{0.720} & 0.730 \\\\"
    flags = sweep_flag_rows(_fake_sweep()).splitlines()
    assert flags[0] == "a set & \\textbf{TTT} & TFT & FFF \\\\"


def test_paper_tables_command(tmp_path, capsys):
    from pgm_complexity.cli import main

    rows = _fake_sweep()
    with open(tmp_path / "selection_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "selection_meta.json").write_text(
        json.dumps(
            {
                "env": {"base_tol": 1e-10},
                "folds": 5,
                "repeats": 10,
                "split_seed": 42,
                "copies": [1, 2, 3],
                "rule": "one-standard-error",
            }
        )  # fmt: skip
    )
    assert main(["paper", "tables", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "% Table 7" in out and "\\textbf{TTT}" in out


def test_paper_tables_without_a_run(tmp_path):
    from pgm_complexity.cli import main

    with pytest.raises(SystemExit, match="pgm select"):
        main(["paper", "tables", "--results", str(tmp_path)])
