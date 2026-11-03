"""The POVM diagnostics: the residual function and the `pgm diag povm` command."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.torch

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261103)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1], [0.2, -0.8, 0.7]])
    X = np.vstack([c + 0.35 * rng.normal(size=(30, 3)) for c in centres])
    y = np.repeat([0, 1, 2], 30)
    return X, y


def _rcpgm(**kw):
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank import (
        PGMHQC_gpu_cpu_dtype as RcPGM,
    )

    return RcPGM(encoding="amplit", **kw)


@pytest.mark.parametrize("c", [1, 2, 3])
def test_residuals_of_a_fitted_model(data, c):
    from pgm_complexity.diagnostics import povm_residuals

    X, y = data
    res = povm_residuals(_rcpgm(n_copies=c, tol=1e-2).fit(X, y))

    assert res.dim == math.comb(4 + c - 1, c)
    assert res.completeness_2 < 1e-8
    assert res.completeness_fro < 1e-7
    assert res.min_eig_F > -1e-10
    assert 0 < res.rank_sigma <= res.dim
    assert res.lam_min_kept > res.tol
    if res.rank_sigma < res.dim:
        assert res.lam_max_discarded <= res.tol


def test_band_counts_positive_eigenvalues_below_the_threshold(data):
    """With a coarse threshold some strictly positive eigenvalues are
    discarded; with a negligible one, none are."""
    from pgm_complexity.diagnostics import povm_residuals

    X, y = data
    coarse = povm_residuals(_rcpgm(n_copies=3, tol=1e-2).fit(X, y))
    fine = povm_residuals(_rcpgm(n_copies=3, tol=1e-14).fit(X, y))
    assert coarse.n_band > 0
    assert fine.n_band == 0
    assert coarse.rank_sigma + coarse.n_band <= fine.rank_sigma + fine.n_band


def test_command_writes_the_table(tmp_path, repo_root):
    if not (repo_root / "datasets" / "iris.csv").exists():
        pytest.skip("datasets/ not populated; run `pgm data fetch`")
    from pgm_complexity.cli import main

    rc = main(
        [
            "diag",
            "povm",
            "--datasets",
            "iris",
            "haberman",
            "--copies",
            "1",
            "2",
            "--outdir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    with open(tmp_path / "povm.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["dataset"], r["c"]) for r in rows] == [
        ("iris", "1"),
        ("iris", "2"),
        ("haberman", "1"),
        ("haberman", "2"),
    ]
    for r in rows:
        assert float(r["completeness_2"]) < 1e-8
        assert float(r["min_eig_F"]) > -1e-10
        assert int(r["N"]) in (120, 244)

    meta = json.loads((tmp_path / "povm_meta.json").read_text())
    assert meta["copies"] == [1, 2] and meta["estimator"] == "rcpgm"


def test_command_skips_what_does_not_fit(tmp_path, repo_root):
    if not (repo_root / "datasets" / "car.csv").exists():
        pytest.skip("datasets/ not populated; run `pgm data fetch`")
    from pgm_complexity.cli import main

    main(
        [
            "diag",
            "povm",
            "--datasets",
            "car",
            "--copies",
            "9",
            "--mem-limit-gb",
            "0.001",
            "--outdir",
            str(tmp_path),
        ]
    )
    meta = json.loads((tmp_path / "povm_meta.json").read_text())
    assert meta["skipped"] == [{"dataset": "car", "c": 9, "dsym": 5005}]
    assert not (tmp_path / "povm.csv").exists()
