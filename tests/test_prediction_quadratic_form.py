"""Prediction of the Rc-PGM as a quadratic form.

The class score of a pure test state is Tr(F_i |z><z|) = <z|F_i|z>. The
Rc-PGM evaluates it as a quadratic form on the reduced vectors, without
building the m test densities of size d_sym x d_sym. The check here is
against the trace computed from explicit densities, as the estimator did up
to v1.0.1.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] - 0.4 * X[:, 2] > 0.1).astype(int) + (X[:, 1] > 0.8).astype(int)
    Xte = rng.normal(size=(25, 3))
    return X, y, Xte


def _rcpgm(**kw):
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank import (
        PGMHQC_gpu_cpu_dtype as RcPGM,
    )

    return RcPGM(encoding="amplit", **kw)


def _scores_from_densities(model, X, operators):
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank import (
        X_prime_func,
    )

    Z = model.map_batch_efficiently(X_prime_func(model, X, X.shape[0]))
    rho = torch.einsum("bi,bj->bij", Z, Z)
    return torch.stack(
        [torch.einsum("bij,ji->b", rho, op) for op in operators], dim=1
    ).numpy()


@pytest.mark.parametrize("c", [1, 2, 3])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("n_splits", [1, 3])
def test_pgm_scores_equal_the_trace_with_densities(data, c, class_weight, n_splits):
    X, y, Xte = data
    model = _rcpgm(
        n_copies=c, tol=1e-10, class_weight=class_weight, n_splits=n_splits
    ).fit(X, y)
    expected = _scores_from_densities(model, Xte, model.pgms_)
    np.testing.assert_allclose(
        model.predict_proba(Xte), expected, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("c", [1, 2])
def test_helstrom_scores_equal_the_trace_with_densities(data, c):
    X, y, Xte = data
    yb = (y > 0).astype(int)
    model = _rcpgm(n_copies=c, tol=1e-10, measure="hels").fit(X, yb)
    expected = _scores_from_densities(model, Xte, model.proj_sums_)
    np.testing.assert_allclose(
        model.predict_proba(Xte), expected, rtol=1e-12, atol=1e-14
    )
