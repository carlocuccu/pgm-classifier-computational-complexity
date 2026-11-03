"""The rank-aware realization of the k-PGM (solver='features').

Let Y (N x d_sym) be the feature matrix of the homogeneous polynomial kernel
of degree c, so that G^c = Y Y^T. Then K = W^{1/2} G^c W^{1/2} factors as
(W^{1/2} Y)(W^{1/2} Y)^T, and the thin SVD of W^{1/2} Y gives the nonzero
spectrum of K and its
eigenvectors without forming the N x N matrix, so the quadratic memory of the
dense k-PGM is a property of that implementation, not of the classifier.
The checks: same spectrum, same retained rank, same scores and labels as the
dense solver, and no N x N array on the fitted model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1], [0.2, -0.8, 0.7]])
    sizes = [60, 35, 25]
    X = np.vstack(
        [c + 0.5 * rng.normal(size=(n, 3)) for c, n in zip(centres, sizes, strict=True)]
    )
    y = np.repeat([0, 1, 2], sizes)
    Xte = rng.normal(size=(30, 3))
    return X, y, Xte


def _kpgm(**kw):
    from qunica.classifiers.KPGMC_Low_Rank import KPGM

    return KPGM(encoding="amplit", **kw)


def _scores(model, X):
    return model._scores(model.X_prime_func(X, X.shape[0]))


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_feature_matrix_gives_the_gram_matrix(data, c):
    X, y, _ = data
    m = _kpgm(n_copies=c).fit(X, y)
    Y = m._symmetric_features(m.X_prime_train)
    assert Y.shape == (len(X), math.comb(4 + c - 1, c))
    G = (m.X_prime_train @ m.X_prime_train.T) ** c
    np.testing.assert_allclose((Y @ Y.T).numpy(), G.numpy(), rtol=0, atol=1e-13)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_same_classifier_as_the_dense_solver(data, c, class_weight):
    X, y, Xte = data
    dense = _kpgm(n_copies=c, tol=1e-10, class_weight=class_weight).fit(X, y)
    feat = _kpgm(
        n_copies=c, tol=1e-10, class_weight=class_weight, solver="features"
    ).fit(X, y)

    nonzero = np.sort(dense.lam.numpy())[::-1][: feat.lam.shape[0]]
    np.testing.assert_allclose(
        np.sort(feat.lam.numpy())[::-1], nonzero, rtol=1e-9, atol=1e-14
    )
    assert feat.lam_inv_sqrt.shape[0] == dense.lam_inv_sqrt.shape[0]
    np.testing.assert_allclose(
        _scores(feat, Xte), _scores(dense, Xte), rtol=1e-8, atol=1e-12
    )
    np.testing.assert_array_equal(feat.predict(Xte), dense.predict(Xte))


@pytest.mark.parametrize("c", [2, 3])
def test_same_scores_as_the_rcpgm(data, c):
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank import (
        PGMHQC_gpu_cpu_dtype as RcPGM,
    )

    X, y, Xte = data
    feat = _kpgm(n_copies=c, tol=1e-12, solver="features").fit(X, y)
    rc = RcPGM(n_copies=c, encoding="amplit", tol=1e-12).fit(X, y)
    assert rc.rank_sigma_ == rc.pgms_.shape[-1], "full-rank sigma: no kernel term"
    np.testing.assert_allclose(
        _scores(feat, Xte), rc.predict_proba(Xte), rtol=1e-8, atol=1e-12
    )


def test_no_quadratic_array_is_stored(data):
    X, y, _ = data
    n = len(X)
    m = _kpgm(n_copies=2, solver="features").fit(X, y)
    for name, value in vars(m).items():
        if hasattr(value, "shape") and len(value.shape) == 2:
            assert value.shape[1] < n, f"{name} has shape {tuple(value.shape)}"


@pytest.mark.parametrize(("c", "expected"), [(2, "features"), (6, "dense")])
def test_auto_picks_the_smaller_side(data, c, expected):
    """d_enc = 4: d_sym = 10 at c = 2, below N = 120, so 'features'; d_sym = 84
    at c = 6, above the N = 60 of the subset, so 'dense'."""
    X, y, _ = data
    idx = np.r_[0:20, 60:80, 95:115] if expected == "dense" else np.arange(len(y))
    m = _kpgm(n_copies=c, solver="auto").fit(X[idx], y[idx])
    assert m.solver_ == expected


def test_unknown_solver_is_rejected(data):
    X, y, _ = data
    with pytest.raises(ValueError, match="solver"):
        _kpgm(n_copies=2, solver="lanczos").fit(X, y)
