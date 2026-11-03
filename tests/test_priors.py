"""Class priors in the k-PGM, and its equivalence with the primal estimators.

The k-PGM realizes the PGM with priors p_j by weighting every training sample
of class j with w_i = p_j / #k_j: sigma = Y^T W Y, whose dual is the weighted
Gram matrix K = W^{1/2} G^c W^{1/2}, with the same nonzero spectrum. With the
same threshold `tol`, the k-PGM and the Rc-PGM therefore discard the same
eigendirections and compute the same class scores, for any priors. Without
weights (the k-PGM up to v1.0.1) only the empirical priors p_j = #k_j / N are
reproduced, up to the scale factor N of the spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def imbalanced():
    """Three overlapping, strongly imbalanced classes: the priors matter."""
    rng = np.random.default_rng(20261103)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1], [0.2, -0.8, 0.7]])
    sizes = [70, 20, 8]
    X = np.vstack(
        [c + 0.6 * rng.normal(size=(n, 3)) for c, n in zip(centres, sizes, strict=True)]
    )
    y = np.repeat([0, 1, 2], sizes)
    Xte = np.vstack([c + 0.6 * rng.normal(size=(40, 3)) for c in centres])
    return X, y, Xte


def _kpgm(**kw):
    from qunica.classifiers.KPGMC_Low_Rank import KPGM

    return KPGM(encoding="amplit", **kw)


def _rcpgm(**kw):
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank import (
        PGMHQC_gpu_cpu_dtype as RcPGM,
    )

    return RcPGM(encoding="amplit", **kw)


def _kpgm_scores(model, X):
    return model._scores(model.X_prime_func(X, X.shape[0]))


@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("c", [1, 2])
def test_scores_coincide_with_the_rcpgm(imbalanced, class_weight, c):
    """Full-rank sigma (no kernel term): Tr(F_i rho_z) of the Rc-PGM equals
    the k-PGM score of class i, sample by sample."""
    X, y, Xte = imbalanced
    tol = 1e-12
    k = _kpgm(n_copies=c, tol=tol, class_weight=class_weight).fit(X, y)
    r = _rcpgm(n_copies=c, tol=tol, class_weight=class_weight).fit(X, y)
    assert r.rank_sigma_ == r.pgms_.shape[-1], "sigma must be full rank here"

    np.testing.assert_allclose(
        _kpgm_scores(k, Xte), r.predict_proba(Xte), rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("c", [2, 3, 4])
@pytest.mark.parametrize("tol", [1e-8, 1e-4])
def test_same_threshold_same_rank_same_labels(imbalanced, class_weight, c, tol):
    """With a shared threshold the two truncate the same spectrum."""
    X, y, Xte = imbalanced
    k = _kpgm(n_copies=c, tol=tol, class_weight=class_weight).fit(X, y)
    r = _rcpgm(n_copies=c, tol=tol, class_weight=class_weight).fit(X, y)

    assert k.lam_inv_sqrt.shape[0] == r.rank_sigma_
    np.testing.assert_array_equal(k.predict(Xte), r.predict(Xte))


def test_priors_change_the_classifier(imbalanced):
    """Guard on the fixture: empirical and uniform priors are different PGMs
    on these data, so the tests above do discriminate between them."""
    X, y, Xte = imbalanced
    emp = _kpgm(n_copies=2, tol=1e-10).fit(X, y).predict(Xte)
    uni = _kpgm(n_copies=2, tol=1e-10, class_weight="balanced").fit(X, y).predict(Xte)
    assert np.any(emp != uni)


def test_empirical_priors_are_the_unweighted_gram_over_n(imbalanced):
    """class_weight=None gives K = G^c / N: the spectrum of the unweighted
    Gram matrix divided by N, and the labels it gives with threshold N * tol."""
    X, y, Xte = imbalanced
    N, c = len(y), 3
    k = _kpgm(n_copies=c, tol=1e-6 / N).fit(X, y)

    Xp = k.X_prime_train
    G = (Xp @ Xp.T) ** c
    lam_G = torch.linalg.eigvalsh(G.double())
    np.testing.assert_allclose(
        np.sort(k.lam.numpy()), np.sort(lam_G.numpy()) / N, atol=1e-14
    )
    assert k.lam_inv_sqrt.shape[0] == int((lam_G > 1e-6).sum())
    np.testing.assert_allclose(k.priors_, np.bincount(y) / N)


def test_dict_priors_are_normalized_and_validated(imbalanced):
    X, y, Xte = imbalanced
    a = _kpgm(n_copies=2, tol=1e-10, class_weight={0: 1, 1: 1, 2: 1}).fit(X, y)
    b = _kpgm(n_copies=2, tol=1e-10, class_weight="balanced").fit(X, y)
    np.testing.assert_allclose(a.priors_, b.priors_)
    np.testing.assert_array_equal(a.predict(Xte), b.predict(Xte))

    with pytest.raises(ValueError, match="no prior"):
        _kpgm(n_copies=2, class_weight={0: 1, 1: 1}).fit(X, y)
    with pytest.raises(ValueError, match="nonnegative"):
        _kpgm(n_copies=2, class_weight={0: -1, 1: 1, 2: 1}).fit(X, y)
    with pytest.raises(ValueError, match="class_weight"):
        _kpgm(n_copies=2, class_weight="uniform").fit(X, y)
