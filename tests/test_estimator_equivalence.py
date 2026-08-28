"""The published estimators, checked against each other and against theory.

Everything here needs PyTorch and the qunica package, so it is skipped where
they are absent and is the part of the suite that has to run on the machine
that reproduces the measurements. It is also the only place where the claim
that makes the whole paper coherent -- that the k-PGM and the Rc-PGM are the
same classifier -- is exercised end to end rather than argued.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def factories(harness):
    return harness.get_classifiers()


@pytest.fixture(scope="module")
def data():
    """A small, well-conditioned three-class problem."""
    rng = np.random.default_rng(20260828)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1], [0.2, -0.8, 0.7]])
    X = np.vstack([c + 0.35 * rng.normal(size=(40, 3)) for c in centres])
    y = np.repeat([0, 1, 2], 40)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    return X[:90], y[:90], X[90:], y[90:]


@pytest.mark.parametrize("c", [1, 2, 3])
def test_kpgm_and_rcpgm_predict_the_same_labels(harness, factories, data, c):
    """The equivalence of Theorem 1, on every test sample.

    The two estimators truncate different matrices -- G^c and sigma = G^c / N --
    whose spectra differ by the factor N, so the Rc-PGM is given the rescaled
    threshold the harness uses. Without that rescaling the two would discard
    different eigendirections and could disagree near the truncation.
    """
    X_train, y_train, X_test, _ = data

    kpgm = factories["kpgm"](c, n_train=len(X_train)).fit(X_train, y_train)
    rcpgm = factories["rcpgm"](c, n_train=len(X_train)).fit(X_train, y_train)

    np.testing.assert_array_equal(kpgm.predict(X_test), rcpgm.predict(X_test))


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_the_retained_rank_respects_its_bound(harness, factories, data, c):
    """r_{G^c} <= min(N, d_sym), the consistency bound the Note of Table 7 states."""
    X_train, y_train, _, _ = data
    model = factories["kpgm"](c, n_train=len(X_train)).fit(X_train, y_train)

    r_gc = model.lam_inv_sqrt.shape[0]
    d_sym = math.comb(X_train.shape[1] + 1 + c - 1, c)

    assert r_gc <= min(len(X_train), d_sym)


def test_the_retained_rank_is_that_of_the_power_gram_matrix(factories, data):
    """It is rank(G^c), not rank(G): the two differ as soon as c > 1.

    With three raw features the encoded dimension is 4, so rank(G) <= 4 while
    rank(G^c) can reach d_sym = C(4 + c - 1, c). Observing a rank above 4 is
    what settles the reading of the symbol.
    """
    X_train, y_train, _, _ = data
    r_by_c = {
        c: factories["kpgm"](c, n_train=len(X_train))
        .fit(X_train, y_train)
        .lam_inv_sqrt.shape[0]
        for c in (1, 3)
    }

    assert r_by_c[1] <= X_train.shape[1] + 1
    assert r_by_c[3] > X_train.shape[1] + 1


def test_the_rcpgm_model_does_not_grow_with_the_training_set(harness, factories, data):
    """Its stored model is O(l d_sym^2), independent of N -- the paper's claim."""
    X_train, y_train, _, _ = data

    sizes = []
    for n in (40, 90):
        model = factories["rcpgm"](2, n_train=n).fit(X_train[:n], y_train[:n])
        sizes.append(harness.model_bytes(model))

    assert sizes[0] == sizes[1], f"the Rc-PGM model changed with N: {sizes}"
