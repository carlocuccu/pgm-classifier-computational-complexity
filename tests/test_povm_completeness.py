"""The completed PGM operators form a POVM, also after spectral truncation.

The pseudo-inverse sigma^{-1/2} is regularized by discarding the eigenvalues
lam <= tol, i.e. it is the exact pseudo-inverse of the truncated operator
sigma_eps. Its kernel is spanned by every discarded eigendirection, including
those with 0 < lam <= tol, and the completion term (1/l) P_ker(sigma_eps) has
to be built from that same spectral mask: only then is

    sum_i F_i = sigma_eps^{-1/2} sigma sigma_eps^{-1/2} + P_ker = I.

A null space computed separately, with its own rank threshold, misses the
directions with 0 < lam <= tol, and the operators no longer sum to the
identity. The completion does not change the predicted labels (it adds the
same term to every class score), which is why label agreement alone cannot
detect the defect and the residuals are checked directly here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def data():
    """Three classes in R^3, so that sigma is rank deficient for c >= 2 in the
    c-PGM (the tensor space is larger than the symmetric subspace) and has a
    spread of small eigenvalues that a coarse threshold partly discards."""
    rng = np.random.default_rng(20261103)
    centres = np.array([[1.0, 0.2, -0.3], [-0.6, 0.9, 0.1], [0.2, -0.8, 0.7]])
    X = np.vstack([c + 0.35 * rng.normal(size=(30, 3)) for c in centres])
    y = np.repeat([0, 1, 2], 30)
    return X, y


def _estimators():
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype import (
        PGMHQC_gpu_cpu_dtype as CPGM,
    )
    from qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank import (
        PGMHQC_gpu_cpu_dtype as RcPGM,
    )

    return {"cpgm": CPGM, "rcpgm": RcPGM}


def _residuals(model):
    F = model.pgms_.to(torch.float64)
    total = F.sum(dim=0)
    eye = torch.eye(total.shape[0], dtype=total.dtype)
    completeness = torch.linalg.matrix_norm(total - eye, ord=2).item()
    min_eig = min(torch.linalg.eigvalsh((Fi + Fi.T) / 2).min().item() for Fi in F)
    return completeness, min_eig


@pytest.mark.parametrize("name", ["cpgm", "rcpgm"])
@pytest.mark.parametrize("c", [1, 2, 3])
@pytest.mark.parametrize("tol", [1e-6, 1e-3, 1e-2])
def test_completed_operators_sum_to_identity(data, name, c, tol):
    X, y = data
    model = _estimators()[name](n_copies=c, encoding="amplit", tol=tol)
    model.fit(X, y)

    completeness, min_eig = _residuals(model)
    assert completeness < 1e-8, f"||sum F_i - I||_2 = {completeness:.3e}"
    assert min_eig > -1e-10, f"min eigenvalue of F_i = {min_eig:.3e}"


@pytest.mark.parametrize("c", [2, 3])
def test_coarse_threshold_actually_discards_positive_eigenvalues(data, c):
    """Guard on the test itself: with tol = 1e-2 some strictly positive
    eigenvalues of sigma fall below the threshold, i.e. the case in which a
    separately computed null space would differ from the discarded subspace."""
    X, y = data
    model = _estimators()["rcpgm"](n_copies=c, encoding="amplit", tol=1e-2)
    model.fit(X, y)
    exact_rank = _estimators()["rcpgm"](n_copies=c, encoding="amplit", tol=1e-13)
    exact_rank.fit(X, y)
    assert model.rank_sigma_ < exact_rank.rank_sigma_


@pytest.mark.parametrize("c", [2, 3])
def test_cpgm_and_rcpgm_share_spectrum_and_predictions(data, c):
    """sigma_c and sigma_sym have the same nonzero spectrum (Theorem 1), so the
    same threshold retains the same rank and the two predict the same labels."""
    X, y = data
    est = _estimators()
    cpgm = est["cpgm"](n_copies=c, encoding="amplit", tol=1e-3).fit(X, y)
    rcpgm = est["rcpgm"](n_copies=c, encoding="amplit", tol=1e-3).fit(X, y)
    assert cpgm.rank_sigma_ == rcpgm.rank_sigma_
    np.testing.assert_array_equal(cpgm.predict(X), rcpgm.predict(X))
