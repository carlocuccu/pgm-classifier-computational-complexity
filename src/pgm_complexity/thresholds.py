"""Cost proxies and advantage conditions of the PGM classifiers.

Every quantity here is a dominant-term cost proxy under unit-cost arithmetic:
the number of elementary operations or stored scalars of each step, with
multiplicative constants dropped. N is the training-set size, l the number of
classes, d the encoded dimension (raw features + 1 for the amplitude
encoding), d_sym = C(d + c - 1, c) the dimension of the symmetric subspace and
r the retained numerical rank of K (equivalently of sigma).

Three realizations of the same classifier are compared:

* ``rcpgm``          the reduced primal (Rc-PGM);
* ``kpgm``           the dense dual (k-PGM as in Cruzeiro et al.): it forms
                     the N x N Gram matrix and diagonalizes it;
* ``kpgm_features``  the dual taken from the thin SVD of W^{1/2} Y, Y the
                     N x d_sym feature matrix (solver='features').

The advantage conditions of the paper compare ``rcpgm`` with ``kpgm``. No
measurement enters here, which is why the tables are reproducible on any
machine.
"""

from __future__ import annotations

from math import comb, sqrt


def dsym_of(d_raw: int, c: int, encoded: bool = True) -> int:
    d_enc = d_raw + 1 if encoded else d_raw
    return comb(d_enc + c - 1, c)


def cost_proxies(N, d, dsym, l, r=None, n_max=None):
    """Dominant-term proxies of the three realizations.

    Training time takes the max over the steps, as in the tables of the paper:
    for the Rc-PGM the reduction (N d d_sym), the centroids (N d_sym^2) and the
    operators E_i (l d_sym^3); for the dense k-PGM the Gram matrix (d N^2) and
    its eigendecomposition (N^3); for the k-PGM with explicit features the
    feature matrix (N d d_sym) and its thin SVD (N d_sym min(N, d_sym)).

    Training memory is given under two accountings for the Rc-PGM:
    ``train_mem_batched`` is the upper bound of the tables (all N densities at
    once, N d_sym^2); ``train_mem_class`` is the peak of the implementation,
    which forms the densities one class at a time (n_max d_sym^2, n_max the
    size of the largest class) on top of the reduced vectors (N d_sym) and the
    l operators (l d_sym^2). The k-PGM stores its N x N matrix and the encoded
    training vectors (N^2 + N d), or N d_sym + N d with explicit features.

    Prediction (per test sample) and the stored model: l d_sym^2 for the
    Rc-PGM, N (d + r) for both duals.
    """
    r = min(N, dsym) if r is None else r
    n_max = N if n_max is None else n_max
    return {
        "rcpgm": {
            "train_time": max(N * d * dsym, N * dsym**2, l * dsym**3),
            "train_mem_batched": N * dsym**2,
            "train_mem_class": n_max * dsym**2 + N * dsym + l * dsym**2,
            "pred_time": l * dsym**2,
            "pred_mem": l * dsym**2,
        },
        "kpgm": {
            "train_time": max(d * N**2, N**3),
            "train_mem": N**2 + N * d,
            "pred_time": N * (d + r),
            "pred_mem": N * (d + r),
        },
        "kpgm_features": {
            "train_time": max(N * d * dsym, N * dsym * min(N, dsym)),
            "train_mem": N * dsym + N * d,
            "pred_time": N * (d + r),
            "pred_mem": N * (d + r),
        },
    }


def rcpgm_fit_peak_bytes(n, n_max, l, dsym):
    """Peak memory of an Rc-PGM fit, in bytes (float64): the densities of
    the largest class (n_max d_sym^2), the reduced vectors (n d_sym), and the
    l centroids, the l operators and about five d_sym x d_sym transients.

    Used to decide which configurations exceed --mem-limit-gb. It matches
    the peaks measured in the runs of v1.1.0-rc1 within a few per cent
    (car at c = 9: about 196 GB; ecoli at c = 9: about 132 GB).
    """
    return 8 * ((n_max + 2 * l + 5) * dsym**2 + n * dsym)


def _quadratic_threshold(b, q):
    """Positive root of N^2 - b N - q = 0, q > 0: N^2 > b N + q iff N > root."""
    return (b + sqrt(b * b + 4 * q)) / 2


def thresholds(N, d_raw, c, l, r_g=None, class_fraction=None):
    """Thresholds N* above which the Rc-PGM is cheaper than the dense k-PGM.

    ``tr_time_thr`` = l^(1/3) d_sym. The training cost of the dense k-PGM is
    max(d N^2, N^3); the d N^2 term does not move the threshold, because the
    condition already implies N > d_sym >= d, where N^3 >= d N^2 (checked in
    the tests).

    ``tr_mem_thr`` = d_sym^2, the batched upper bound of the tables: a
    sufficient condition. ``tr_mem_thr_class`` uses the per-class peak of the
    implementation with the largest class a fraction ``class_fraction`` of N
    (default 1, a single class); it solves the quadratic inequality exactly,
    against the k-PGM memory N^2 + N d.

    ``pred_thr`` = l d_sym^2 / (d + r): prediction time and stored model.
    r_g defaults to min(N, d_sym), the largest possible rank.
    """
    d = d_raw + 1
    ds = dsym_of(d_raw, c)
    r = r_g if r_g is not None else min(N, ds)
    pi = 1.0 if class_fraction is None else class_fraction
    return {
        "dsym": ds,
        "tr_time_thr": l ** (1 / 3) * ds,
        "tr_mem_thr": ds**2,
        "tr_mem_thr_class": _quadratic_threshold(pi * ds**2 + ds - d, l * ds**2),
        "pred_thr": l * ds**2 / (d + r),
    }


_PAIRS = {
    "train_time": ("train_time", "train_time"),
    "train_mem_batched": ("train_mem_batched", "train_mem"),
    "train_mem_class": ("train_mem_class", "train_mem"),
    "pred_time": ("pred_time", "pred_time"),
}


def crossover(
    metric, d_raw, c, l, other="kpgm", r_g=None, class_fraction=None, n_max=10**9
):
    """Smallest N at which the Rc-PGM proxy is strictly below the other one.

    A direct search over the proxies of ``cost_proxies``, used to check the
    closed-form thresholds. ``r_g`` None means min(N, d_sym) at each N. Returns
    None if no N <= n_max qualifies. The search is a bisection, so it assumes
    the advantage, once gained, is kept for larger N; the tests check this.
    """
    d = d_raw + 1
    ds = dsym_of(d_raw, c)
    mine, theirs = _PAIRS[metric]

    def wins(N):
        nm = None if class_fraction is None else class_fraction * N
        p = cost_proxies(N, d, ds, l, r=r_g, n_max=nm)
        return p["rcpgm"][mine] < p[other][theirs]

    if not wins(n_max):
        return None
    lo, hi = 0, n_max  # wins(hi) holds; find the smallest such N
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if wins(mid):
            hi = mid
        else:
            lo = mid
    return hi
