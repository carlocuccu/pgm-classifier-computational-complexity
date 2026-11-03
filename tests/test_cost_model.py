"""The cost model behind the advantage conditions, term by term.

The thresholds of the paper are closed forms; here they are checked against a
direct search over the cost proxies of each realization, including the terms
the closed forms drop (d N^2 in the dense k-PGM training).
"""

from __future__ import annotations

import math

import pytest

from pgm_complexity.thresholds import cost_proxies, crossover, dsym_of, thresholds

GRID = [
    (d_raw, c, l)
    for d_raw in (2, 3, 4, 7)
    for c in (1, 2, 3, 4, 5)
    for l in (2, 3, 6, 10)
]


@pytest.mark.parametrize("d_raw,c,l", GRID)
def test_gram_term_leaves_the_training_time_threshold_unchanged(d_raw, c, l):
    """With max(d N^2, N^3) for the k-PGM the crossover is still l^(1/3) d_sym."""
    thr = thresholds(1, d_raw, c, l)["tr_time_thr"]
    n_star = crossover("train_time", d_raw, c, l)
    assert n_star == math.floor(thr) + 1
    # In that region N >= d_sym >= d, so the Gram term never dominates.
    d = d_raw + 1
    assert n_star >= dsym_of(d_raw, c) >= d
    assert d * n_star**2 <= n_star**3


@pytest.mark.parametrize("d_raw,c,l", GRID)
def test_memory_thresholds_match_the_search(d_raw, c, l):
    """The quadratic root is the exact crossover of the per-class peak."""
    thr = thresholds(1, d_raw, c, l, class_fraction=0.5)["tr_mem_thr_class"]
    n_star = crossover("train_mem_class", d_raw, c, l, class_fraction=0.5)
    assert n_star - 1 <= thr < n_star


@pytest.mark.parametrize("d_raw,c,l", GRID)
def test_batched_memory_bound_is_conservative(d_raw, c, l):
    """Past d_sym^2 the batched densities fit; the full peak needs d_sym + l more.

    The bound of the tables counts the N densities only. The per-class peak of
    the implementation adds the reduced vectors and the l operators, which
    shifts the crossover by at most d_sym + l, for any class fraction.
    """
    thr = thresholds(1, d_raw, c, l, class_fraction=1.0)
    ds = dsym_of(d_raw, c)
    n = ds**2 + 1
    p = cost_proxies(n, d_raw + 1, ds, l, n_max=n)
    assert p["rcpgm"]["train_mem_batched"] < p["kpgm"]["train_mem"]
    assert thr["tr_mem_thr_class"] <= ds**2 + ds + l


@pytest.mark.parametrize("d_raw,c,l", GRID)
def test_prediction_threshold_matches_the_search(d_raw, c, l):
    """l d_sym^2 / (d + r) is the crossover of the per-sample prediction cost."""
    r = 7
    thr = thresholds(1, d_raw, c, l, r_g=r)["pred_thr"]
    n_star = crossover("pred_time", d_raw, c, l, r_g=r)
    assert n_star - 1 <= thr < n_star


@pytest.mark.parametrize("d_raw,c,l", GRID)
@pytest.mark.parametrize("n", [10, 100, 1000, 10**4, 10**5])
def test_rcpgm_training_never_beats_the_feature_dual(d_raw, c, l, n):
    """Against the k-PGM from explicit features the Rc-PGM has no training edge.

    Both pay N d d_sym for the features; the Rc-PGM then pays N d_sym^2 and
    l d_sym^3, the feature dual N d_sym min(N, d_sym) <= N d_sym^2.
    """
    ds = dsym_of(d_raw, c)
    p = cost_proxies(n, d_raw + 1, ds, l)
    assert p["kpgm_features"]["train_time"] <= p["rcpgm"]["train_time"]
    # Prediction is the same as for the dense dual.
    assert p["kpgm_features"]["pred_time"] == p["kpgm"]["pred_time"]


def test_old_keys_are_unchanged():
    """The keys used by the benchmarks and the tables keep their values."""
    thr = thresholds(500, 3, 5, 2)
    assert thr["dsym"] == 56
    assert thr["tr_mem_thr"] == 3136
    assert thr["tr_time_thr"] == pytest.approx(2 ** (1 / 3) * 56)
    assert thr["pred_thr"] == pytest.approx(2 * 56**2 / (4 + 56))
