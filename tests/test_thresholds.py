"""The three advantage conditions, and the table the paper builds from them.

These are the inequalities of Subsection "Rc-PGM versus k-PGM". They are exact
arithmetic in N, l, d and the retained rank, not estimates, so they can be
checked against the published table row by row.
"""

from __future__ import annotations

import math

import pytest

# Table 7 of the manuscript, verbatim: name, N, d (encoded), c, l, d_sym,
# r_{G^c}, and the three advantage flags in the order training time, training
# memory, prediction. This is the golden oracle: if a change to the formulas
# moves any of these, it has changed a published claim.
TABLE7 = [
    ("analcatdata_dmft", 629, 5, 4, 6, 70, 50, True, False, True),
    ("balance-scale", 500, 5, 1, 3, 5, 5, True, True, True),
    ("car", 1381, 7, 9, 4, 5005, 1230, False, False, False),
    ("cleveland-nominal", 242, 8, 1, 5, 8, 8, True, True, True),
    ("cloud", 86, 8, 1, 4, 8, 8, True, True, True),
    ("confidence", 57, 4, 5, 6, 56, 21, False, False, False),
    ("ecoli", 261, 8, 8, 5, 6435, 137, False, False, False),
    ("haberman", 244, 4, 1, 2, 4, 4, True, True, True),
    ("iris", 120, 5, 2, 3, 15, 15, True, False, True),
    ("led7", 2560, 8, 4, 10, 330, 99, True, False, False),
    ("new-thyroid", 172, 6, 5, 3, 252, 31, False, False, False),
]


@pytest.mark.parametrize("row", TABLE7, ids=lambda r: r[0])
def test_table7_row_reproduces(harness, row):
    """Every flag printed in Table 7 follows from the tabulated quantities."""
    _, n, d, c, l, d_sym, r_gc, win_time, win_mem, win_pred = row
    thr = harness.thresholds(N=n, d_raw=d - 1, c=c, l=l, r_g=r_gc)

    assert thr["dsym"] == d_sym
    assert (n > thr["tr_time_thr"]) is win_time
    assert (n > thr["tr_mem_thr"]) is win_mem
    assert (n > thr["pred_thr"]) is win_pred


@pytest.mark.parametrize("row", TABLE7, ids=lambda r: r[0])
def test_retained_rank_within_its_bound(row):
    """r_{G^c} <= min(N, d_sym), the consistency bound the Note claims."""
    _, n, _, _, _, d_sym, r_gc, *_ = row
    assert r_gc <= min(n, d_sym)


def test_dsym_is_the_symmetric_subspace_dimension(harness):
    """d_sym = C(d + c - 1, c) on the encoded dimension, d = d_raw + 1."""
    for d_raw in range(1, 10):
        for c in range(1, 10):
            assert harness.dsym_of(d_raw, c) == math.comb(d_raw + c, c)
    # c = 1 leaves the space untouched.
    assert harness.dsym_of(7, 1) == 8


def test_threshold_formulas(harness):
    """The three thresholds are the expressions of Eqs. (26), (27) and (28)."""
    n, d_raw, c, l, r = 1000, 4, 3, 5, 17
    thr = harness.thresholds(N=n, d_raw=d_raw, c=c, l=l, r_g=r)
    d_sym = math.comb(d_raw + c, c)

    assert thr["tr_time_thr"] == pytest.approx(l ** (1 / 3) * d_sym)
    assert thr["tr_mem_thr"] == d_sym**2
    assert thr["pred_thr"] == pytest.approx(l * d_sym**2 / (d_raw + 1 + r))


def test_retained_rank_defaults_to_its_upper_bound(harness):
    """Without a measured rank the conditions use min(N, d_sym), the worst case."""
    n, d_raw, c, l = 500, 4, 3, 5
    d_sym = math.comb(d_raw + c, c)
    assert harness.thresholds(N=n, d_raw=d_raw, c=c, l=l)["pred_thr"] == pytest.approx(
        harness.thresholds(N=n, d_raw=d_raw, c=c, l=l, r_g=min(n, d_sym))["pred_thr"]
    )


@pytest.mark.parametrize("row", TABLE7, ids=lambda r: r[0])
def test_conditions_are_monotone_in_the_copy_number(harness, row):
    """Each advantage holds on an initial segment of copy numbers.

    d_sym grows with c while N, l and d do not, so once a condition fails it
    fails for every larger c. This is what lets the table be read as a snapshot
    of a monotone trade-off rather than as an isolated measurement, and it is
    the property the appendix sweep table summarises with three integers.
    """
    _, n, d, _, l, *_ = row
    for name in ("tr_time_thr", "tr_mem_thr", "pred_thr"):
        holds = [
            n > harness.thresholds(N=n, d_raw=d - 1, c=c, l=l)[name]
            for c in range(1, 10)
        ]
        # True* False* : no True may follow a False.
        assert holds == sorted(holds, reverse=True), f"{name} not monotone in c"
