"""The three advantage conditions of Subsection "Rc-PGM versus k-PGM".

Exact arithmetic in N, l, the encoded dimension and the retained rank -- no
measurement enters here, which is why Table 7 is reproducible on any machine.
"""

from __future__ import annotations

from math import comb


def dsym_of(d_raw: int, c: int, encoded: bool = True) -> int:
    d_enc = d_raw + 1 if encoded else d_raw
    return comb(d_enc + c - 1, c)


def thresholds(N, d_raw, c, l, r_g=None):
    """Corrected Table-7 conditions; r_g defaults to min(N, dsym)."""
    ds = dsym_of(d_raw, c)
    r = r_g if r_g is not None else min(N, ds)
    return {
        "dsym": ds,
        "tr_time_thr": l ** (1 / 3) * ds,
        "tr_mem_thr": ds**2,
        "pred_thr": l * ds**2 / (d_raw + 1 + r),
    }
