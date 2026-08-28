"""Constants of the measurement protocol.

Every number here is one the manuscript quotes or the harness records, kept in
one place so that a run and its metadata cannot disagree about them.
"""

from __future__ import annotations

# Eigenvalue threshold of the pseudo-inverse square root. BASE_TOL is the
# threshold of the k-PGM, applied to the spectrum of G^c; HARMONIZE_TOL rescales
# it for the estimators that truncate the spectrum of sigma = G^c / N instead
# (see rc_tol). Both values are recorded in the *_meta.json of every run.
BASE_TOL = 1e-6
HARMONIZE_TOL = True

# Copy numbers of Table 7 of the manuscript (accuracy saturation point).
TABLE7_C = {
    "analcatdata_dmft": 4,
    "balance-scale": 1,
    "car": 9,
    "cleveland-nominal": 1,
    "cloud": 1,
    "confidence": 5,
    "ecoli": 8,
    "haberman": 1,
    "iris": 2,
    "led7": 4,
    "new-thyroid": 5,
}
# Default Component A selection: two all-True, two partial, two all-False
# (regimes according to Table 7).
DEFAULT_A = ["balance-scale", "haberman", "iris", "led7", "ecoli", "car"]

SKIN_FILE = "Skin_NonSkin.txt"
SKIN_C = 5  # d_raw = 3 -> d_enc = 4, dsym = C(8,5) = 56
SKIN_TEST = 2000
DEFAULT_SWEEP = [250, 500, 1000, 1750, 3000, 5000, 8000]
