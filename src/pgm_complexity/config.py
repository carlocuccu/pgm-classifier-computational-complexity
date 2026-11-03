"""Constants of the measurement protocol.

Every number here is one the manuscript quotes or the harness records, kept in
one place so that a run and its metadata cannot disagree about them.
"""

from __future__ import annotations

# Eigenvalue threshold of the pseudo-inverse square root. All three estimators
# truncate the spectrum of sigma = sum_j p_j rho_j, a density operator of trace
# one (the k-PGM through K = W^{1/2} G^c W^{1/2}, which has the same nonzero
# spectrum), and all three use the same threshold, independent of N. Its value
# is a margin above the rounding noise of a double-precision eigensolver,
# about d_sym * eps * lambda_max <~ 1e-12 for d_sym up to a few thousand.
# It is recorded in every *_meta.json.
#
# Up to v1.0.1 the threshold was 1e-6 on the spectrum of the unweighted G^c,
# i.e. 1e-6 / N on sigma; the runs of that version are reproducible from its
# tag.
BASE_TOL = 1e-10

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
