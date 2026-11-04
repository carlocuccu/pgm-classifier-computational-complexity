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

# The eleven datasets of Table 7. Their copy numbers are not fixed here: they
# are chosen by `pgm select` (cross-validation on the training set) and read
# by Component A from results_selection/selection.csv.
DATASETS = (
    "analcatdata_dmft",
    "balance-scale",
    "car",
    "cleveland-nominal",
    "cloud",
    "confidence",
    "ecoli",
    "haberman",
    "iris",
    "led7",
    "new-thyroid",
)
# Default Component A selection: the six datasets of Table 8.
DEFAULT_A = ["balance-scale", "haberman", "iris", "led7", "ecoli", "car"]

SKIN_FILE = "Skin_NonSkin.txt"
# Component B sweeps N at several copy numbers. d_raw = 3 -> d_enc = 4, so
# d_sym = C(c + 3, c) = 10, 20, 35, 56, 84, 120 for c = 2 ... 7.
SKIN_COPIES = [2, 3, 4, 5, 6, 7]
SKIN_TEST = 2000
# From N = 25, so that the thresholds of the smallest copy numbers (about 13
# at c = 2) fall inside the sweep, up to N = 8000.
DEFAULT_SWEEP = [25, 50, 100, 250, 500, 1000, 1750, 3000, 5000, 8000]
