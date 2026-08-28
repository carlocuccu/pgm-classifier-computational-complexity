"""Reading the data, splitting it, and recording what a run produced."""

from __future__ import annotations

import csv
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from pgm_complexity import DATA_DIR
from pgm_complexity.config import BASE_TOL, HARMONIZE_TOL, SKIN_FILE
from pgm_complexity.threads import THREAD_VARS


def stratified_split(y, test_frac=0.2, seed=42):
    """Stratified train/test split, identical to the one of the Table 7 notebook.

    `sklearn.model_selection.train_test_split(..., shuffle=True, stratify=y,
    random_state=seed)` is used so that the training-set sizes N reported by the
    benchmarks coincide with those of Table 7; the partition depends only on `y`
    and on the seed, so splitting the index array reproduces the notebook's
    split of the data frame exactly.

    A numpy fallback keeps `selftest` runnable in an environment without
    scikit-learn; it rounds per class and can therefore differ by a unit or two
    in N, so it must not be used for reported runs.
    """
    index = np.arange(len(y))
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * test_frac)))
            te.extend(idx[:k])
            tr.extend(idx[k:])
        return np.array(sorted(tr)), np.array(sorted(te))

    tr, te = train_test_split(
        index, test_size=test_frac, shuffle=True, stratify=y, random_state=seed
    )
    return np.sort(tr), np.sort(te)


def stratified_subsample(y, n, seed):
    """~n indices, class proportions preserved (at least 1 per class)."""
    rng = np.random.default_rng(seed)
    out = []
    classes = np.unique(y)
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        k = max(1, int(round(n * len(idx) / len(y))))
        out.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.array(sorted(out))


def load_dataset(name: str):
    """CSV with the class label in the LAST column; Skin file is TSV.

    The features are used exactly as stored, with no scaling: this is what the
    Table 7 notebook does, and the amplitude encoding of the estimators
    normalises every sample to the unit sphere anyway. Together with the split
    of `stratified_split`, it makes the benchmarks fit the very same models the
    table describes, down to the reported accuracies.
    """
    path = DATA_DIR / (SKIN_FILE if name == "skin" else f"{name}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The datasets are not carried in the "
            f"repository; run `pgm data fetch` to download "
            f"and rebuild them."
        )
    raw = np.loadtxt(path, delimiter="\t" if name == "skin" else ",")
    return raw[:, :-1].astype(np.float64), raw[:, -1]


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "dataset",
        "N",
        "d",
        "c",
        "l",
        "fit_mean",
        "fit_std",
        "pred_mean",
        "pred_std",
        "fit_rss_peak",
        "pred_rss_peak",
        "model_bytes",
        "accuracy",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        # The value BLAS actually saw, not the one requested: a thread count
        # exported by the caller wins over --threads, and the record has to say
        # which one was in force.
        "threads": int(os.environ.get(THREAD_VARS[0], 1)),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_tol": BASE_TOL,
        "harmonize_tol": HARMONIZE_TOL,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    return info
