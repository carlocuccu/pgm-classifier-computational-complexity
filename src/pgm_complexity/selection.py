"""Choice of the copy number by cross-validation on the training set.

For every dataset the seeded stratified 80/20 split of the harness is drawn
once. The copy number is chosen on the training part only: stratified k-fold
cross-validation, repeated with different fold assignments, at every c of a
fixed grid. The selected c follows the one-standard-error rule (Breiman et
al., 1984): the smallest c whose mean cross-validated accuracy is within one
standard error of the largest one. The standard error is that of the mean over
the repetitions, i.e. the standard deviation of the per-repetition accuracies
divided by sqrt(repeats). The c of the largest mean (plain argmax) is recorded
too. The test part is used once per c, after the
selection, to report the accuracy of the model refitted on the whole training
set; it plays no role in the choice.

The same refit gives the retained rank r_{G^c} at every c, so the three
advantage conditions are evaluated at every copy number of the grid with the
rank of that c, rather than read off at the selected one.

The classifier is the k-PGM. The three estimators share the same predictions
(same priors, same threshold on the spectrum of sigma), so the accuracies hold
for all of them.
"""

from __future__ import annotations

import json

import numpy as np

from pgm_complexity import ROOT

SELECTION_DIR = ROOT / "results_selection"


def cv_accuracy(factory, X, y, c, folds, repeats):
    """Mean accuracy of each repetition of stratified ``folds``-fold CV.

    Repetition k shuffles the folds with ``random_state=k``. Returns an array
    of ``repeats`` values, each the mean over the folds of that repetition.
    """
    from sklearn.model_selection import StratifiedKFold

    out = np.empty(repeats)
    for k in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=k)
        acc = []
        for fit_idx, val_idx in skf.split(X, y):
            model = factory(c).fit(X[fit_idx], y[fit_idx])
            acc.append(np.mean(model.predict(X[val_idx]) == y[val_idx]))
        out[k] = np.mean(acc)
    return out


def argmax_copy_number(cv_mean, copies):
    """Smallest c whose mean CV accuracy equals the maximum over the grid."""
    cv_mean = np.asarray(cv_mean, dtype=float)
    best = cv_mean.max()
    for c, m in zip(copies, cv_mean, strict=True):
        if np.isclose(m, best, rtol=0.0, atol=1e-12):
            return c
    raise AssertionError("unreachable")


def select_copy_number(cv_mean, cv_se, copies):
    """One-standard-error rule: the smallest c whose mean CV accuracy is at
    least the maximum minus the standard error at the maximum."""
    cv_mean = np.asarray(cv_mean, dtype=float)
    c_best = argmax_copy_number(cv_mean, copies)
    i_best = list(copies).index(c_best)
    floor = cv_mean[i_best] - float(cv_se[i_best])
    for c, m in zip(copies, cv_mean, strict=True):
        if m >= floor - 1e-12:
            return c
    raise AssertionError("unreachable")


def sweep_dataset(name, copies, folds, repeats, seed, factory):
    """Rows of the sweep for one dataset, one per copy number."""
    from pgm_complexity.bench.io import load_dataset, stratified_split
    from pgm_complexity.thresholds import thresholds

    X, y = load_dataset(name)
    tr, te = stratified_split(y, 0.2, seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    n, d_raw = Xtr.shape
    l = len(np.unique(ytr))

    rows = []
    for c in copies:
        cv = cv_accuracy(factory, Xtr, ytr, c, folds, repeats)
        model = factory(c).fit(Xtr, ytr)
        r = int(model.lam_inv_sqrt.shape[0])
        test_acc = float(np.mean(model.predict(Xte) == yte))
        thr = thresholds(n, d_raw, c, l, r_g=r)
        rows.append(
            {
                "dataset": name,
                "N": n,
                "d_enc": d_raw + 1,
                "c": c,
                "l": l,
                "dsym": thr["dsym"],
                "r_Gc": r,
                "cv_mean": float(cv.mean()),
                "cv_std": float(cv.std(ddof=1)) if repeats > 1 else 0.0,
                "cv_se": float(cv.std(ddof=1) / np.sqrt(repeats))
                if repeats > 1
                else 0.0,
                "test_acc": test_acc,
                "tr_time_thr": thr["tr_time_thr"],
                "tr_mem_thr": thr["tr_mem_thr"],
                "pred_thr": thr["pred_thr"],
                "win_tr_time": bool(n > thr["tr_time_thr"]),
                "win_tr_mem": bool(n > thr["tr_mem_thr"]),
                "win_pred": bool(n > thr["pred_thr"]),
            }
        )
        print(
            f"{name:18s} c={c} dsym={thr['dsym']:5d} r={r:5d} "
            f"cv={cv.mean():.3f}±{rows[-1]['cv_std']:.3f} test={test_acc:.3f}"
        )
        del model
    cv_mean = [row["cv_mean"] for row in rows]
    c_star = select_copy_number(cv_mean, [row["cv_se"] for row in rows], copies)
    c_max = argmax_copy_number(cv_mean, copies)
    for row in rows:
        row["selected"] = row["c"] == c_star
        row["cv_argmax"] = row["c"] == c_max
    return rows


def selection_table(datasets, copies, folds=5, repeats=10, seed=42, outdir=None):
    """Run the sweep on every dataset and write the results.

    Writes ``selection_sweep.csv`` (every dataset at every c),
    ``selection.csv`` (the selected row of each dataset) and
    ``selection_meta.json`` in ``outdir`` (default ``results_selection/``).
    Returns the rows of the sweep.
    """
    import csv

    from pgm_complexity.bench.estimators import get_classifiers
    from pgm_complexity.bench.io import env_info

    outdir = SELECTION_DIR if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)
    factory = get_classifiers()["kpgm"]

    rows = []
    for name in datasets:
        rows.extend(sweep_dataset(name, copies, folds, repeats, seed, factory))

    fields = list(rows[0])
    with open(outdir / "selection_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    selected = [row for row in rows if row["selected"]]
    with open(outdir / "selection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)
    (outdir / "selection_meta.json").write_text(
        json.dumps(
            {
                "env": env_info(),
                "estimator": "kpgm",
                "split_seed": seed,
                "test_frac": 0.2,
                "folds": folds,
                "repeats": repeats,
                "cv_random_states": list(range(repeats)),
                "copies": list(copies),
                "rule": (
                    "one-standard-error: smallest c with mean CV accuracy >= "
                    "max - SE at the max, SE = std over repetitions / sqrt(repeats)"
                ),
            },
            indent=2,
        )
    )

    argmax = {row["dataset"]: row["c"] for row in rows if row["cv_argmax"]}
    print("\nselected copy number (cross-validation on the training set, 1-SE rule)")
    for row in selected:
        print(
            f"{row['dataset']:18s} c*={row['c']} (argmax {argmax[row['dataset']]}) "
            f"cv={row['cv_mean']:.3f}"
            f"±{row['cv_std']:.3f} test={row['test_acc']:.3f} "
            f"train time {row['win_tr_time']!s:5s} train mem {row['win_tr_mem']!s:5s} "
            f"pred {row['win_pred']!s:5s}"
        )
        if row["c"] == max(copies):
            print(f"{'':18s} c* is the last value of the grid")
    print(f"\n[select] {len(rows)} rows -> {outdir / 'selection_sweep.csv'}")
    return rows
