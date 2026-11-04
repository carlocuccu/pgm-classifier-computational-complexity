"""Component A and Component B, the two experiments of the empirical section."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pgm_complexity import OUT_DIR
from pgm_complexity.bench.estimators import get_classifiers, rc_preprocessing_time
from pgm_complexity.bench.io import (
    env_info,
    load_dataset,
    stratified_split,
    stratified_subsample,
    write_rows,
)
from pgm_complexity.bench.measure import measure
from pgm_complexity.config import DEFAULT_SWEEP, SKIN_COPIES, SKIN_TEST
from pgm_complexity.crossovers import crossover_table, format_table
from pgm_complexity.figures import plot_B
from pgm_complexity.thresholds import dsym_of, thresholds


def _outdir(args) -> Path:
    out = Path(args.outdir).resolve() if getattr(args, "outdir", None) else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _copy_numbers(args):
    """The copy number of each dataset, as chosen by `pgm select`."""
    from pgm_complexity.selection import SELECTION_DIR, selected_copies

    path = (
        Path(args.selection).resolve()
        if getattr(args, "selection", None)
        else SELECTION_DIR / "selection.csv"
    )
    if not path.exists():
        raise SystemExit(
            f"{path} not found: Component A runs at the copy numbers chosen by "
            "`pgm select`; run it first, or point --selection at its output."
        )
    chosen = selected_copies(path)
    missing = [name for name in args.datasets if name not in chosen]
    if missing:
        raise SystemExit(f"{path} has no copy number for: {', '.join(missing)}")
    return chosen, path


def component_A(args):
    outdir = _outdir(args)
    chosen, selection_path = _copy_numbers(args)
    factories = get_classifiers(include_cpgm=args.include_cpgm)
    rows, meta = [], []
    for name in args.datasets:
        c = chosen[name]
        X, y = load_dataset(name)
        tr, te = stratified_split(y, 0.2, args.seed)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        info = thresholds(len(tr), X.shape[1], c, len(np.unique(y)))
        ds = info["dsym"]

        # Calibrated Rc-PGM fit-peak estimate: 2l persistent matrices
        # (centroids + PGM operators) plus ~5 dsym^2 transients (sigma, the
        # eigh workspace, the sandwich temporaries), in float64.
        est_bytes = (2 * len(np.unique(y)) + 5) * ds * ds * 8
        if est_bytes > args.mem_limit_gb * 1e9:
            print(
                f"[skip] {name}: estimated Rc-PGM footprint "
                f"{est_bytes / 1e9:.1f} GB > --mem-limit-gb {args.mem_limit_gb}"
            )
            continue

        print(
            f"\n=== {name}: N={len(tr)}, d={X.shape[1]}, c={c}, "
            f"l={len(np.unique(y))}, dsym={ds} ==="
        )
        preds = {}
        for method in ["kpgm", "rcpgm"] + (["cpgm"] if args.include_cpgm else []):
            if method == "cpgm" and (X.shape[1] + 1) ** c > args.cpgm_dim_limit:
                print(f"  [skip] c-PGM: d_enc^c = {(X.shape[1] + 1) ** c} too large")
                continue
            cell, holder = measure(
                method, factories[method], c, Xtr, ytr, Xte, yte, name, args.reps
            )
            preds[method] = holder["yhat"]
            print(
                f"  {method:6s} fit {cell.fit_mean * 1e3:9.2f} ± {cell.fit_std * 1e3:7.2f} ms | "
                f"pred {cell.pred_mean * 1e3:8.2f} ± {cell.pred_std * 1e3:6.2f} ms | "
                f"peak ΔRSS fit {cell.fit_rss_peak / 1e6:8.1f} MB | "
                f"model {cell.model_bytes / 1e6:8.1f} MB | acc {cell.accuracy:.3f}"
            )
            rows.append(cell)

        # Rc-PGM preprocessing, timed standalone
        (tb, sb), (tm, sm) = rc_preprocessing_time(
            factories["rcpgm"], Xtr, c, args.reps
        )
        print(
            f"  rcpgm  preprocessing: basis+factors {tb * 1e3:.2f} ± {sb * 1e3:.2f} ms | "
            f"encode+map {tm * 1e3:.2f} ± {sm * 1e3:.2f} ms"
        )

        # Equivalence check
        if "kpgm" in preds and "rcpgm" in preds:
            agree = float(np.mean(preds["kpgm"] == preds["rcpgm"]))
            if agree == 1.0:
                status = "OK (identical)"
            elif agree > 0.99:
                status = (
                    "near-identical; with a shared threshold the residual "
                    "mismatches can only come from rounding at eigenvalues "
                    "within machine precision of the threshold, or from ties"
                )
            else:
                status = "CHECK: large disagreement, the timings are not comparable"
            print(
                f"  equivalence k-PGM vs Rc-PGM: argmax agreement {agree:.4f}  [{status}]"
            )
            meta.append(
                {
                    "dataset": name,
                    "argmax_agreement": agree,
                    "rc_prep_basis_s": tb,
                    "rc_prep_map_s": tm,
                    **info,
                }
            )

    write_rows(outdir / "componentA.csv", rows)
    (outdir / "componentA_meta.json").write_text(
        json.dumps(
            {
                "env": env_info(),
                "copy_numbers_from": str(selection_path),
                "meta": meta,
            },
            indent=2,
        )
    )
    print(f"\n[A] results -> {outdir / 'componentA.csv'}")


def _rank(method, est) -> int:
    """Retained rank of a fitted model: r_{G^c} for the k-PGM, the rank of
    sigma for the Rc-PGM (the same number, by the shared threshold)."""
    if method == "rcpgm":
        return int(est.rank_sigma_)
    return int(est.lam_inv_sqrt.shape[0])


def component_B(args):
    """The sweep over N on Skin Segmentation, at every copy number of
    ``args.copies``, for the dense k-PGM, the k-PGM from explicit features and
    the Rc-PGM; then the empirical crossovers against the thresholds."""
    outdir = _outdir(args)
    factories = get_classifiers()
    X, y = load_dataset("skin")
    d_raw, l = X.shape[1], len(np.unique(y))
    copies = list(getattr(args, "copies", None) or SKIN_COPIES)
    rng_split = stratified_subsample(y, SKIN_TEST, args.seed)
    mask = np.zeros(len(y), bool)
    mask[rng_split] = True
    Xte, yte = X[mask], y[mask]
    Xpool, ypool = X[~mask], y[~mask]

    grid = [n for n in (args.sweep or DEFAULT_SWEEP) if n <= args.nmax]
    # The same training sets at every copy number.
    subsets = {n: stratified_subsample(ypool, n, args.seed + n) for n in grid}
    methods = ("kpgm", "kpgm_features", "rcpgm")

    rows, ranks, infos = [], {}, {}
    for c in copies:
        print(f"\n[B] Skin: d={d_raw}, c={c}, l={l}, dsym={dsym_of(d_raw, c)}")
        ranks[c] = {}
        for n in grid:
            idx = subsets[n]
            Xtr, ytr = Xpool[idx], ypool[idx]
            print(f"--- c = {c}, N = {len(idx)} ---")
            for method in methods:
                cell, holder = measure(
                    method,
                    factories[method],
                    c,
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    f"skin_N{len(idx)}",
                    args.reps,
                )
                if method == "kpgm":
                    ranks[c][len(idx)] = _rank(method, holder["est"])
                print(
                    f"  {method:13s} fit {cell.fit_mean:9.4f} ± {cell.fit_std:7.4f} s | "
                    f"pred {cell.pred_mean:8.4f} ± {cell.pred_std:6.4f} s | "
                    f"peak ΔRSS fit {cell.fit_rss_peak / 1e6:8.1f} MB | "
                    f"model {cell.model_bytes / 1e6:8.1f} MB | acc {cell.accuracy:.4f}"
                )
                rows.append(cell)

        # Thresholds at the largest N of the sweep, with the rank retained
        # there and the fraction of the largest class in that training set.
        n_last = len(subsets[grid[-1]])
        y_last = ypool[subsets[grid[-1]]]
        fraction = float(np.max(np.unique(y_last, return_counts=True)[1]) / n_last)
        infos[c] = thresholds(
            n_last, d_raw, c, l, r_g=ranks[c][n_last], class_fraction=fraction
        )

    write_rows(outdir / "componentB.csv", rows)
    table = crossover_table([asdict(cell) for cell in rows], infos)
    with open(outdir / "componentB_crossovers.csv", "w", newline="") as fh:
        fields = [
            "c",
            "dsym",
            "quantity",
            "theoretical",
            "theoretical_class",
            "empirical",
            "status",
            "ratio",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    (outdir / "componentB_meta.json").write_text(
        json.dumps(
            {
                "env": env_info(),
                "copies": copies,
                "methods": list(methods),
                "thresholds": {str(c): info for c, info in infos.items()},
                "ranks": {
                    str(c): {str(n): r for n, r in rk.items()}
                    for c, rk in ranks.items()
                },
                "grid": grid,
                "test_size": int(mask.sum()),
            },
            indent=2,
        )
    )
    print(f"\n[B] results -> {outdir / 'componentB.csv'}")
    print("\n[B] crossovers of the dense k-PGM and the Rc-PGM:")
    print(format_table(table))
    try:
        for c in copies:
            plot_B([r for r in rows if r.c == c], infos[c], outdir, suffix=f"_c{c}")
    except Exception as exc:  # matplotlib optional
        print(f"[B] plot skipped ({exc}); use componentB.csv")
