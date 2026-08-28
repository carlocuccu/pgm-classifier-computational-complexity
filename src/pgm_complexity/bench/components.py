"""Component A and Component B, the two experiments of the empirical section."""

from __future__ import annotations

import json

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
from pgm_complexity.config import (
    DEFAULT_SWEEP,
    SKIN_C,
    SKIN_TEST,
    TABLE7_C,
)
from pgm_complexity.figures import plot_B
from pgm_complexity.thresholds import thresholds


def component_A(args):
    factories = get_classifiers(include_cpgm=args.include_cpgm)
    rows, meta = [], []
    for name in args.datasets:
        c = TABLE7_C[name]
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
                    "near-identical; the residual mismatches come from the "
                    "pseudo-inverse truncation at near-threshold eigenvalues "
                    "-- set HARMONIZE_TOL=True for exact agreement"
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

    write_rows(OUT_DIR / "componentA.csv", rows)
    (OUT_DIR / "componentA_meta.json").write_text(
        json.dumps({"env": env_info(), "meta": meta}, indent=2)
    )
    print(f"\n[A] results -> {OUT_DIR / 'componentA.csv'}")


def component_B(args):
    factories = get_classifiers()
    X, y = load_dataset("skin")
    d_raw, c, l = X.shape[1], SKIN_C, len(np.unique(y))
    rng_split = stratified_subsample(y, SKIN_TEST, args.seed)
    mask = np.zeros(len(y), bool)
    mask[rng_split] = True
    Xte, yte = X[mask], y[mask]
    Xpool, ypool = X[~mask], y[~mask]

    info = thresholds(max(DEFAULT_SWEEP), d_raw, c, l)
    print(
        f"[B] Skin: d={d_raw}, c={c}, l={l}, dsym={info['dsym']}  |  theoretical "
        f"thresholds: train-time N*≈{info['tr_time_thr']:.0f}, "
        f"train-mem N*={info['tr_mem_thr']}, pred N*≈{info['pred_thr']:.0f}"
    )

    grid = [n for n in (args.sweep or DEFAULT_SWEEP) if n <= args.nmax]
    rows = []
    for n in grid:
        idx = stratified_subsample(ypool, n, args.seed + n)
        Xtr, ytr = Xpool[idx], ypool[idx]
        print(f"\n--- N = {len(idx)} ---")
        for method in ("kpgm", "rcpgm"):
            cell, _ = measure(
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
            print(
                f"  {method:6s} fit {cell.fit_mean:9.4f} ± {cell.fit_std:7.4f} s | "
                f"pred {cell.pred_mean:8.4f} ± {cell.pred_std:6.4f} s | "
                f"peak ΔRSS fit {cell.fit_rss_peak / 1e6:8.1f} MB | "
                f"model {cell.model_bytes / 1e6:8.1f} MB | acc {cell.accuracy:.4f}"
            )
            rows.append(cell)

    write_rows(OUT_DIR / "componentB.csv", rows)
    (OUT_DIR / "componentB_meta.json").write_text(
        json.dumps(
            {
                "env": env_info(),
                "thresholds": info,
                "grid": grid,
                "test_size": int(mask.sum()),
            },
            indent=2,
        )
    )
    print(f"\n[B] results -> {OUT_DIR / 'componentB.csv'}")
    try:
        plot_B(rows, info)
    except Exception as exc:  # matplotlib optional
        print(f"[B] plot skipped ({exc}); use componentB.csv")
