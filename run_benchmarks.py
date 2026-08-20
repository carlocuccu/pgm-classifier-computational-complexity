#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical benchmarks for the section "Some empirical evidence" of

    Computational Complexity Analysis of Quantum-Inspired Pretty Good
    Measurement Classifiers.

Run this file from the repository root, next to the `qunica/` package and the
`datasets/` folder:

    pgm-repo/
    ├── run_benchmarks.py          <-- this file
    ├── qunica/
    │   └── classifiers/
    │       ├── KPGMC_Low_Rank.py
    │       ├── PGMHQC_gpu_cpu_dtype.py
    │       └── PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank.py
    └── datasets/
        ├── iris.csv ...           (header-less, last column = class label)
        └── Skin_NonSkin.txt       (datasets/download_skin_segmentation.py)

Component A  (multi-dataset spot checks at the tabulated copy number c):
    python run_benchmarks.py A --reps 5

Component B  (N-sweep around the theoretical crossover, Skin dataset, c=5):
    python run_benchmarks.py B --reps 3 --nmax 8000

Both:
    python run_benchmarks.py all

Pipeline self-test without torch/qunica (mock classifiers, synthetic data):
    python run_benchmarks.py selftest

Measurement protocol (see README.md, "Measurement protocol"):
  * BLAS/torch threads are pinned BEFORE importing numpy/torch (default 1,
    override with --threads). Report the value with the results.
  * Every phase is measured with 1 warm-up + R timed repetitions
    (mean +/- std of time.perf_counter).
  * Memory is reported two ways: (i) peak RSS delta during the phase, sampled
    from /proc/self/statm by a background thread; (ii) bytes of the arrays /
    tensors actually stored on the fitted estimator ("model bytes"), which is
    the quantity the theoretical memory proxies describe.
  * Each run ends with an equivalence check: k-PGM and Rc-PGM must produce the
    same argmax on the test set. A failure there invalidates the timings, and
    the script says so explicitly.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

# ----------------------------------------------------------------------------
# Thread pinning must happen before numpy / torch are imported.
# ----------------------------------------------------------------------------
def _pin_threads_from_argv() -> int:
    n = 1
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
    return n

N_THREADS = _pin_threads_from_argv()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "results_benchmark"

# Copy numbers of Table 7 of the manuscript (accuracy saturation point).
TABLE7_C = {
    "analcatdata_dmft": 4, "balance-scale": 1, "car": 9, "cleveland-nominal": 1,
    "cloud": 1, "confidence": 5, "ecoli": 8, "haberman": 1, "iris": 2,
    "led7": 4, "new-thyroid": 5,
}
# Default Component A selection: two all-True, two partial, two all-False
# (regimes according to Table 7).
DEFAULT_A = ["balance-scale", "haberman", "iris", "led7", "ecoli", "car"]

SKIN_FILE = "Skin_NonSkin.txt"
SKIN_C = 5            # d_raw = 3 -> d_enc = 4, dsym = C(8,5) = 56
SKIN_TEST = 2000
DEFAULT_SWEEP = [250, 500, 1000, 1750, 3000, 5000, 8000]


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
class PeakRSS:
    """Background sampler of the process resident set size (Linux)."""

    def __init__(self, interval: float = 0.002):
        self.interval = interval
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._stop = threading.Event()
        self._thread = None
        self.baseline = 0
        self.peak = 0

    def _rss(self) -> int:
        try:
            with open("/proc/self/statm") as fh:
                return int(fh.read().split()[1]) * self._page
        except OSError:  # non-Linux fallback
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        gc.collect()
        self.baseline = self._rss()
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.baseline)


def model_bytes(est) -> int:
    """Bytes of the arrays / tensors stored on a fitted estimator."""
    total, seen = 0, set()
    for value in vars(est).values():
        oid = id(value)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif hasattr(value, "element_size") and hasattr(value, "nelement"):
            total += value.element_size() * value.nelement()  # torch tensor
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, np.ndarray):
                    total += item.nbytes
                elif hasattr(item, "element_size"):
                    total += item.element_size() * item.nelement()
    return total


def timed(fn, reps: int, warmup: int = 1):
    """Run fn() warmup+reps times; return (mean, std, last_result, times)."""
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std(ddof=1)) if reps > 1 else 0.0, result, times


def stratified_split(y, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        te.extend(idx[:k]); tr.extend(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


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


def load_dataset(name: str, minmax: bool = True):
    """CSV with the class label in the LAST column; Skin file is TSV."""
    if name == "skin":
        raw = np.loadtxt(DATA_DIR / SKIN_FILE, delimiter="\t")
    else:
        raw = np.loadtxt(DATA_DIR / f"{name}.csv", delimiter=",")
    X, y = raw[:, :-1].astype(np.float64), raw[:, -1]
    if minmax:
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        X = (X - lo) / span
    return X, y


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
        "tr_mem_thr": ds ** 2,
        "pred_thr": l * ds ** 2 / (d_raw + 1 + r),
    }


# ----------------------------------------------------------------------------
# Classifier access (lazy, so `selftest` runs without torch/qunica)
# ----------------------------------------------------------------------------
def get_classifiers(include_cpgm=False):
    import importlib

    def load(modname):
        for prefix in ("qunica.classifiers.", "qunica."):
            try:
                return importlib.import_module(prefix + modname)
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            f"cannot import {modname!r}: expected qunica/classifiers/{modname}.py "
            f"(or qunica/{modname}.py) under the project root {ROOT}")

    kmod = load("KPGMC_Low_Rank")
    rmod = load("PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank")
    KPGM = getattr(kmod, "KPGM")
    RcPGM = getattr(rmod, "PGMHQC_gpu_cpu_dtype")
    out = {"kpgm": lambda c: KPGM(n_copies=c, encoding="amplit"),
           "rcpgm": lambda c: RcPGM(n_copies=c, encoding="amplit")}
    if include_cpgm:
        cmod = load("PGMHQC_gpu_cpu_dtype")
        CPGM = getattr(cmod, "PGMHQC_gpu_cpu_dtype")
        out["cpgm"] = lambda c: CPGM(n_copies=c, encoding="amplit")
    return out


def rc_preprocessing_time(RcFactory, X_train, c, reps):
    """Standalone timing of the Rc-PGM reduction pipeline (basis + factors +
    encoding + mapping), replicated outside fit so it can be reported
    separately. The symmetric basis is built on the encoded dimension
    (self.d = d_raw + 1 when an encoding is used)."""
    est = RcFactory(c)

    def basis():
        est.d = X_train.shape[1] + 1  # encoded dimension (amplit)
        est.occupation_numbers = sorted(est._enumerate_occupation_numbers(), reverse=True)
        est.dsym = len(est.occupation_numbers)
        est.multinomial_factors = est._calculate_multinomial_factors()
        return est.dsym

    def mapping():
        Xp = est.X_prime_func(X_train, X_train.shape[0]) if hasattr(est, "X_prime_func") \
            else sys.modules[type(est).__module__].X_prime_func(est, X_train, X_train.shape[0])
        return est.map_batch_efficiently(Xp)

    t_basis = timed(basis, reps)[:2]
    t_map = timed(mapping, reps)[:2]
    return t_basis, t_map


# ----------------------------------------------------------------------------
# Core measurement of one (classifier, dataset) cell
# ----------------------------------------------------------------------------
@dataclass
class Cell:
    method: str
    dataset: str
    N: int
    d: int
    c: int
    l: int
    fit_mean: float = 0.0
    fit_std: float = 0.0
    pred_mean: float = 0.0
    pred_std: float = 0.0
    fit_rss_peak: int = 0
    pred_rss_peak: int = 0
    model_bytes: int = 0
    accuracy: float = float("nan")
    extra: dict = field(default_factory=dict)


def measure(method, factory, c, Xtr, ytr, Xte, yte, dataset, reps):
    cell = Cell(method, dataset, len(Xtr), Xtr.shape[1], c, len(np.unique(ytr)))

    est_holder = {}

    def do_fit():
        est = factory(c)
        est.fit(Xtr, ytr)
        est_holder["est"] = est
        return est

    with PeakRSS() as mem:
        cell.fit_mean, cell.fit_std, est, _ = timed(do_fit, reps)
    cell.fit_rss_peak = mem.delta
    cell.model_bytes = model_bytes(est)

    def do_pred():
        return est.predict(Xte)

    with PeakRSS() as mem:
        cell.pred_mean, cell.pred_std, yhat, _ = timed(do_pred, reps)
    cell.pred_rss_peak = mem.delta
    cell.accuracy = float(np.mean(yhat == yte))
    est_holder["yhat"] = yhat
    return cell, est_holder


# ----------------------------------------------------------------------------
# Component A
# ----------------------------------------------------------------------------
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

        est_bytes = 3 * len(np.unique(y)) * ds * ds * 8  # rough Rc footprint
        if est_bytes > args.mem_limit_gb * 1e9:
            print(f"[skip] {name}: estimated Rc-PGM footprint "
                  f"{est_bytes/1e9:.1f} GB > --mem-limit-gb {args.mem_limit_gb}")
            continue

        print(f"\n=== {name}: N={len(tr)}, d={X.shape[1]}, c={c}, "
              f"l={len(np.unique(y))}, dsym={ds} ===")
        preds = {}
        for method in (["kpgm", "rcpgm"] + (["cpgm"] if args.include_cpgm else [])):
            if method == "cpgm" and (X.shape[1] + 1) ** c > args.cpgm_dim_limit:
                print(f"  [skip] c-PGM: d_enc^c = {(X.shape[1]+1)**c} too large")
                continue
            cell, holder = measure(method, factories[method], c,
                                   Xtr, ytr, Xte, yte, name, args.reps)
            preds[method] = holder["yhat"]
            print(f"  {method:6s} fit {cell.fit_mean*1e3:9.2f} ± {cell.fit_std*1e3:7.2f} ms | "
                  f"pred {cell.pred_mean*1e3:8.2f} ± {cell.pred_std*1e3:6.2f} ms | "
                  f"peak ΔRSS fit {cell.fit_rss_peak/1e6:8.1f} MB | "
                  f"model {cell.model_bytes/1e6:8.1f} MB | acc {cell.accuracy:.3f}")
            rows.append(cell)

        # Rc-PGM preprocessing, timed standalone
        (tb, sb), (tm, sm) = rc_preprocessing_time(factories["rcpgm"], Xtr, c, args.reps)
        print(f"  rcpgm  preprocessing: basis+factors {tb*1e3:.2f} ± {sb*1e3:.2f} ms | "
              f"encode+map {tm*1e3:.2f} ± {sm*1e3:.2f} ms")

        # Equivalence check
        if "kpgm" in preds and "rcpgm" in preds:
            agree = float(np.mean(preds["kpgm"] == preds["rcpgm"]))
            status = "OK" if agree > 0.999 else "FAILED (k-PGM and Rc-PGM disagree)"
            print(f"  equivalence k-PGM vs Rc-PGM: argmax agreement {agree:.4f}  [{status}]")
            meta.append({"dataset": name, "argmax_agreement": agree,
                         "rc_prep_basis_s": tb, "rc_prep_map_s": tm, **info})

    write_rows(OUT_DIR / "componentA.csv", rows)
    (OUT_DIR / "componentA_meta.json").write_text(json.dumps(
        {"env": env_info(), "meta": meta}, indent=2))
    print(f"\n[A] results -> {OUT_DIR/'componentA.csv'}")


# ----------------------------------------------------------------------------
# Component B
# ----------------------------------------------------------------------------
def component_B(args):
    factories = get_classifiers()
    X, y = load_dataset("skin")
    d_raw, c, l = X.shape[1], SKIN_C, len(np.unique(y))
    rng_split = stratified_subsample(y, SKIN_TEST, args.seed)
    mask = np.zeros(len(y), bool); mask[rng_split] = True
    Xte, yte = X[mask], y[mask]
    Xpool, ypool = X[~mask], y[~mask]

    info = thresholds(max(DEFAULT_SWEEP), d_raw, c, l)
    print(f"[B] Skin: d={d_raw}, c={c}, l={l}, dsym={info['dsym']}  |  theoretical "
          f"thresholds: train-time N*≈{info['tr_time_thr']:.0f}, "
          f"train-mem N*={info['tr_mem_thr']}, pred N*≈{info['pred_thr']:.0f}")

    grid = [n for n in (args.sweep or DEFAULT_SWEEP) if n <= args.nmax]
    rows = []
    for n in grid:
        idx = stratified_subsample(ypool, n, args.seed + n)
        Xtr, ytr = Xpool[idx], ypool[idx]
        print(f"\n--- N = {len(idx)} ---")
        for method in ("kpgm", "rcpgm"):
            cell, _ = measure(method, factories[method], c,
                              Xtr, ytr, Xte, yte, f"skin_N{len(idx)}", args.reps)
            print(f"  {method:6s} fit {cell.fit_mean:9.4f} ± {cell.fit_std:7.4f} s | "
                  f"pred {cell.pred_mean:8.4f} ± {cell.pred_std:6.4f} s | "
                  f"peak ΔRSS fit {cell.fit_rss_peak/1e6:8.1f} MB | "
                  f"model {cell.model_bytes/1e6:8.1f} MB | acc {cell.accuracy:.4f}")
            rows.append(cell)

    write_rows(OUT_DIR / "componentB.csv", rows)
    (OUT_DIR / "componentB_meta.json").write_text(json.dumps(
        {"env": env_info(), "thresholds": info, "grid": grid,
         "test_size": int(mask.sum())}, indent=2))
    print(f"\n[B] results -> {OUT_DIR/'componentB.csv'}")
    try:
        plot_B(rows, info)
    except Exception as exc:  # matplotlib optional
        print(f"[B] plot skipped ({exc}); use componentB.csv")


def plot_B(rows, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(method, attr):
        pts = sorted([(r.N, getattr(r, attr)) for r in rows if r.method == method])
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    panels = [("fit_mean", "training time [s]", info["tr_time_thr"]),
              ("model_bytes", "model memory [bytes]", info["tr_mem_thr"]),
              ("pred_mean", "prediction time [s]", info["pred_thr"])]
    for ax, (attr, label, thr) in zip(axes, panels):
        for method, color in (("kpgm", "#C00000"), ("rcpgm", "#1E7B34")):
            xs, ys = series(method, attr)
            ax.plot(xs, ys, "o-", color=color, label=method)
        ax.axvline(thr, color="#1F4E79", ls="--", lw=1,
                   label=f"theoretical N*≈{thr:.0f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N (training samples)"); ax.set_ylabel(label)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Component B — Skin segmentation, c = %d (dsym = %d)"
                 % (SKIN_C, info["dsym"]))
    fig.tight_layout()
    out = OUT_DIR / "componentB_crossover.png"
    fig.savefig(out, dpi=160)
    print(f"[B] figure -> {out}")


# ----------------------------------------------------------------------------
# Self-test: exercises the whole pipeline with numpy mock classifiers
# ----------------------------------------------------------------------------
class _MockPGM:
    """Numpy stand-in with the same API (fit/predict) used by the harness."""

    def __init__(self, n_copies=1, flavor="kpgm"):
        self.n_copies, self.flavor = n_copies, flavor

    @staticmethod
    def _enc(X):
        Xe = np.hstack([X, np.ones((len(X), 1))])
        return Xe / np.linalg.norm(Xe, axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.y_ = yi
        self.Xp_ = self._enc(X)
        G = (self.Xp_ @ self.Xp_.T) ** self.n_copies
        lam, E = np.linalg.eigh((G + G.T) / 2)
        keep = lam > 1e-6
        self.E_, self.lam_ = E[:, keep], lam[keep] ** -0.5
        return self

    def predict(self, X):
        W = (self.Xp_ @ self._enc(X).T) ** self.n_copies
        V = self.E_ @ (self.lam_[:, None] * (self.E_.T @ W))
        scores = np.stack([np.sum(V[self.y_ == k] ** 2, axis=0)
                           for k in range(len(self.classes_))])
        return self.classes_[np.argmax(scores, axis=0)]

    # attributes probed by rc_preprocessing_time
    def _enumerate_occupation_numbers(self):
        from itertools import combinations_with_replacement
        out = []
        for raw in combinations_with_replacement(range(self.d), self.n_copies):
            t = [0] * self.d
            for i in raw:
                t[i] += 1
            out.append(tuple(t))
        return out

    def _calculate_multinomial_factors(self):
        from math import factorial
        return np.array([math.sqrt(factorial(self.n_copies) /
                                   np.prod([factorial(n) for n in t]))
                         for t in self.occupation_numbers])

    def X_prime_func(self, X, m):
        return self._enc(X)

    def map_batch_efficiently(self, Xp):
        return np.stack([f * np.prod(Xp ** np.array(t), axis=1)
                         for t, f in zip(self.occupation_numbers,
                                         self.multinomial_factors)], axis=1)


def selftest(args):
    print("[selftest] synthetic data, numpy mock classifiers (no torch/qunica)")
    rng = np.random.default_rng(0)
    X = rng.random((240, 3)); y = (X @ [1.0, -0.7, 0.4] > 0.35).astype(int)
    tr, te = stratified_split(y, 0.2, 0)
    factories = {"kpgm": lambda c: _MockPGM(c, "kpgm"),
                 "rcpgm": lambda c: _MockPGM(c, "rcpgm")}
    preds = {}
    for m in ("kpgm", "rcpgm"):
        cell, holder = measure(m, factories[m], 3, X[tr], y[tr], X[te], y[te],
                               "synthetic", reps=3)
        preds[m] = holder["yhat"]
        assert cell.fit_mean > 0 and cell.model_bytes > 0
        print(f"  {m}: fit {cell.fit_mean*1e3:.2f} ms, pred {cell.pred_mean*1e3:.2f} ms, "
              f"model {cell.model_bytes/1e3:.1f} kB, acc {cell.accuracy:.3f}")
    agree = float(np.mean(preds["kpgm"] == preds["rcpgm"]))
    (tb, sb), (tm, sm) = rc_preprocessing_time(factories["rcpgm"], X[tr], 3, 3)
    print(f"  preprocessing basis {tb*1e3:.3f} ms, mapping {tm*1e3:.3f} ms")
    print(f"  argmax agreement (mock): {agree:.3f}")
    th = thresholds(500, 3, 5, 2)
    assert th["dsym"] == comb(8, 5) == 56 and th["tr_mem_thr"] == 3136
    write_rows(OUT_DIR / "selftest.csv",
               [Cell("kpgm", "synthetic", len(tr), 3, 3, 2)])
    print("[selftest] harness OK; thresholds OK (dsym=56, N*_mem=3136); "
          f"CSV written under {OUT_DIR}/")


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "dataset", "N", "d", "c", "l", "fit_mean", "fit_std",
              "pred_mean", "pred_std", "fit_rss_peak", "pred_rss_peak",
              "model_bytes", "accuracy"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "threads": N_THREADS,
            "numpy": np.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    return info


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("component", choices=["A", "B", "all", "selftest"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_A,
                    choices=sorted(TABLE7_C))
    ap.add_argument("--include-cpgm", action="store_true",
                    help="also run the explicit c-PGM where d_enc^c is small")
    ap.add_argument("--cpgm-dim-limit", type=int, default=4100)
    ap.add_argument("--mem-limit-gb", type=float, default=8.0)
    ap.add_argument("--nmax", type=int, default=8000)
    ap.add_argument("--sweep", nargs="+", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print(f"[env] {json.dumps(env_info())}")
    if args.component == "selftest":
        selftest(args)
    elif args.component == "A":
        component_A(args)
    elif args.component == "B":
        component_B(args)
    else:
        component_A(args)
        component_B(args)


if __name__ == "__main__":
    main()
