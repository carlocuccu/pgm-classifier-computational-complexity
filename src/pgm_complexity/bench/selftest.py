"""The pipeline exercised end to end with numpy stand-ins.

No PyTorch, no qunica, seconds rather than hours: it checks the plumbing --
the split, the timing, the memory accounting, the thresholds and the argmax
agreement -- so that a change to the harness is caught without running a
benchmark.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np

from pgm_complexity import OUT_DIR
from pgm_complexity.bench.estimators import rc_preprocessing_time
from pgm_complexity.bench.io import stratified_split, write_rows
from pgm_complexity.bench.measure import Cell, measure
from pgm_complexity.thresholds import thresholds


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
        scores = np.stack(
            [np.sum(V[self.y_ == k] ** 2, axis=0) for k in range(len(self.classes_))]
        )
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

        return np.array(
            [
                math.sqrt(factorial(self.n_copies) / np.prod([factorial(n) for n in t]))
                for t in self.occupation_numbers
            ]
        )

    def X_prime_func(self, X, m):
        return self._enc(X)

    def map_batch_efficiently(self, Xp):
        return np.stack(
            [
                f * np.prod(Xp ** np.array(t), axis=1)
                for t, f in zip(
                    self.occupation_numbers,
                    self.multinomial_factors,
                    strict=True,
                )
            ],
            axis=1,
        )


def selftest(args):
    print("[selftest] synthetic data, numpy mock classifiers (no torch/qunica)")
    rng = np.random.default_rng(0)
    X = rng.random((240, 3))
    y = (X @ [1.0, -0.7, 0.4] > 0.35).astype(int)
    tr, te = stratified_split(y, 0.2, 0)
    factories = {
        "kpgm": lambda c, n_train=None: _MockPGM(c, "kpgm"),
        "rcpgm": lambda c, n_train=None: _MockPGM(c, "rcpgm"),
    }
    preds = {}
    for m in ("kpgm", "rcpgm"):
        cell, holder = measure(
            m, factories[m], 3, X[tr], y[tr], X[te], y[te], "synthetic", reps=3
        )
        preds[m] = holder["yhat"]
        assert cell.fit_mean > 0 and cell.model_bytes > 0
        print(
            f"  {m}: fit {cell.fit_mean * 1e3:.2f} ms, pred {cell.pred_mean * 1e3:.2f} ms, "
            f"model {cell.model_bytes / 1e3:.1f} kB, acc {cell.accuracy:.3f}"
        )
    agree = float(np.mean(preds["kpgm"] == preds["rcpgm"]))
    (tb, sb), (tm, sm) = rc_preprocessing_time(factories["rcpgm"], X[tr], 3, 3)
    print(f"  preprocessing basis {tb * 1e3:.3f} ms, mapping {tm * 1e3:.3f} ms")
    print(f"  argmax agreement (mock): {agree:.3f}")
    th = thresholds(500, 3, 5, 2)
    assert th["dsym"] == comb(8, 5) == 56 and th["tr_mem_thr"] == 3136
    write_rows(OUT_DIR / "selftest.csv", [Cell("kpgm", "synthetic", len(tr), 3, 3, 2)])
    print(
        "[selftest] harness OK; thresholds OK (dsym=56, N*_mem=3136); "
        f"CSV written under {OUT_DIR}/"
    )
