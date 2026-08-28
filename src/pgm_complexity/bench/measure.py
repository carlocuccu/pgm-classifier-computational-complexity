"""What a single measurement is, and how it is taken.

Two quantities stand for memory, and they answer different questions: the peak
resident-set size over a phase, which includes whatever the allocator did, and
the bytes of the arrays the fitted model actually holds, which is what the
theoretical bounds describe.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np


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
        result = None  # free the previous repetition's model
        gc.collect()  # before the timer, so collection does not pollute it
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std(ddof=1)) if reps > 1 else 0.0, result, times


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
        est = factory(c, n_train=len(Xtr))
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
