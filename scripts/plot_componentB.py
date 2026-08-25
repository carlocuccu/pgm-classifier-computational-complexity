#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the Component B crossover figure from a completed run.

`run_benchmarks.py B` already writes a PNG at the end of the sweep; this script
redraws the same figure from `componentB.csv` and `componentB_meta.json`, as a
vector PDF suitable for the manuscript (and a PNG alongside it), without
re-running the benchmark.

    python scripts/plot_componentB.py
    python scripts/plot_componentB.py --results results_benchmark --outdir figures

Each panel carries the threshold of the condition that actually governs the
quantity plotted. Training time is governed by `N > l^(1/3) dsym`; prediction
time by `N > l dsym^2 / (d + r_G)`; and so is the *stored model*, because the
memory a fitted k-PGM retains is the O(N(d + r_G)) term that its prediction
cost is made of -- not the O(N^2) Gram matrix, which is transient and released
when `fit` returns. The training-memory condition `N > dsym^2` bounds the peak
working set, not the stored model, and is therefore not the threshold to draw
here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Categorical slots 1 and 2 of the reference palette: validated for
# colour-vision deficiency (worst-case CVD dE 24.7, normal-vision dE 33.6),
# unlike the red/green pair, which is indistinguishable under deuteranopia.
COLOURS = {"kpgm": "#2a78d6", "rcpgm": "#eb6834"}
MARKERS = {"kpgm": "o", "rcpgm": "s"}          # identity survives greyscale
LABELS = {"kpgm": "k-PGM", "rcpgm": "Rc-PGM"}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
THRESHOLD = "#52514e"


def load(results: Path):
    rows = list(csv.DictReader((results / "componentB.csv").open()))
    meta = json.loads((results / "componentB_meta.json").read_text())
    return rows, meta


def series(rows, method, field, scale=1.0):
    pts = sorted((float(r["N"]), float(r[field]) * scale)
                 for r in rows if r["method"] == method)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def crossover(N, y_k, y_r):
    """First N at which the k-PGM curve rises above the Rc-PGM one (log-log)."""
    diff = np.log(y_k) - np.log(y_r)
    for i in range(len(N) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(np.exp(np.log(N[i]) + t * (np.log(N[i + 1]) - np.log(N[i]))))
    return None


def draw(rows, meta, outdir: Path, stem: str = "componentB_crossover"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3
        "savefig.bbox": "tight",
    })

    thr = meta["thresholds"]
    panels = [
        ("fit_mean", 1.0, "training time [s]", thr["tr_time_thr"],
         r"$N^{*}=\sqrt[3]{l}\,d_{\mathrm{sym}}$"),
        ("model_bytes", 1e-3, "stored model [kB]", thr["pred_thr"],
         r"$N^{*}=l\,d_{\mathrm{sym}}^{2}/(d+r_G)$"),
        ("pred_mean", 1e3, "prediction time [ms]", thr["pred_thr"],
         r"$N^{*}=l\,d_{\mathrm{sym}}^{2}/(d+r_G)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))

    for panel_index, (ax, (field, scale, ylabel, threshold, thr_label)) in enumerate(
            zip(axes, panels)):
        N, y_k = series(rows, "kpgm", field, scale)
        _, y_r = series(rows, "rcpgm", field, scale)

        ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        ax.axvline(threshold, color=THRESHOLD, ls="--", lw=1.1, zorder=1,
                   label=f"theoretical {thr_label}")
        xc = crossover(N, y_k, y_r)
        if xc is not None:
            ax.axvline(xc, color=THRESHOLD, ls=":", lw=1.1, zorder=1,
                       label=f"empirical crossover $N\\approx{xc:.0f}$")

        for method, y in (("kpgm", y_k), ("rcpgm", y_r)):
            # The series are named once, in the first panel; the marker shape
            # carries identity in the other two, and in greyscale.
            ax.plot(N, y, marker=MARKERS[method], color=COLOURS[method], lw=2.0,
                    ms=5.5, mew=0, zorder=3,
                    label=LABELS[method] if panel_index == 0 else None)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N$ (training samples)")
        ax.set_ylabel(ylabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.18)
        ax.legend(fontsize=7, loc="upper left", handlelength=1.8,
                  borderaxespad=0.2, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.85)

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        path = outdir / (stem + suffix)
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results_benchmark"),
                        help="folder holding componentB.csv and componentB_meta.json")
    parser.add_argument("--outdir", default=str(ROOT / "figures"),
                        help="where to write the figure (default: figures/)")
    args = parser.parse_args()

    rows, meta = load(Path(args.results))
    for path in draw(rows, meta, Path(args.outdir)):
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
