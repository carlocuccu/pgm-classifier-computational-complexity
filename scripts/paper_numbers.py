#!/usr/bin/env python3
"""Regenerate every number quoted in the "Runtime and memory benchmarks" subsection.

The subsection reports a table of measurements plus a number of derived
quantities (ratios, fitted scaling exponents, empirical crossovers). Recomputing
those by hand after every run is where transcription errors come from, so this
script derives all of them from the files the harness writes:

    results_benchmark/componentA.csv   componentA_meta.json
    results_benchmark/componentB.csv   componentB_meta.json

Usage:

    python scripts/paper_numbers.py                 # report + LaTeX table body
    python scripts/paper_numbers.py --results DIR   # read another results folder

Every printed figure is labelled with the sentence of the subsection it belongs
to, so the prose can be checked line by line against a fresh run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ORDER = ["balance-scale", "haberman", "iris", "led7", "ecoli", "car"]


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} $\\mu$s"
    if seconds < 1.0:
        value = seconds * 1e3
        return f"{value:.1f} ms" if value < 100 else f"{value:.0f} ms"
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.0f} s"


def fmt_bytes(n: float) -> str:
    if n < 1e3:
        return f"{n:.0f} B"
    if n < 1e6:
        value = n / 1e3
        return f"{value:.1f} kB" if value < 100 else f"{value:.0f} kB"
    if n < 1e9:
        return f"{n / 1e6:.2f} MB" if n < 1e7 else f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.2f} GB"


def bold(text: str, is_best: bool) -> str:
    return f"\\textbf{{{text}}}" if is_best else text


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(results: Path):
    a = {
        (r["method"], r["dataset"]): r
        for r in csv.DictReader((results / "componentA.csv").open())
    }
    a_meta = json.loads((results / "componentA_meta.json").read_text())
    b = list(csv.DictReader((results / "componentB.csv").open()))
    b_meta = json.loads((results / "componentB_meta.json").read_text())
    return a, a_meta, b, b_meta


# --------------------------------------------------------------------------
# Component A
# --------------------------------------------------------------------------
def component_a_table(a, a_meta) -> str:
    dsym = {m["dataset"]: m["dsym"] for m in a_meta["meta"]}
    lines = []
    for name in ORDER:
        if ("kpgm", name) not in a:
            continue
        k, r = a[("kpgm", name)], a[("rcpgm", name)]
        cells = []
        for field, formatter in (
            ("fit_mean", fmt_time),
            ("pred_mean", fmt_time),
            ("model_bytes", fmt_bytes),
        ):
            kv, rv = float(k[field]), float(r[field])
            cells.append(bold(formatter(kv), kv < rv))
            cells.append(bold(formatter(rv), rv < kv))
        lines.append(
            f"    {name.replace('_', ' ')} & {k['N']} & {dsym[name]} & "
            + " & ".join(cells)
            + r" \\"
        )
    return "\n".join(lines)


def component_a_report(a, a_meta) -> None:
    print("=" * 78)
    print("COMPONENT A")
    print("=" * 78)

    worst = 0.0
    for row in a.values():
        for mean, std in (("fit_mean", "fit_std"), ("pred_mean", "pred_std")):
            worst = max(worst, float(row[std]) / float(row[mean]))
    print(f"\nCaption: largest std/mean over all cells = {100 * worst:.1f}%")

    print("\nRatios Rc-PGM / k-PGM (>1 means the Rc-PGM is the more expensive):")
    print(
        f"  {'dataset':<16}{'training':>12}{'prediction':>12}{'model mem':>12}"
        f"{'orders (train)':>16}"
    )
    for name in ORDER:
        if ("kpgm", name) not in a:
            continue
        k, r = a[("kpgm", name)], a[("rcpgm", name)]
        ratios = [
            float(r[f]) / float(k[f]) for f in ("fit_mean", "pred_mean", "model_bytes")
        ]
        print(
            f"  {name:<16}"
            + "".join(f"{x:>12.4g}" for x in ratios)
            + f"{np.log10(ratios[0]):>+16.2f}"
        )

    print("\nEquivalence check (argmax agreement k-PGM vs Rc-PGM):")
    for m in a_meta["meta"]:
        print(f"  {m['dataset']:<16}{m['argmax_agreement']:.6f}")

    env = a_meta["env"]
    print(
        f"\nEnvironment: python {env['python']}, numpy {env['numpy']}, "
        f"torch {env.get('torch')}, threads {env['threads']}"
    )
    print(
        f"Threshold convention: base_tol={env.get('base_tol')}, "
        f"harmonize_tol={env.get('harmonize_tol')}"
    )

    print("\nLaTeX body of the measurement table:\n")
    print(component_a_table(a, a_meta))


# --------------------------------------------------------------------------
# Component B
# --------------------------------------------------------------------------
def log_slope(N, values, first: int = 0) -> float:
    x, y = np.log(np.asarray(N, float)), np.log(np.asarray(values, float))
    return float(np.polyfit(x[first:], y[first:], 1)[0])


def crossover(N, series_k, series_r) -> float | None:
    """First N at which the k-PGM curve rises above the Rc-PGM one (log-log)."""
    N = np.asarray(N, float)
    diff = np.log(np.asarray(series_k, float)) - np.log(np.asarray(series_r, float))
    for i in range(len(N) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(np.exp(np.log(N[i]) + t * (np.log(N[i + 1]) - np.log(N[i]))))
    return None


def component_b_report(b, b_meta) -> None:
    print()
    print("=" * 78)
    print("COMPONENT B")
    print("=" * 78)

    k = [r for r in b if r["method"] == "kpgm"]
    r = [r for r in b if r["method"] == "rcpgm"]
    N = [float(x["N"]) for x in k]
    half = len(N) // 2

    series = {
        "training time": ("fit_mean", None),
        "prediction time": ("pred_mean", None),
        "model memory": ("model_bytes", None),
    }

    print("\nFitted log-log slopes (exponent of N):")
    print(
        f"  {'quantity':<18}{'k-PGM full':>12}{'k-PGM upper':>13}"
        f"{'Rc-PGM full':>13}{'Rc-PGM upper':>14}"
    )
    for label, (field, _) in series.items():
        kv = [float(x[field]) for x in k]
        rv = [float(x[field]) for x in r]
        print(
            f"  {label:<18}{log_slope(N, kv):>12.2f}{log_slope(N, kv, half):>13.2f}"
            f"{log_slope(N, rv):>13.2f}{log_slope(N, rv, half):>14.2f}"
        )
    print(
        "  ('upper' = fitted over the upper half of the sweep; quote the range you use)"
    )

    thr = b_meta["thresholds"]
    print("\nCrossovers (N at which the k-PGM becomes the more expensive):")
    print(f"  {'quantity':<18}{'empirical':>12}{'theoretical':>14}  condition")
    pairs = [
        ("training time", "fit_mean", thr["tr_time_thr"], "N > l^(1/3) dsym"),
        (
            "model memory",
            "model_bytes",
            thr["pred_thr"],
            "N > l dsym^2 / (d + r_{G^c})   [prediction condition]",
        ),
        (
            "prediction time",
            "pred_mean",
            thr["pred_thr"],
            "N > l dsym^2 / (d + r_{G^c})",
        ),
    ]
    for label, field, threshold, condition in pairs:
        kv = [float(x[field]) for x in k]
        rv = [float(x[field]) for x in r]
        emp = crossover(N, kv, rv)
        emp_text = f"{emp:.0f}" if emp else "not in range"
        print(f"  {label:<18}{emp_text:>12}{threshold:>14.0f}  {condition}")
    print(
        f"\n  Training-memory condition N > dsym^2 = {thr['tr_mem_thr']}: "
        "not probed by the stored-model measurement."
    )

    print(
        "\nPeak RSS during training (the quantity the training-memory "
        "condition bounds):"
    )
    print(f"  {'N':>6}{'k-PGM':>14}{'Rc-PGM':>14}")
    for a_row, b_row in zip(k, r, strict=True):
        print(
            f"  {a_row['N']:>6}{fmt_bytes(float(a_row['fit_rss_peak'])):>14}"
            f"{fmt_bytes(float(b_row['fit_rss_peak'])):>14}"
        )

    last_k, last_r = k[-1], r[-1]
    ratio = float(last_k["fit_mean"]) / float(last_r["fit_mean"])
    print(
        f"\nAt N = {last_k['N']}: k-PGM training {float(last_k['fit_mean']):.0f} s "
        f"against {float(last_r['fit_mean']):.2f} s for the Rc-PGM, "
        f"a factor of {ratio:.0f}."
    )

    same = all(
        a_row["accuracy"] == b_row["accuracy"]
        for a_row, b_row in zip(k, r, strict=True)
    )
    print(f"Accuracies identical at every N: {same}")

    env = b_meta["env"]
    print(
        f"\nEnvironment: python {env['python']}, numpy {env['numpy']}, "
        f"torch {env.get('torch')}, threads {env['threads']}, "
        f"base_tol={env.get('base_tol')}, harmonize_tol={env.get('harmonize_tol')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        default=str(ROOT / "results_benchmark"),
        help="folder holding the harness output",
    )
    args = parser.parse_args()

    results = Path(args.results)
    a, a_meta, b, b_meta = load(results)
    component_a_report(a, a_meta)
    component_b_report(b, b_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
