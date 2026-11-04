"""Every figure the empirical section quotes, derived from a deposited run.

Nothing here is typed into the manuscript by hand: the ratios, the fitted
exponents and the crossovers are all read back out of the CSV files, so a
re-run cannot leave a stale number in the text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ORDER = ["balance-scale", "haberman", "iris", "led7", "ecoli", "car"]


def _threshold(env):
    """The pseudo-inverse threshold a run declares. Runs made up to v1.0.1
    record `harmonize_tol = True`: 1e-6 on the spectrum of G^c."""
    if env.get("harmonize_tol"):
        return f"{env.get('base_tol')} on the spectrum of G^c (v1.0.x convention)"
    return f"{env.get('base_tol')} on the spectrum of sigma"


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


def load(results: Path):
    a = {
        (r["method"], r["dataset"]): r
        for r in csv.DictReader((results / "componentA.csv").open())
    }
    a_meta = json.loads((results / "componentA_meta.json").read_text())
    b = list(csv.DictReader((results / "componentB.csv").open()))
    b_meta = json.loads((results / "componentB_meta.json").read_text())
    return a, a_meta, b, b_meta


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
    print(f"Threshold: {_threshold(env)}")

    print("\nLaTeX body of the measurement table:\n")
    print(component_a_table(a, a_meta))


def log_slope(N, values, first: int = 0) -> float:
    x, y = np.log(np.asarray(N, float)), np.log(np.asarray(values, float))
    return float(np.polyfit(x[first:], y[first:], 1)[0])


def _thresholds_by_c(b, b_meta):
    """Thresholds keyed by copy number, for single-c runs (up to v1.0.x)
    and for sweeps over several copy numbers."""
    thr = b_meta["thresholds"]
    if "dsym" in thr:
        return {int(float(b[0]["c"])): thr}
    return {int(key): value for key, value in thr.items()}


def _cells(b, c, method=None):
    rows = [r for r in b if int(float(r["c"])) == c]
    if method is not None:
        rows = [r for r in rows if r["method"] == method]
    return sorted(rows, key=lambda r: float(r["N"]))


def component_b_report(b, b_meta) -> None:
    from pgm_complexity.crossovers import crossover_table, format_table

    print()
    print("=" * 78)
    print("COMPONENT B")
    print("=" * 78)

    infos = _thresholds_by_c(b, b_meta)
    methods = [
        m
        for m in ("kpgm", "kpgm_features", "rcpgm")
        if any(r["method"] == m for r in b)
    ]

    print("\nFitted log-log slopes (exponent of N), full sweep / upper half:")
    print(f"  {'c':>2} {'quantity':<18}" + "".join(f"{m:>18}" for m in methods))
    for c in sorted(infos):
        for label, field in (
            ("training time", "fit_mean"),
            ("prediction time", "pred_mean"),
            ("model memory", "model_bytes"),
        ):
            parts = []
            for m in methods:
                rows = _cells(b, c, m)
                N = [float(x["N"]) for x in rows]
                v = [float(x[field]) for x in rows]
                half = len(N) // 2
                parts.append(
                    f"{log_slope(N, v):>11.2f} / {log_slope(N, v, half):<4.2f}"
                )
            print(f"  {c:>2} {label:<18}" + "".join(parts))
    print(
        "  ('upper' = fitted over the upper half of the sweep; quote the range you use)"
    )

    print("\nCrossovers of the dense k-PGM and the Rc-PGM (N at which the k-PGM")
    print("becomes, and stays, the more expensive), against the thresholds:")
    print(format_table(crossover_table(b, infos)))

    print("\nTraining time at the largest N of the sweep, relative to the Rc-PGM:")
    for c in sorted(infos):
        cells = _cells(b, c)
        n_last = float(cells[-1]["N"])
        last = {
            r["method"]: float(r["fit_mean"]) for r in cells if float(r["N"]) == n_last
        }
        ratios = ", ".join(
            f"{m} {last[m] / last['rcpgm']:.3g}x" for m in methods if m != "rcpgm"
        )
        print(f"  c = {c}, N = {n_last:.0f}: {ratios}")

    same = all(
        len({r["accuracy"] for r in _cells(b, c) if r["N"] == n}) == 1
        for c in infos
        for n in {r["N"] for r in _cells(b, c)}
    )
    print(f"Accuracies identical across the classifiers at every c and N: {same}")

    env = b_meta["env"]
    print(
        f"\nEnvironment: python {env['python']}, numpy {env['numpy']}, "
        f"torch {env.get('torch')}, threads {env['threads']}, "
        f"threshold {_threshold(env)}"
    )


# ---------------------------------------------------------------------------
# Table 7 and the copy-number sweep of the appendix, from `pgm select`
# ---------------------------------------------------------------------------
def load_selection(results: Path):
    """The sweep and the protocol written by `pgm select`."""
    sweep = list(csv.DictReader((results / "selection_sweep.csv").open()))
    meta = json.loads((results / "selection_meta.json").read_text())
    return sweep, meta


def _by_dataset(sweep):
    out: dict[str, list[dict]] = {}
    for row in sweep:
        out.setdefault(row["dataset"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["c"]))
    return dict(sorted(out.items()))


def _flag(value: str) -> str:
    return "True" if value == "True" else "False"


def table7_rows(sweep) -> str:
    """LaTeX body of Table 7: one row per dataset, at the selected c.

    Columns: dataset, N, d, c, l, d_sym, r_{G^c}, CV accuracy (mean and
    standard deviation over the repetitions), test accuracy, and the
    training-time, training-memory and prediction flags.
    """
    lines = []
    for name, rows in _by_dataset(sweep).items():
        r = next(r for r in rows if r["selected"] == "True")
        lines.append(
            f"{name.replace('_', ' ')} & {r['N']} & {r['d_enc']} & {r['c']} & "
            f"{r['l']} & {r['dsym']} & {r['r_Gc']} & "
            f"{float(r['cv_mean']):.3f} $\\pm$ {float(r['cv_std']):.3f} & "
            f"{float(r['test_acc']):.3f} & {_flag(r['win_tr_time'])} & "
            f"{_flag(r['win_tr_mem'])} & {_flag(r['win_pred'])} \\\\"
        )
    return "\n".join(lines)


def sweep_accuracy_rows(sweep) -> str:
    """LaTeX body of the appendix accuracy table: the mean CV accuracy of
    every dataset at every c, the selected c in boldface."""
    lines = []
    for name, rows in _by_dataset(sweep).items():
        cells = [
            bold(f"{float(r['cv_mean']):.3f}", r["selected"] == "True") for r in rows
        ]
        lines.append(f"{name.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def sweep_flag_rows(sweep) -> str:
    """LaTeX body of the appendix flag table: at every c, the three
    conditions as a triple of T/F (training time, training memory,
    prediction), each evaluated with the r_{G^c} of that c."""

    def tf(value: str) -> str:
        return "T" if value == "True" else "F"

    lines = []
    for name, rows in _by_dataset(sweep).items():
        cells = [
            bold(
                tf(r["win_tr_time"]) + tf(r["win_tr_mem"]) + tf(r["win_pred"]),
                r["selected"] == "True",
            )
            for r in rows
        ]
        lines.append(f"{name.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def selection_report(sweep, meta) -> None:
    copies = meta["copies"]
    print(
        f"% pgm select: {meta['folds']}-fold CV x {meta['repeats']} repetitions on "
        f"the training set, split seed {meta['split_seed']}, c = "
        f"{copies[0]}..{copies[-1]}; rule: {meta['rule']}; threshold "
        f"{_threshold(meta['env'])}"
    )
    print(
        "\n% Table 7: dataset & N & d & c & l & d_sym & r_Gc & CV acc. & test acc. "
        "& train time & train memory & prediction"
    )
    print(table7_rows(sweep))
    print("\n% Appendix: mean CV accuracy at c = " + ", ".join(map(str, copies)))
    print(sweep_accuracy_rows(sweep))
    print(
        "\n% Appendix: conditions at c = "
        + ", ".join(map(str, copies))
        + " (training time, training memory, prediction)"
    )
    print(sweep_flag_rows(sweep))
    edge = [
        name
        for name, rows in _by_dataset(sweep).items()
        if any(r["selected"] == "True" and int(r["c"]) == copies[-1] for r in rows)
    ]
    if edge:
        print(f"\n% selected c at the last value of the grid: {', '.join(edge)}")
