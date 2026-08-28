"""Figure 4: the crossover sweep, and its redrawing from a deposited run."""

from __future__ import annotations

import json
import math

from pgm_complexity import OUT_DIR
from pgm_complexity.bench.measure import Cell


def _log_ticks(axis, lo, hi):
    """Make a logarithmic axis readable between `lo` and `hi`.

    A panel that spans one or two decades gets a single label from the
    default decade locator, which is not enough to read a plateau or a
    crossing off the plot. Such an axis is labelled at 1, 2 and 5 times each
    power of ten; an axis spanning more than three decades keeps the decade
    labels, which are already plentiful. Both get unlabelled minor ticks at
    every 2..9, so intermediate values can be interpolated by eye.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    decades = math.log10(hi) - math.log10(lo) if lo > 0 and hi > lo else 0.0
    subs = (1.0,) if decades > 3.0 else (1.0, 2.0, 5.0)

    def fmt(value, _pos):
        if value <= 0:
            return ""
        if 1e-3 <= value < 1e4:
            text = f"{value:.10g}"
            return text
        exponent = int(round(math.log10(value)))
        mantissa = value / 10.0**exponent
        if abs(mantissa - 1.0) < 1e-9:
            return f"$10^{{{exponent}}}$"
        return f"${mantissa:.10g}\\times10^{{{exponent}}}$"

    axis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=32))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
    )
    axis.set_minor_formatter(NullFormatter())


def plot_B(rows, info):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # TrueType rather than Type 3 in the PDF: most publishers require it.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    def series(method, attr):
        pts = sorted([(r.N, getattr(r, attr)) for r in rows if r.method == method])
        return [p[0] for p in pts], [p[1] for p in pts]

    def crossing(attr):
        """First N at which the k-PGM curve rises above the Rc-PGM one."""
        xs, yk = series("kpgm", attr)
        _, yr = series("rcpgm", attr)
        diff = [math.log(a) - math.log(b) for a, b in zip(yk, yr, strict=True)]
        for i in range(len(xs) - 1):
            if diff[i] < 0 <= diff[i + 1]:
                t = -diff[i] / (diff[i + 1] - diff[i])
                return math.exp(
                    math.log(xs[i]) + t * (math.log(xs[i + 1]) - math.log(xs[i]))
                )
        return None

    # The stored model of the k-PGM is the O(N (d + r_{G^c})) term that governs its
    # PREDICTION cost, so the middle panel is compared with the prediction
    # threshold, not with the training-memory one.
    panels = [
        ("fit_mean", "training time [s]", info["tr_time_thr"]),
        ("model_bytes", "stored model memory [bytes]", info["pred_thr"]),
        ("pred_mean", "prediction time [s]", info["pred_thr"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, (attr, label, thr) in zip(axes, panels, strict=True):
        for method, name, color in (
            ("kpgm", "k-PGM", "#C00000"),
            ("rcpgm", "Rc-PGM", "#1E7B34"),
        ):
            xs, ys = series(method, attr)
            ax.plot(xs, ys, "o-", color=color, label=name)
        ax.axvline(
            thr,
            color="#1F4E79",
            ls="--",
            lw=1,
            label=f"theoretical N*$\\approx${thr:.0f}",
        )
        emp = crossing(attr)
        if emp is not None:
            ax.axvline(
                emp,
                color="#1F4E79",
                ls=":",
                lw=1.2,
                label=f"empirical N$\\approx${emp:.0f}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (training samples)")
        ax.set_ylabel(label)
        _log_ticks(ax.yaxis, *ax.get_ylim())
        ax.legend(fontsize=8)
        ax.grid(which="major", alpha=0.3)
        ax.grid(which="minor", alpha=0.15, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"componentB_crossover.{ext}"
        fig.savefig(out, dpi=160)
        print(f"[B] figure -> {out}")


def replot_B():
    """Redraw the Component B figure from the deposited CSV, without measuring.

    The figure is a rendering of `componentB.csv`; regenerating it after a
    change to the axes or the styling does not require re-running the sweep,
    and leaves the measurements untouched.
    """
    import csv as _csv

    csv_path = OUT_DIR / "componentB.csv"
    meta_path = OUT_DIR / "componentB_meta.json"
    if not csv_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"{csv_path.name} and {meta_path.name} are both needed; "
            f"run `python run_benchmarks.py B` first."
        )

    numeric = {"N", "d", "c", "l", "fit_rss_peak", "pred_rss_peak", "model_bytes"}
    rows = []
    with csv_path.open() as handle:
        for record in _csv.DictReader(handle):
            values = {}
            for key, raw in record.items():
                if key in ("method", "dataset"):
                    values[key] = raw
                elif key in numeric:
                    values[key] = int(float(raw))
                else:
                    values[key] = float(raw)
            rows.append(Cell(**values))

    info = json.loads(meta_path.read_text())["thresholds"]
    plot_B(rows, info)
    print(
        f"[replot] redrawn from {csv_path} ({len(rows)} rows); "
        f"the measurements were not touched."
    )
    return 0
