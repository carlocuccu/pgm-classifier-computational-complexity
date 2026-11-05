"""Figure 4: the crossover sweep, and its redrawing from a deposited run.

One figure per copy number of the sweep. The crossover marked on it is the one
of `crossovers.py`, between the dense k-PGM and the Rc-PGM; the k-PGM from
explicit features is drawn alongside.
"""

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


def plot_B(rows, info, outdir=None, suffix=""):
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
        """Where the dense k-PGM becomes the more expensive (crossovers.py)."""
        from pgm_complexity.crossovers import empirical_crossover

        xs, yk = series("kpgm", attr)
        _, yr = series("rcpgm", attr)
        return empirical_crossover(xs, yk, yr)[0]

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
        for method, name, color, style in (
            ("kpgm", "k-PGM (dense)", "#C00000", "o-"),
            ("kpgm_features", "k-PGM (explicit features)", "#E69F00", "s--"),
            ("rcpgm", "Rc-PGM", "#1E7B34", "o-"),
        ):
            xs, ys = series(method, attr)
            if xs:
                ax.plot(xs, ys, style, color=color, label=name)
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
        _log_ticks(ax.xaxis, *ax.get_xlim())
        _log_ticks(ax.yaxis, *ax.get_ylim())
        ax.legend(fontsize=8)
        ax.grid(which="major", alpha=0.3)
        ax.grid(which="minor", alpha=0.15, lw=0.5)
    c_values = sorted({r.c for r in rows})
    if c_values:
        fig.suptitle(f"c = {c_values[0]}, $d_{{sym}}$ = {info['dsym']}", fontsize=10)
    fig.tight_layout()
    outdir = OUT_DIR if outdir is None else outdir
    for ext in ("png", "pdf"):
        out = outdir / f"componentB_crossover{suffix}.{ext}"
        fig.savefig(out, dpi=160)
        print(f"[B] figure -> {out}")


def plot_B_grid(groups, outdir=None):
    """All the copy numbers of a sweep on one page: a row per c.

    `groups` is a list of (c, rows, info) in increasing c. The three panels of
    each row are those of `plot_B`, drawn at the size of a printed page
    (a 372 pt text width) with one legend for the whole figure, so that the
    sweep can be read in full in an appendix. Writes
    `componentB_crossover_all.{pdf,png}` and nothing else.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from pgm_complexity.crossovers import empirical_crossover

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    methods = (
        ("kpgm", "k-PGM (dense)", "#C00000", "o-"),
        ("kpgm_features", "k-PGM (explicit features)", "#E69F00", "s--"),
        ("rcpgm", "Rc-PGM", "#1E7B34", "o-"),
    )
    columns = (
        ("fit_mean", "training time [s]", "tr_time_thr"),
        ("model_bytes", "stored model [bytes]", "pred_thr"),
        ("pred_mean", "prediction time [s]", "pred_thr"),
    )

    nrows = len(groups)
    fig, axes = plt.subplots(nrows, 3, figsize=(5.15, 1.0 * nrows + 0.4), squeeze=False)
    for row, (c, rows, info) in enumerate(groups):

        def series(method, attr, rows=rows):
            pts = sorted((r.N, getattr(r, attr)) for r in rows if r.method == method)
            return [p[0] for p in pts], [p[1] for p in pts]

        for col, (attr, label, thr_key) in enumerate(columns):
            ax = axes[row][col]
            for method, _name, color, style in methods:
                xs, ys = series(method, attr)
                if xs:
                    ax.plot(xs, ys, style, color=color, ms=2, lw=0.8)
            ax.axvline(info[thr_key], color="#1F4E79", ls="--", lw=0.7)
            xs, yk = series("kpgm", attr)
            _, yr = series("rcpgm", attr)
            emp = empirical_crossover(xs, yk, yr)[0] if xs else None
            if emp is not None:
                ax.axvline(emp, color="#1F4E79", ls=":", lw=0.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            ax.grid(which="major", alpha=0.3, lw=0.4)
            if row == 0:
                ax.set_title(label, fontsize=6.5, pad=3)
            if row == nrows - 1:
                ax.set_xlabel("N (training samples)", fontsize=6, labelpad=1)
            if col == 0:
                ax.set_ylabel(
                    f"c = {c}, $d_{{sym}}$ = {info['dsym']}", fontsize=6, labelpad=1
                )

    handles = [
        Line2D([], [], color=color, ls=style[1:], marker=style[0], ms=2, lw=0.8)
        for _m, _n, color, style in methods
    ] + [
        Line2D([], [], color="#1F4E79", ls="--", lw=0.7),
        Line2D([], [], color="#1F4E79", ls=":", lw=0.9),
    ]
    labels = [name for _m, name, _c, _s in methods] + [
        "threshold",
        "empirical crossover",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=5.5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.tight_layout(
        rect=(0, 0, 1, 1 - 0.22 / fig.get_figheight()), h_pad=0.4, w_pad=0.6
    )
    outdir = OUT_DIR if outdir is None else outdir
    for ext in ("png", "pdf"):
        out = outdir / f"componentB_crossover_all.{ext}"
        fig.savefig(out, dpi=200)
        print(f"[B] figure -> {out}")
    plt.close(fig)


def replot_B(outdir=None, combined=False):
    """Redraw the Component B figures from the deposited CSV, without measuring.

    The figures are a rendering of `componentB.csv`; regenerating them after a
    change to the axes or the styling does not require re-running the sweep,
    and leaves the measurements untouched. Runs with a single copy number
    (up to v1.0.x) give one figure without suffix; runs over several copy
    numbers give one figure per c. With `combined`, only the one-page figure
    with every copy number is drawn (`plot_B_grid`), and the per-c figures
    are left as they are.
    """
    import csv as _csv

    outdir = OUT_DIR if outdir is None else outdir
    csv_path = outdir / "componentB.csv"
    meta_path = outdir / "componentB_meta.json"
    if not csv_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"{csv_path.name} and {meta_path.name} are both needed; "
            f"run `pgm bench b` first."
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

    thresholds = json.loads(meta_path.read_text())["thresholds"]
    if combined:
        if "dsym" in thresholds:
            raise SystemExit("--combined needs a run over several copy numbers.")
        groups = [
            (int(key), [r for r in rows if r.c == int(key)], info)
            for key, info in sorted(thresholds.items(), key=lambda kv: int(kv[0]))
        ]
        plot_B_grid(groups, outdir)
    elif "dsym" in thresholds:  # a single copy number
        plot_B(rows, thresholds, outdir)
    else:
        for key, info in thresholds.items():
            c = int(key)
            plot_B([r for r in rows if r.c == c], info, outdir, suffix=f"_c{c}")
    print(
        f"[replot] redrawn from {csv_path} ({len(rows)} rows); "
        f"the measurements were not touched."
    )
    return 0
