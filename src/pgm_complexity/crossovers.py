"""Empirical crossovers of Component B, set against the theoretical thresholds.

For every copy number and every measured quantity, the empirical crossover is
the training-set size N at which the dense k-PGM becomes more expensive than
the Rc-PGM and stays so up to the end of the sweep. It is located by linear
interpolation of log(cost_k-PGM / cost_Rc-PGM) in log N between the two sizes
that bracket the last change of sign. The ratio empirical / theoretical is the
implementation constant the thresholds leave out.

Nothing here is read off a figure: the table is recomputed from the CSV of a
run, so it can be checked and regenerated.
"""

from __future__ import annotations

import math

# Measured quantity, label, threshold key it is compared with.
METRICS = (
    ("fit_mean", "training time", "tr_time_thr"),
    ("fit_rss_peak", "training memory (peak RSS)", "tr_mem_thr"),
    ("pred_mean", "prediction time", "pred_thr"),
    ("model_bytes", "stored model", "pred_thr"),
)

BELOW = "before the grid (k-PGM costlier at every N)"
ABOVE = "beyond the grid (k-PGM cheaper at the largest N)"
CROSSED = "crossed"


def empirical_crossover(N, cost_k, cost_r):
    """(N*, status) for two cost series sampled at the sizes ``N``.

    Costs are floored at 1e-12 before taking logarithms, so that a zero
    reading (a peak RSS below the sampling resolution) does not break the
    interpolation.
    """
    pts = sorted(zip(N, cost_k, cost_r, strict=True))
    xs = [math.log(p[0]) for p in pts]
    diff = [math.log(max(k, 1e-12)) - math.log(max(r, 1e-12)) for _, k, r in pts]
    if diff[-1] < 0:
        return None, ABOVE
    if all(v >= 0 for v in diff):
        return None, BELOW
    i = max(j for j in range(len(diff)) if diff[j] < 0)
    t = -diff[i] / (diff[i + 1] - diff[i])
    return math.exp(xs[i] + t * (xs[i + 1] - xs[i])), CROSSED


def crossover_table(rows, thresholds_by_c):
    """One record per copy number and measured quantity.

    ``rows`` are the cells of Component B as mappings (method, N, c and the
    measured fields); ``thresholds_by_c`` maps c to the output of
    ``thresholds()`` for that copy number.
    """
    out = []
    for c in sorted(thresholds_by_c):
        thr = thresholds_by_c[c]
        cells = [r for r in rows if int(float(r["c"])) == c]
        k = sorted((r for r in cells if r["method"] == "kpgm"), key=_n)
        r = sorted((r for r in cells if r["method"] == "rcpgm"), key=_n)
        if not k or not r:
            continue
        N = [_n(x) for x in k]
        for field, label, key in METRICS:
            emp, status = empirical_crossover(
                N, [float(x[field]) for x in k], [float(x[field]) for x in r]
            )
            theo = float(thr[key])
            record = {
                "c": c,
                "dsym": thr["dsym"],
                "quantity": label,
                "theoretical": theo,
                "empirical": emp,
                "status": status,
                "ratio": emp / theo if emp is not None else None,
            }
            if key == "tr_mem_thr" and "tr_mem_thr_class" in thr:
                # The per-class peak of the implementation (runs from v1.1.0).
                record["theoretical_class"] = float(thr["tr_mem_thr_class"])
            out.append(record)
    return out


def format_table(records) -> str:
    lines = [
        f"{'c':>2} {'d_sym':>5}  {'quantity':<27}{'theoretical':>12}"
        f"{'empirical':>12}{'ratio':>8}  note"
    ]
    for rec in records:
        emp = f"{rec['empirical']:.0f}" if rec["empirical"] is not None else "-"
        ratio = f"{rec['ratio']:.2f}" if rec["ratio"] is not None else "-"
        note = "" if rec["status"] == CROSSED else rec["status"]
        if "theoretical_class" in rec:
            extra = f"per-class peak: {rec['theoretical_class']:.0f}"
            note = f"{note}; {extra}" if note else extra
        lines.append(
            f"{rec['c']:>2} {rec['dsym']:>5}  {rec['quantity']:<27}"
            f"{rec['theoretical']:>12.0f}{emp:>12}{ratio:>8}  {note}"
        )
    return "\n".join(lines)


def _n(row) -> int:
    return int(float(row["N"]))
