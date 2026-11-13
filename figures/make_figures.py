"""Draw Figures 1-3 of the paper, the three analytic figures.

They plot closed-form expressions only; no data and no measurement enter them.

  pgm_vs_kpgm           Figure 1: N < d^(2c) and N > d^(2c)
  reduced               Figure 2: d^c against d_sym = binomial(d + c - 1, c)
  pgm_vs_kpgm_vs_rcpgm  Figure 3: N < d_sym^2, d_sym^2 < N < d^(2c), N > d^(2c)

Each classifier keeps one colour in the three figures: pink for the k-PGM, blue
for the c-PGM and green for the Rc-PGM. The legends state the conditions only.

Run from the repository root:

    uv run python figures/make_figures.py

The figures are written as PDF next to this script.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator

OUT = Path(__file__).resolve().parent

PINK = "#e2a6b4"  # k-PGM
BLUE = "#a9bfe0"  # c-PGM
GREEN = "#a7c79a"  # Rc-PGM
EDGE = (0, 0, 0, 0.05)

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,  # embed editable TrueType text
        "ps.fonttype": 42,
    }
)

_lgamma = np.vectorize(math.lgamma)


def log10_dsym(d, c):
    """log10 of binomial(d + c - 1, c), continued to real c through Gamma."""
    return (_lgamma(d + c) - _lgamma(c + 1.0) - _lgamma(d)) / math.log(10.0)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _zlog_ticks(ax, lo, hi):
    ks = list(range(lo, hi + 1))
    ax.set_zlim(lo, hi)
    ax.zaxis.set_major_locator(FixedLocator(ks))
    ax.zaxis.set_major_formatter(FixedFormatter([f"$10^{{{k}}}$" for k in ks]))


def _style3d(ax, zlabel, zlo, zhi, dticks):
    ax.set_xlabel("$d$", labelpad=12)
    ax.set_ylabel("$c$", labelpad=10)
    ax.set_zlabel(zlabel, labelpad=16, rotation=90)
    _zlog_ticks(ax, zlo, zhi)
    ax.set_xticks(dticks)
    ax.set_yticks([2, 3, 4, 5, 6])
    ax.view_init(elev=14, azim=-128)


def _legend(fig, entries, fontsize=11):
    handles = [Patch(facecolor=colour, label=label) for colour, label in entries]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.97),
        framealpha=1.0,
        edgecolor="0.7",
        fontsize=fontsize,
    ).set_zorder(1000)


def _grid(nd, nc, nz):
    """Voxel corners and cell centres over d in [1, 50], c in [2, 6], log10 N in
    [5, 6]."""
    de = np.linspace(1, 50, nd + 1)
    ce = np.linspace(2, 6, nc + 1)
    ze = np.linspace(5, 6, nz + 1)
    X, Y, Z = np.meshgrid(de, ce, ze, indexing="ij")
    dc = 0.5 * (de[:-1] + de[1:])
    cc = 0.5 * (ce[:-1] + ce[1:])
    zc = 0.5 * (ze[:-1] + ze[1:])
    Dc, Cc, Zc = np.meshgrid(dc, cc, zc, indexing="ij")
    return X, Y, Z, Dc, Cc, Zc


def _voxels(ax, X, Y, Z, mask, colour):
    ax.voxels(X, Y, Z, mask, facecolors=colour + "99", edgecolors=EDGE, linewidth=0.08)


N_LABEL = r"$N$  (training samples)"
D_TICKS = [1, 10, 20, 30, 40, 50]


def fig_pgm_vs_kpgm():
    X, Y, Z, Dc, Cc, Zc = _grid(49, 22, 26)
    hi = 2 * Cc * np.log10(Dc)  # log10(d^(2c))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _voxels(ax, X, Y, Z, Zc <= hi, PINK)
    _voxels(ax, X, Y, Z, Zc > hi, BLUE)
    _style3d(ax, N_LABEL, 5, 6, D_TICKS)
    _legend(fig, [(PINK, r"$N < d^{\,2c}$"), (BLUE, r"$N > d^{\,2c}$")])
    _save(fig, "pgm_vs_kpgm")


def fig_reduced():
    c = np.linspace(2, 6, 140)
    d = np.linspace(2, 20, 140)
    C, D = np.meshgrid(c, d, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Unlike Figures 1 and 3, c runs along the first horizontal axis and d along
    # the second, and the surfaces are seen from the front. The d^c surface lies
    # above the d_sym one and is drawn translucent, so that both stay visible.
    surfaces = ((C * np.log10(D), BLUE, 0.55), (log10_dsym(D, C), GREEN, 0.9))
    for values, colour, alpha in surfaces:
        ax.plot_surface(
            C,
            D,
            values,
            color=colour,
            alpha=alpha,
            edgecolor=(0, 0, 0, 0.12),
            linewidth=0.1,
            rcount=30,
            ccount=30,
            shade=False,
        )
    ax.set_xlabel("$c$", labelpad=10)
    ax.set_ylabel("$d$", labelpad=10)
    ax.set_zlabel("dimension", labelpad=12)
    _zlog_ticks(ax, 1, 8)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_yticks([2, 5, 10, 15, 20])
    ax.view_init(elev=20, azim=-58)
    _legend(
        fig,
        [
            (BLUE, r"$d^{\,c}$"),
            (GREEN, r"$d_{\mathrm{sym}} = \binom{d+c-1}{c}$"),
        ],
    )
    _save(fig, "reduced")


def fig_pgm_vs_kpgm_vs_rcpgm():
    X, Y, Z, Dc, Cc, Zc = _grid(49, 22, 26)
    hi = 2 * Cc * np.log10(Dc)  # log10(d^(2c))
    lo = 2 * log10_dsym(Dc, Cc)  # log10(d_sym^2)
    fig = plt.figure(figsize=(8.4, 6))
    ax = fig.add_subplot(111, projection="3d")
    _voxels(ax, X, Y, Z, Zc <= lo, PINK)
    _voxels(ax, X, Y, Z, (Zc > lo) & (Zc <= hi), GREEN)
    _voxels(ax, X, Y, Z, Zc > hi, BLUE)
    _style3d(ax, N_LABEL, 5, 6, D_TICKS)
    _legend(
        fig,
        [
            (PINK, r"$N < d_{\mathrm{sym}}^{2}$"),
            (GREEN, r"$d_{\mathrm{sym}}^{2} < N < d^{\,2c}$"),
            (BLUE, r"$N > d^{\,2c}$"),
        ],
        fontsize=10.5,
    )
    _save(fig, "pgm_vs_kpgm_vs_rcpgm")


if __name__ == "__main__":
    fig_pgm_vs_kpgm()
    fig_reduced()
    fig_pgm_vs_kpgm_vs_rcpgm()
    print("written to", OUT)
