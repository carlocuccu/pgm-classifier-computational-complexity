# Figures

The three *analytic* figures of the manuscript, as PDF. They are plots of the
closed-form complexity expressions derived in the paper — no data and no
measurement enter them — and are drawn by `make_figures.py`, with matplotlib:

```bash
uv run python figures/make_figures.py
# -> figures/pgm_vs_kpgm.pdf, reduced.pdf, pgm_vs_kpgm_vs_rcpgm.pdf
```

The exact expressions are listed below, so that each surface can also be
re-derived independently.

Throughout, `N` is the training-set size, `d` the feature dimension, `c` the
number of tensor copies, and

```
d_sym = binomial(d + c - 1, c)
```

is the dimension of the symmetric subspace. Each classifier keeps one colour in
the three figures: pink for the k-PGM, blue for the c-PGM and green for the
Rc-PGM. The legends state the conditions only.

## `pgm_vs_kpgm` — Figure 1

Training-memory condition of the c-PGM against the k-PGM, over
`d ∈ [1, 50]`, `c ∈ [2, 6]`, `N ∈ [10^5, 10^6]`.

| region | colour | inequality |
|---|---|---|
| below the surface | pink (k-PGM) | `N < d^(2c)` |
| above the surface | blue (c-PGM) | `N > d^(2c)` |

The c-PGM stores the `N` density operators of size `d^c × d^c` under the batched
convention of the paper, hence the `d^(2c)` term; the dense k-PGM stores the
`N × N` Gram matrix. `N > d^(2c)` is a sufficient condition: where it holds, the
batched c-PGM needs less training memory than the dense k-PGM; where it does
not, the comparison depends on the implementation (a streaming c-PGM needs less
memory already for `N > sqrt(l) d^c`).

## `reduced` — Figure 2

Growth of the tensor-space dimension `d^c` against the symmetric-subspace
dimension `d_sym`, over `d ∈ [2, 20]` and `c ∈ [2, 6]`, on a base-10
logarithmic vertical axis. `d_sym` is continued to real `c` through the Gamma
function.

| surface | colour | expression |
|---|---|---|
| upper | blue (c-PGM) | `d^c` |
| lower | green (Rc-PGM) | `binomial(d + c - 1, c)` |

The gap between the two surfaces is the reduction the Rc-PGM exploits: every
occurrence of `d^(2c)` in the c-PGM complexities becomes `d_sym^2`.

## `pgm_vs_kpgm_vs_rcpgm` — Figure 3

The same comparison as Figure 1, with the Rc-PGM added, over the same ranges.

| region | colour | inequality |
|---|---|---|
| lowest | pink (k-PGM) | `N < d_sym^2` |
| middle | green (Rc-PGM) | `d_sym^2 < N < d^(2c)` |
| highest | blue (c-PGM) | `N > d^(2c)` |

The regions are those of the two sufficient conditions `N > d_sym^2` and
`N > d^(2c)`, which compare the batched Rc-PGM and c-PGM with the dense k-PGM.
In the middle region the condition holds for the Rc-PGM, thanks to the
dimensionality reduction, but not for the unreduced c-PGM; in the lowest region
neither condition guarantees that the Rc-PGM or the c-PGM needs less training
memory than the dense k-PGM. A k-PGM that exploits the rank of the Gram matrix
is not described by the figure.

## Note on the encoded dimension

As everywhere in the paper, `d` is the dimension of the *encoded* space, the one
the classifier operates in — not the number of raw features of a dataset. The
amplitude encoding used in the experiments appends one component to each vector
before ℓ₂-normalisation, so for a dataset with `p` raw features these plots are
read at `d = p + 1`.

## The fourth figure

The manuscript also contains a figure that is not analytic: the crossover sweep
on Skin Segmentation. It is data-derived and therefore not kept here — the
benchmark harness writes it, one figure per copy number, as PNG and as PDF,
next to the measurements it plots:

```bash
pgm bench b --reps 3
# -> results_benchmark/componentB_crossover_c{2..7}.{png,pdf}
```

The same panels for every copy number on one page, one row per `c`, are
redrawn from the deposited measurements without running the sweep again:

```bash
pgm bench replot --combined
# -> results_benchmark/componentB_crossover_all.{png,pdf}
```

Its three panels compare the dense k-PGM, the k-PGM from explicit features and
the Rc-PGM on training time, stored model memory and prediction time, each
against the threshold of the condition that governs it, with the empirical
crossover of the dense k-PGM and the Rc-PGM. Note that the stored model is bounded by the *prediction*
condition `N > l·d_sym²/(d + r_{G^c})`, not by the training-memory one: the
memory a fitted k-PGM retains is the `O(N(d + r_{G^c}))` term that drives its
prediction cost.
