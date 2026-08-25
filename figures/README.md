# Figures

The three *analytic* figures of the manuscript, as PNG and as PDF. They are plots of the
closed-form complexity expressions derived in the paper — no data and no
measurement enter them — and were produced with Wolfram Mathematica. The exact
expressions are listed below, so that each surface can be re-derived
independently.

Throughout, `N` is the training-set size, `d` the feature dimension, `c` the
number of tensor copies, and

```
d_sym = binomial(d + c - 1, c)
```

is the dimension of the symmetric subspace.

## `pgm_vs_kpgm` — Figure 1

Training-memory comparison between the c-PGM and the k-PGM, over
`d ∈ [1, 50]`, `c ∈ [2, 6]`, `N ∈ [10^5, 10^6]`.

| region | inequality | more memory-efficient |
|---|---|---|
| red | `N < d^(2c)` | k-PGM |
| blue | `N > d^(2c)` | c-PGM |

The c-PGM stores `l` centroids of size `d^c × d^c`, hence the `d^(2c)` term;
the k-PGM stores the `N × N` Gram matrix.

## `reduced` — Figure 2

Growth of the tensor-space dimension `d^c` against the symmetric-subspace
dimension `d_sym`, over `d ∈ [2, 20]` and `c ∈ [2, 6]`, on a base-10
logarithmic vertical axis.

| surface | expression |
|---|---|
| red | `d^c` |
| blue | `binomial(d + c - 1, c)` |

The gap between the two surfaces is the reduction the Rc-PGM exploits: every
occurrence of `d^(2c)` in the c-PGM complexities becomes `d_sym^2`.

## `pgm_vs_kpgm_vs_rcpgm` — Figure 3

The same comparison as Figure 1, with the Rc-PGM added, over the same ranges.

| region | inequality | most memory-efficient |
|---|---|---|
| red | `N < d_sym^2` | k-PGM |
| green | `d_sym^2 < N < d^(2c)` | Rc-PGM |
| blue | `N > d^(2c)` | c-PGM |

The green volume is the regime opened up by the symmetric-subspace reduction:
there the Rc-PGM already overtakes the k-PGM in training memory, while the
unreduced c-PGM does not.

## Note on the encoded dimension

As everywhere in the paper, `d` is the dimension of the *encoded* space, the one
the classifier operates in — not the number of raw features of a dataset. The
amplitude encoding used in the experiments appends one component to each vector
before ℓ₂-normalisation, so for a dataset with `p` raw features these plots are
read at `d = p + 1`.

## The fourth figure

The manuscript also contains a figure that is not analytic: the crossover sweep
on Skin Segmentation. It is data-derived and therefore not kept here — the
benchmark harness writes it, as PNG and as PDF, next to the measurements it
plots:

```bash
python run_benchmarks.py B --reps 3
# -> results_benchmark/componentB_crossover.{png,pdf}
```

Its three panels compare the k-PGM and the Rc-PGM on training time, stored
model memory and prediction time, each against the threshold of the condition
that governs it. Note that the stored model is bounded by the *prediction*
condition `N > l·d_sym²/(d + r_G)`, not by the training-memory one: the memory a
fitted k-PGM retains is the `O(N(d + r_G))` term that drives its prediction cost.
