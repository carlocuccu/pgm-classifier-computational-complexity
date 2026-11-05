# Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement Classifiers

[![CI](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/ci.yml/badge.svg)](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/ci.yml)
[![Datasets](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/datasets.yml/badge.svg)](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/datasets.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code and data accompanying

> C. Cuccu, G. Sergioli, R. Giuntini, A. C. Granda Arango, R. Era,
> *Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement
> Classifiers*, submitted to **Quantum Machine Intelligence**.

The paper compares three quantum-inspired classifiers built on the Pretty Good
Measurement:

| name | idea | key dimension |
|---|---|---|
| **c-PGM** | explicit tensor copies of the encoded sample | `d^c` |
| **k-PGM** | the kernel trick replaces the tensor power by the entrywise `c`-th power of the Gram matrix | `N`, `r_{G^c}` |
| **Rc-PGM** | the c-PGM restricted to the symmetric subspace | `d_sym = C(d + c - 1, c)` |

The three are equivalent as classifiers and differ only in their computational
profile. This repository contains the implementations, the analytic-condition
evaluation behind Table 7, the empirical benchmark harness, and a script that
downloads and rebuilds the datasets exactly as they were used.

## Quickstart

```bash
git clone https://github.com/carlocuccu/pgm-classifier-computational-complexity
cd pgm-classifier-computational-complexity

make setup      # the environment, from uv.lock
make data       # download and rebuild the twelve datasets
make test       # the test suite
make selftest   # exercise the whole measurement pipeline
```

None of that needs PyTorch: the harness imports it lazily, so the datasets, the
self-test, the figure and most of the tests run without it. `make estimators`
adds it when you want the classifiers themselves.

`uv` is asked to use an interpreter it manages rather than whichever Python is
on `PATH` (`python-preference = "only-managed"`). A pyenv or distribution build
compiled without libffi has no `_ctypes`, numpy and scipy will not import on it,
and the failure appears far from its cause; a project about reproducible
environments should not inherit that lottery.

`make` on its own lists everything the repository offers. Without `uv`, the
classic path still works: `python -m venv .venv && pip install -r
requirements.txt`, then run the scripts directly.

The data is not carried in the repository: `datasets/` is empty until
`pgm data fetch` fills it. Everything it needs is in the repository -- the
source, the licence, the transformation, the row order, the column formatting
and the digest of every file -- and the twelve files it writes are
byte-identical to the ones used in the paper. Files already present and already
correct are left alone; `--check` re-verifies them without touching the
network, and `--help` prints the full provenance of every dataset.

`torch` is required by the three estimators and by `pgm select`. The dataset
checks and the harness self-test run without it.

## What produces what

| Artefact in the paper | Command | Output |
|---|---|---|
| **Table 7** and the copy-number sweep of the appendix (copy number chosen by cross-validation, Rc-PGM advantage conditions at every `c`) | `pgm select`<br>`pgm paper tables > results_selection/tables.tex` | `results_selection/selection.csv`, `selection_sweep.csv`, `selection_meta.json`; the LaTeX rows of Table 7 and of the appendix tables |
| **Section "Some empirical evidence"**, per-dataset timings, at the copy numbers chosen by `pgm select` | `pgm select`<br>`pgm bench a --reps 5 --mem-limit-gb 500` | `results_benchmark/componentA.csv`, `componentA_meta.json` |
| **Section "Some empirical evidence"**, crossover sweep on Skin Segmentation at `c = 2 … 7`, for the dense k-PGM, the k-PGM from explicit features and the Rc-PGM | `pgm data fetch --only-skin`<br>`pgm bench b --reps 3` | `results_benchmark/componentB.csv`, `componentB_meta.json`, `componentB_crossovers.csv` (empirical crossover, threshold and their ratio, per `c` and quantity), `componentB_crossover_c{c}.png` |
| **Figures 1–3** | analytic surfaces, not recomputed here | `figures/` (see `figures/README.md` for the plotted expressions) |
| **Tables 7 and 8, Figure 4, the appendices** | the runs as reported | already in `results_benchmark/`, `results_selection/` and `results_diagnostics/`; `pgm paper numbers` and `pgm paper tables` re-derive every figure and table row quoted in the text |
| POVM residuals of the truncated measurement (completeness, positivity, retained rank) | `pgm diag povm --mem-limit-gb 500` | `results_diagnostics/povm.csv`, `povm_meta.json` |

`--mem-limit-gb` is not a cap on memory: a configuration whose estimated peak
exceeds it is skipped and reported with `[skip]`. The default, 8 GB, skips car
and ecoli at the larger copy numbers; the deposited run used 500 and skipped
nothing.

`pgm bench … --outdir DIR` writes a run to `DIR` instead of `results_benchmark/`,
leaving the deposited run untouched; `pgm paper numbers --results DIR` and
`pgm bench replot --outdir DIR` read it back.

## Layout

```
pgm-repo/
├── qunica/classifiers/       the three estimators
│   ├── PGMHQC_gpu_cpu_dtype.py                     c-PGM     (qunica.CPGM)
│   ├── KPGMC_Low_Rank.py                           k-PGM     (qunica.KPGM)
│   └── PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank.py    Rc-PGM    (qunica.RcPGM)
├── src/pgm_complexity/       the package, installed as the `pgm` command
│   ├── cli.py                pins the BLAS threads, then dispatches
│   ├── config.py             the constants of the measurement protocol
│   ├── thresholds.py         the three advantage conditions
│   ├── selection.py          the copy number chosen by cross-validation
│   ├── crossovers.py         empirical crossovers of Component B against the thresholds
│   ├── data/                 specifications, rebuild, verification
│   ├── bench/                splits, measurement, Components A and B, self-test
│   ├── figures.py            Figure 4 and its redrawing
│   └── paper.py              every figure the empirical section quotes
├── qunica/                   the estimators as published with the paper
├── tests/                    what holds on any machine: identities, invariants, digests
├── datasets/                 empty; filled by `pgm data fetch`
├── results_benchmark/        the measurements reported in the paper, as produced
├── results_selection/        the copy-number selection behind Table 7
├── results_diagnostics/      the residuals of the truncated measurement
├── figures/                  the three figures of the paper
└── docs/
    ├── datasets.md           sources, licences and transformations of the data
    └── environment.md        the machine the deposited runs were measured on
```

## Using the estimators

The three classifiers follow the scikit-learn estimator interface.

```python
import torch
from qunica import KPGM, RcPGM

model = KPGM(n_copies=2, encoding="amplit", dtype=torch.float64)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Two conventions matter throughout, and are the reason the same `c` can look
different in the two families:

- **Encoded dimension.** Throughout the paper and here, `d` is the dimension of
  the *encoded* space — the one the classifier works in — not the number of raw
  features. `encoding="amplit"` appends one component to each vector before
  ℓ₂-normalisation, so a dataset with `p` raw features gives `d = p + 1`. The
  symmetric basis of the Rc-PGM and every occurrence of `d_sym` are built on
  that `d`; with this convention `r_{G^c} ≤ min(N, d_sym)` holds in every row of
  Table 7.
- **Retained rank.** The three estimators truncate the spectrum of
  `σ = Σ_j p_j ρ_j`, a density operator of trace one: the c-PGM and the Rc-PGM
  directly, the k-PGM through the weighted Gram matrix `K = W^{1/2} G^c W^{1/2}`
  (`w_i = p_j / #k_j`), which has the same nonzero spectrum. With the same
  `tol` they keep the same eigendirections; the number kept by the k-PGM is
  `r_{G^c}`, the numerical rank of the `c`-th power Gram matrix (not of `G`),
  available after `fit` as `model.lam_inv_sqrt.shape[0]`.

The estimators are used with their default empirical priors
(`class_weight=None`, `p_j = #k_j / N`), under which `K = G^c / N`; the k-PGM
also accepts `'balanced'` and a dict of priors.

## The advantage conditions

For a dataset with `N` training samples, `l` classes, encoded dimension `d` and
retained rank `r_{G^c}` of the `c`-th power Gram matrix, the Rc-PGM is
preferable to the k-PGM when

| resource | condition |
|---|---|
| training time | `N > l^(1/3) · d_sym` |
| training memory | `N > d_sym²` |
| prediction time **and** memory | `N > l · d_sym² / (d + r_{G^c})` |

The full prediction cost of the k-PGM is `O(N(d + r_{G^c}))` in both time and
memory, so a single threshold governs the two prediction metrics. These are the
inequalities evaluated in Table 7, and the ones `pgm select` recomputes for
every dataset at every copy number.

## Table 7: how the copy number is chosen

`pgm select` chooses the copy number on the training set only. For every
dataset it draws the seeded stratified 80/20 split of the harness
(`random_state=42`) and, on the training part, runs stratified 5-fold
cross-validation of the k-PGM at every `c` of the grid `c = 1 … 9`, repeated 10
times with different fold assignments (`random_state = 0 … 9`). The selected
`c` follows the **one-standard-error rule**: the smallest `c` whose mean
cross-validated accuracy is within one standard error of the largest one, the
standard error being the standard deviation of the 10 per-repetition accuracies
divided by √10. Where the accuracy curve is flat this keeps the cheaper copy
number instead of one ahead by a difference within the noise; the `c` of the
largest mean is recorded as well (`cv_argmax`). The test part plays no role in the choice: the model refitted on
the whole training set is evaluated on it once per `c`, and that accuracy is
only reported.

The same refit gives the retained rank `r_{G^c}` at every `c`, and the three
advantage conditions are evaluated at every copy number of the grid with the
rank of that `c`. The two training conditions depend on `c` only through
`d_sym`, so each of them holds up to some `c` and fails beyond it; the
prediction condition also depends on `r_{G^c}`, and is reported at every `c`
rather than summarised.

The output is the whole sweep (`selection_sweep.csv`: mean, standard
deviation and standard error over the repetitions of the CV accuracy, test accuracy, `d_sym`,
`r_{G^c}`, thresholds and flags, for every dataset and every `c`), the selected
row of each dataset (`selection.csv`) and the protocol (`selection_meta.json`).
A selected `c` equal to the last value of the grid is flagged on screen.
`pgm paper tables` reads them back and prints the LaTeX rows of Table 7 and of
the two appendix tables (mean CV accuracy and the three conditions at every
`c`, the selected one in boldface), so that no number is copied by hand.

## Working on it

| command | what it does |
|---|---|
| `make setup` | the development environment from `uv.lock`, without PyTorch |
| `make estimators` | add PyTorch, for the classifiers and the `-m torch` tests |
| `make lint` / `make format` | what CI checks / fix it in place |
| `make test` | the test suite; `-m torch` needs PyTorch, `-m network` is opt-in |
| `make data` / `make check` | rebuild the datasets / verify them offline |
| `make selftest` | the measurement pipeline end to end, with numpy stand-ins |
| `make figure` / `make numbers` | redraw Figure 4 · re-derive the quoted figures |
| `make repro` | the full reproduction: hours on one core, about 200 GB of memory; overwrites the deposited results |

The deposited measurements were taken in the locked environment, synced with
`uv sync --locked --group dev --extra estimators` from tag `v1.1.0-rc1`
(Python 3.14.7, numpy 2.5.2, torch 2.13.0); `docs/environment.md` describes the
machine. The runs of v1.0.x used an older stack (Python 3.10, torch 1.13.1),
recorded in `requirements-repro.txt` at tag `v1.0.1`.

`qunica/` is exempt from formatting and from style rules: it is the estimator
code published with the paper, kept verbatim so the measurements can be
reproduced against what produced them. Pyflakes still runs on it.

## Measurement protocol

The benchmark harness follows a fixed protocol, and the numbers it reports are
only comparable across runs that share it:

- **Inputs**: the features are read from the CSV files as stored, with no
  scaling, and the train/test partition is the one of Table 7 -- so Component A,
  run at the copy numbers `pgm select` chose, fits the very same models the
  table describes and reproduces its test accuracies exactly, adding only the
  measurements.
- **Split**: the partition is produced by
  `train_test_split(..., test_size=0.2, shuffle=True, stratify=y,
  random_state=42)`, so the training-set sizes `N` coincide with the tabulated
  ones.
- **Threads are pinned before numpy/torch are imported** — `--threads` (default
  `1`) sets `OMP_NUM_THREADS` and its siblings. Report the value together with
  the results.
- **Timing**: one warm-up followed by `--reps` timed repetitions of each phase;
  mean and sample standard deviation of `time.perf_counter`.
- **Memory**, two ways: the peak RSS delta over the phase, sampled from
  `/proc/self/statm` by a background thread, and the *model bytes*, i.e. the
  size of the arrays and tensors the fitted estimator actually holds. The
  second is the quantity the theoretical memory bounds describe; the first
  includes the allocator's behaviour.
- **Truncation threshold**: one threshold, `BASE_TOL = 1e-10`, on the spectrum
  of `σ` for all three estimators, independent of `N` — a margin above the
  rounding noise of the eigensolver; it is written to `*_meta.json`. Up to
  v1.0.1 the threshold was `1e-6` on the spectrum of `G^c` (`1e-6 / N` on
  `σ`); the runs of that version are reproducible from its tag.
- **Equivalence check**: every run ends by verifying that the k-PGM and the
  Rc-PGM produce the same argmax on the test set. The message distinguishes
  three outcomes: exact agreement; near-agreement, whose residual mismatches
  come from eigenvalues sitting at the truncation threshold; and a real
  disagreement, which makes the timings incomparable.
- Every `*_meta.json` records the environment of its run. Copy it into
  `docs/environment.md` when reporting results.
- **The reported runs are committed** under `results_benchmark/`,
  `results_selection/` and `results_diagnostics/`, together with a note on how
  they were produced (`results_benchmark/README.md`). The commands write into
  those same directories, so a re-run overwrites them: check `git status`
  before committing, or write elsewhere with `--outdir`.

## Datasets

No dataset is carried in this repository. `pgm data fetch`
downloads each of the twelve from PMLB, OpenML or UCI, re-applies the
documented transformation, restores the row order and the column formatting of
the files used in the experiments, verifies the result against the recorded
MD5/SHA-256 digests and writes it into `datasets/`. What it writes is
byte-identical to the data behind Table 7 and Table 8, so the reported numbers
reproduce exactly; nothing is written if any file fails its check.

The row order is part of that guarantee, not a cosmetic detail: the stratified
split is seeded but order-dependent, so the same rows in a different order give
different accuracies -- 0.900 instead of 0.967 on `iris`. Ten of the eleven CSV
files keep the order of their public release; `car` is a stable partition of it,
recorded in the script.

`docs/datasets.md` documents, for every file, its public source, its licence,
its citation and the transformation applied to it; `python
pgm data fetch --help` prints the same from the script itself.

## Citing

See `CITATION.cff`. The code is MIT-licensed (`LICENSE`); the datasets carry
the terms of their own sources, listed in `docs/datasets.md`.
