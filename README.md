# Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement Classifiers

[![CI](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/ci.yml/badge.svg)](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/ci.yml)
[![Datasets](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/datasets.yml/badge.svg)](https://github.com/carlocuccu/pgm-classifier-computational-complexity/actions/workflows/datasets.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code, data and notebooks accompanying

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
`scripts/fetch_datasets.py` fills it. That script is self-contained -- it holds
the source, the licence, the transformation, the row order, the column
formatting and the digest of every file -- and the twelve files it writes are
byte-identical to the ones used in the paper. Files already present and already
correct are left alone; `--check` re-verifies them without touching the
network, and `--help` prints the full provenance of every dataset.

`torch` is required by the three estimators and by the notebook. The dataset
checks and the harness self-test run without it.

## What produces what

| Artefact in the paper | Command | Output |
|---|---|---|
| **Table 7** (accuracy saturation and Rc-PGM advantage conditions) | `jupyter notebook notebooks/table7.ipynb` | `results_table7.csv` and the LaTeX body of the table |
| **Section "Some empirical evidence"**, per-dataset timings | `python run_benchmarks.py A --reps 5` | `results_benchmark/componentA.csv`, `componentA_meta.json` |
| **Section "Some empirical evidence"**, crossover sweep on Skin Segmentation | `python scripts/fetch_datasets.py --only-skin`<br>`python run_benchmarks.py B --reps 3` | `results_benchmark/componentB.csv`, `componentB_meta.json`, `componentB_crossover.png` |
| **Figures 1–3** | analytic surfaces, not recomputed here | `figures/` (see `figures/README.md` for the plotted expressions) |
| **Table 8, Figure 4** | the measurements as reported | already in `results_benchmark/`; `python scripts/paper_numbers.py` re-derives every figure quoted in the text |

## Layout

```
pgm-repo/
├── qunica/classifiers/       the three estimators
│   ├── PGMHQC_gpu_cpu_dtype.py                     c-PGM     (qunica.CPGM)
│   ├── KPGMC_Low_Rank.py                           k-PGM     (qunica.KPGM)
│   └── PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank.py    Rc-PGM    (qunica.RcPGM)
├── run_benchmarks.py         benchmark harness (Components A, B, selftest, replot)
├── scripts/
│   ├── fetch_datasets.py     downloads and rebuilds every dataset used in the paper
│   └── paper_numbers.py      derives every figure quoted in the paper from a run
├── tests/                    what holds on any machine: identities, invariants, digests
├── notebooks/table7.ipynb    regenerates Table 7
├── datasets/                 empty; filled by scripts/fetch_datasets.py
├── results_benchmark/        the measurements reported in the paper, as produced
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
- **Retained rank.** The k-PGM keeps the eigenvectors of `G^c` whose eigenvalue
  exceeds `tol = 1e-6`; their number is `r_{G^c}`, the numerical rank of the
  `c`-th power Gram matrix `G^c` — not of `G`, from which it differs as soon as
  `c > 1` — available after `fit` as `model.lam_inv_sqrt.shape[0]`. The c-PGM
  and the Rc-PGM apply their own `tol` to the spectrum of `σ = G^c / N`
  instead, so the same numerical truncation corresponds to a threshold `N`
  times smaller; `run_benchmarks.py` rescales it accordingly (see below).

Both estimators are used with their default empirical priors
(`class_weight=None`, `p_j = #k_j / N`), which is the convention Table 7
reports and the one under which `σ = G^c / N`.

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
inequalities evaluated in Table 7, and the ones `notebooks/table7.ipynb`
recomputes for every dataset.

## Table 7: how the copy number is chosen

`notebooks/table7.ipynb` fits the k-PGM at **every** copy number of a fixed grid,
`c = 1 … MAX_COPIES` (currently 9), for every dataset — an exhaustive sweep, with
no early stopping and no per-dataset range. The row carried into Table 7 is the
**smallest `c` attaining the maximum test-set accuracy** on that grid (`idxmax`
keeps the first occurrence). It is an argmax over the grid, not a
plateau-detection criterion and not a visual reading: where the accuracy is flat,
the smallest `c` of the plateau is the one reported.

The accuracies come from **one** seeded stratified 80/20 split
(`random_state=42`) — the same split `run_benchmarks.py` uses, which is why
Component A reproduces them exactly. They are therefore single-split point
estimates, with no repetitions and no confidence interval; on the smaller
datasets the test set is small (15 samples for `confidence`, 22 for `cloud`, 30
for `iris`), so one misclassification moves the accuracy by several points.

The notebook makes the process auditable: it prints the whole accuracy-vs-`c`
curve with the selected point starred, flags the two cases that must not be read
as saturation — `c* = 1` (copies bring no gain) and `c* = MAX_COPIES` (the
maximum sits at the edge of the grid, so the accuracy may still be rising) — and
writes the complete sweep to `results_table7_sweep.csv`.

The accuracy plays no part in the advantage conditions themselves: it only fixes
the `c` at which they are evaluated. Each condition compares a quantity growing
with `d_sym` against one that does not, so each holds for every `c` up to some
value and fails beyond it; the conditions can be recomputed at any other `c`
straight from `N`, `l`, `d` and `d_sym`.

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
| `make repro` | the full reproduction, hours on one core |

Two environments, kept apart on purpose. `uv.lock` is the working one, on a
current interpreter. The one the deposited measurements were taken in —
numpy 1.26.4, scipy 1.14.1, torch 1.13.1 — is `requirements-repro.txt`, outside
the lock: torch 1.13.1 stops at CPython 3.11, and a pin that old inside the
resolution propagates everywhere and makes the lockfile uninstallable on
anything newer. It is a record of a 2022 stack, not something to build against,
and `docs/environment.md` describes the machine around it.

`qunica/` is exempt from formatting and from style rules: it is the estimator
code published with the paper, kept verbatim so the measurements can be
reproduced against what produced them. Pyflakes still runs on it.

## Measurement protocol

The benchmark harness follows a fixed protocol, and the numbers it reports are
only comparable across runs that share it:

- **Inputs**: the features are read from the CSV files as stored, with no
  scaling, and the train/test partition is the one of Table 7 -- so Component A
  fits the very same models the table describes and reproduces its accuracies
  exactly, adding only the measurements.
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
- **Harmonised truncation threshold**: `BASE_TOL = 1e-6` is applied to the
  spectrum of `G^c` for the k-PGM, and `BASE_TOL / N` to the spectrum of
  `σ = G^c / N` for the c-PGM and the Rc-PGM, so that the two pseudo-inverses
  discard the same eigendirections. Set `HARMONIZE_TOL = False` to disable the
  rescaling; both constants are written to `*_meta.json`.
- **Equivalence check**: every run ends by verifying that the k-PGM and the
  Rc-PGM produce the same argmax on the test set. The message distinguishes
  three outcomes: exact agreement; near-agreement, whose residual mismatches
  come from eigenvalues sitting at the truncation threshold; and a real
  disagreement, which makes the timings incomparable.
- `results_benchmark/*_meta.json` records the environment of the run. Copy it
  into `docs/environment.md` when reporting results.
- **The reported runs are committed** under `results_benchmark/`, together with
  a note on how they were produced. The harness writes into that same
  directory, so a re-run overwrites them: check `git status` before committing.

## Datasets

No dataset is carried in this repository. `python scripts/fetch_datasets.py`
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
scripts/fetch_datasets.py --help` prints the same from the script itself.

## Citing

See `CITATION.cff`. The code is MIT-licensed (`LICENSE`); the datasets carry
the terms of their own sources, listed in `docs/datasets.md`.
