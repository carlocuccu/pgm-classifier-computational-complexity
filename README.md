# Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement Classifiers

Code, data and notebooks accompanying

> C. Cuccu, G. Sergioli, R. Giuntini, A. C. Granda Arango, R. Era,
> *Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement
> Classifiers*, submitted to **Quantum Machine Intelligence**.

The paper compares three quantum-inspired classifiers built on the Pretty Good
Measurement:

| name | idea | key dimension |
|---|---|---|
| **c-PGM** | explicit tensor copies of the encoded sample | `d^c` |
| **k-PGM** | the kernel trick replaces the tensor power by the entrywise `c`-th power of the Gram matrix | `N`, `r_G` |
| **Rc-PGM** | the c-PGM restricted to the symmetric subspace | `d_sym = C(d̃ + c - 1, c)` |

The three are equivalent as classifiers and differ only in their computational
profile. This repository contains the implementations, the analytic-condition
evaluation behind Table 7, the empirical benchmark harness, and the datasets.

## Quickstart

```bash
git clone <REPOSITORY-URL> && cd pgm-repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python datasets/prepare_datasets.py check     # dataset integrity
python run_benchmarks.py selftest             # harness self-test (no torch needed)
```

`torch` is required by the three estimators and by the notebook. The dataset
checks and the harness self-test run without it.

## What produces what

| Artefact in the paper | Command | Output |
|---|---|---|
| **Table 7** (accuracy saturation and Rc-PGM advantage conditions) | `jupyter notebook notebooks/table7.ipynb` | `results_table7.csv` and the LaTeX body of the table |
| **Section "Some empirical evidence"**, per-dataset timings | `python run_benchmarks.py A --reps 5` | `results_benchmark/componentA.csv`, `componentA_meta.json` |
| **Section "Some empirical evidence"**, crossover sweep on Skin Segmentation | `python datasets/download_skin_segmentation.py`<br>`python run_benchmarks.py B --reps 3` | `results_benchmark/componentB.csv`, `componentB_meta.json`, `componentB_crossover.png` |
| **Figures 1–3** | analytic surfaces, not recomputed here | `figures/` (see `figures/README.md` for the plotted expressions) |

## Layout

```
pgm-repo/
├── qunica/classifiers/       the three estimators
│   ├── PGMHQC_gpu_cpu_dtype.py                     c-PGM     (qunica.CPGM)
│   ├── KPGMC_Low_Rank.py                           k-PGM     (qunica.KPGM)
│   └── PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank.py    Rc-PGM    (qunica.RcPGM)
├── run_benchmarks.py         empirical benchmark harness (Components A, B, selftest)
├── scripts/paper_numbers.py  derives every figure quoted in the paper from a run
├── notebooks/table7.ipynb    regenerates Table 7
├── datasets/                 the 11 CSV files, their manifest and preparation scripts
├── figures/                  the three figures of the paper
└── docs/environment.md       where to record the machine the results were produced on
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

- **Encoded dimension.** `encoding="amplit"` appends one component to each
  vector before ℓ₂-normalisation, so the space the classifier really works in
  has dimension `d̃ = d + 1`. The symmetric basis of the Rc-PGM and every
  occurrence of `d_sym` in the advantage conditions are built on `d̃`, not on
  the raw `d`. With this convention `r_G ≤ min(N, d_sym)` holds in every row of
  Table 7.
- **Retained rank.** The k-PGM keeps the eigenvectors of `G^c` whose eigenvalue
  exceeds `tol = 1e-6`; their number is `r_G`, available after `fit` as
  `model.lam_inv_sqrt.shape[0]`. The c-PGM and the Rc-PGM apply their own `tol`
  to the spectrum of `σ = G^c / N` instead, so the same numerical truncation
  corresponds to a threshold `N` times smaller; `run_benchmarks.py` rescales it
  accordingly (see below).

Both estimators are used with their default empirical priors
(`class_weight=None`, `p_j = #k_j / N`), which is the convention Table 7
reports and the one under which `σ = G^c / N`.

## The advantage conditions

For a dataset with `N` training samples, `l` classes, encoded dimension `d̃` and
retained Gram rank `r_G`, the Rc-PGM is preferable to the k-PGM when

| resource | condition |
|---|---|
| training time | `N > l^(1/3) · d_sym` |
| training memory | `N > d_sym²` |
| prediction time **and** memory | `N > l · d_sym² / (d̃ + r_G)` |

The full prediction cost of the k-PGM is `O(N(d̃ + r_G))` in both time and
memory, so a single threshold governs the two prediction metrics. These are the
inequalities evaluated in Table 7, and the ones `notebooks/table7.ipynb`
recomputes for every dataset.

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

## Datasets

Eleven small benchmarks are shipped in `datasets/`, header-less and with the
class label in the last column; Skin Segmentation is fetched on demand.
`datasets/README.md` documents, for every file, its public source, its licence,
its citation and the transformation applied to it, and
`datasets/prepare_datasets.py` both verifies the shipped files and rebuilds
them from those sources.

Reproducing the reported numbers requires the CSV files **as shipped**: the
stratified split is seeded but order-dependent, and the row order is not part
of the documented transformation.

## Citing

See `CITATION.cff`. The code is MIT-licensed (`LICENSE`); the datasets carry
the terms of their own sources, listed in `datasets/README.md`.
