# Reported measurements

The files in this directory, in `results_selection/` and in
`results_diagnostics/` are the output of the runs quoted in the paper. They are
committed on purpose: the manuscript states that the raw measurements are
deposited with the code, and this is where they are.

## This directory: Components A and B

| file | content |
|---|---|
| `componentA.csv` | per-dataset training time, prediction time, model bytes, peak RSS and accuracy, for the k-PGM and the Rc-PGM, at the copy numbers of Table 7 (Table 8) |
| `componentA_meta.json` | environment of the run, the file the copy numbers were read from, per-dataset thresholds, argmax agreement of the equivalence check |
| `componentB.csv` | the same quantities along the Skin Segmentation sweep, `N = 25 … 8000`, at `c = 2 … 7`, for the dense k-PGM, the k-PGM from explicit features and the Rc-PGM |
| `componentB_meta.json` | environment, the copy numbers, the thresholds of each `c` with the rank retained at the largest `N`, the sweep grid and the test-set size |
| `componentB_crossovers.csv` | for each `c` and quantity, the threshold, the empirical crossover of the dense k-PGM and the Rc-PGM, and their ratio |
| `componentB_crossover_c{c}.png`, `.pdf` | one figure per copy number, as written by the harness |
| `componentB_crossover_all.png`, `.pdf` | every copy number on one page, one row per `c`; drawn by `pgm bench replot --combined` from `componentB.csv`, not by the harness |

## The other two directories

| file | content |
|---|---|
| `results_selection/selection_sweep.csv` | every dataset at every `c = 1 … 9`: mean, standard deviation and standard error of the cross-validated accuracy, test accuracy, `d_sym`, `r_{G^c}`, thresholds and the three advantage flags |
| `results_selection/selection.csv` | the selected row of each dataset (Table 7) |
| `results_selection/selection_meta.json` | the protocol: split seed, folds, repetitions, rule, environment |
| `results_selection/tables.tex` | the LaTeX rows of Table 7 and of the two appendix tables, from `pgm paper tables` |
| `results_diagnostics/povm.csv` | completeness and positivity residuals of the truncated measurement, retained rank and discarded eigenvalues, for every dataset at every `c = 1 … 9` |
| `results_diagnostics/povm_meta.json` | environment and protocol of that run |

## Provenance

All the files were produced on 2026-11-04 by the code at tag `v1.1.0-rc1`
(commit `881e4c597f6d121202363ae0dbfd4611b4f21952`), on the machine described in
`docs/environment.md`, in the locked environment, with one BLAS thread and the
truncation threshold `BASE_TOL = 1e-10` on the spectrum of `σ`:

```bash
pgm diag povm --mem-limit-gb 500
pgm select
pgm paper tables > results_selection/tables.tex
pgm bench b --reps 3
pgm bench a --reps 5 --mem-limit-gb 500
```

No configuration was skipped. The release `v1.1.0` changes no code that these
runs exercised; it only deposits them, and makes the estimate behind
`--mem-limit-gb` follow the per-class peak of the Rc-PGM, which the runs
measured.

The runs of v1.0.x (threshold `1e-6` on the spectrum of `G^c`, copy numbers
chosen on the test set) are reproducible from tag `v1.0.1` and from its Zenodo
archive.

## Regenerating them

```bash
pgm data fetch            # once, if datasets/ is empty
pgm diag povm --mem-limit-gb 500
pgm select
pgm paper tables > results_selection/tables.tex
pgm bench b --reps 3
pgm bench a --reps 5 --mem-limit-gb 500
pgm paper numbers         # every figure quoted in the paper
```

The commands write into these directories, so a re-run overwrites the
deposited files in place; `git status` is the safety net, and `--outdir` writes
a run elsewhere.

Timings and peak memory depend on the machine; the equivalence check, the copy
numbers, the cross-validated and test accuracies, the retained ranks
`r_{G^c}`, the residuals and the model sizes do not, and should reproduce
exactly. The peak RSS is a delta over each phase, so at small sizes it is
dominated by memory the allocator reuses within the process; at large sizes it
follows `4 N² · 8` bytes for the dense k-PGM and `max_j #k_j · d_sym² · 8` bytes
for the Rc-PGM.
