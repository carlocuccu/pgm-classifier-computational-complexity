# Reported measurements

The files in this directory are the output of the benchmark runs quoted in the
paper — Table 8 and Figure 4 of the section *Some empirical evidence*. They are
committed on purpose: the manuscript states that the raw measurements are
deposited with the code, and this is where they are.

| file | content |
|---|---|
| `componentA.csv` | per-dataset training time, prediction time, model bytes, peak RSS and accuracy, for the k-PGM and the Rc-PGM |
| `componentA_meta.json` | environment of the run, per-dataset thresholds, argmax agreement of the equivalence check |
| `componentB.csv` | the same quantities along the Skin Segmentation sweep, `N = 250 … 8000` |
| `componentB_meta.json` | environment, the three thresholds, the sweep grid and the test-set size |
| `componentB_crossover.png`, `.pdf` | the figure of the manuscript, as written by the harness |

## Provenance of these files

| | Component A | Component B |
|---|---|---|
| produced on | 2026-08-24 | 2026-08-25 |
| command | `python run_benchmarks.py A --reps 5` | `python run_benchmarks.py B --reps 3` |

Those are the commands as they were at tag `v1.0.0-rc1`, which is the version
the paper cites. The harness has since become a package, and the same two runs
are now `pgm bench a --reps 5` and `pgm bench b --reps 3`; the measurements
themselves are untouched.
| repetitions | 5 after warm-up | 3 after warm-up |

Both runs used `BASE_TOL = 1e-6` with `HARMONIZE_TOL = True`, one BLAS thread,
PyTorch 1.13.1 in double precision on CPU. The `env` block of each
`*_meta.json` records the full configuration; `docs/environment.md` expands it
with the machine description.

## Regenerating them

```bash
pgm data fetch            # once, if datasets/ is empty
pgm bench a --reps 5
pgm bench b --reps 3
pgm paper numbers         # every figure quoted in the paper
```

The harness writes into **this** directory, so a re-run overwrites the
deposited files in place. That is deliberate — it keeps one obvious location
for the results — but it means `git status` is the safety net: check it before
committing, and restore with `git checkout results_benchmark/` if you did not
mean to replace the reported run.

Timings depend on the machine; the equivalence check, the retained ranks
`r_{G^c}`, the accuracies and the model sizes do not, and should reproduce
exactly.
