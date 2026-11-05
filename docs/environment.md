# Environment of the reported runs

Timings and memory figures are only comparable across runs that share the same
machine, the same thread pinning and the same library versions. This is the
machine the runs deposited in `results_benchmark/`, `results_selection/` and
`results_diagnostics/` were measured on; the machine-readable half of it is the
`env` block of every `*_meta.json`, written at start-up.

## Machine

| item | value |
|---|---|
| CPU model | Intel Xeon E5-2683 v4 @ 2.10 GHz (3.0 GHz max), dual socket |
| Physical cores / logical CPUs | 2 × 16 physical, 64 logical; 80 MiB L3 |
| RAM | 1 TiB |
| GPU (if any; the reported runs are CPU-only) | present but unused — the CUDA build of PyTorch runs on the CPU here |
| Operating system | Ubuntu 22.04.5 LTS, kernel 6.8.0-111-generic, glibc 2.35 |

## Software

| item | value |
|---|---|
| Python | 3.14.7 |
| NumPy | 2.5.2 |
| PyTorch | 2.13.0+cu130, `torch.float64`, CPU only |
| other packages | as resolved in `uv.lock` at tag `v1.1.0-rc1`, installed with `uv sync --locked --group dev --extra estimators` |
| BLAS backend | not recorded by the harness |

The three versions in the table are the ones every run recorded. The rest of
the environment is the lockfile itself. The runs of v1.0.x were measured in an
older stack (Python 3.10.12, NumPy 1.26.4, SciPy 1.14.1, PyTorch 1.13.1),
recorded in `requirements-repro.txt` at tag `v1.0.1`.

## Run parameters

| item | value |
|---|---|
| code | tag `v1.1.0-rc1`, commit `881e4c597f6d121202363ae0dbfd4611b4f21952` |
| `--threads` | 1 (BLAS and OpenMP pinned before numpy and torch are imported) |
| `pgm select` | stratified 5-fold CV on the training set, 10 repetitions, one-standard-error rule, `c = 1 … 9` |
| `--reps`, Component A | 5 timed repetitions after one warm-up |
| `--reps`, Component B | 3 timed repetitions after one warm-up |
| `--sweep` / `--copies`, Component B | N ∈ {25, 50, 100, 250, 500, 1000, 1750, 3000, 5000, 8000}, c ∈ {2, …, 7}, test set 2000 |
| `--seed` | 42 |
| `--mem-limit-gb` | 500 (no configuration skipped) |
| Date of the runs | 2026-11-04 |
| Truncation | `BASE_TOL = 1e-10` on the spectrum of `σ`, for the three estimators |

## Notes

- The three estimators run in `torch.float64` throughout; single precision
  changes both the timings and the retained rank `r_{G^c}` of `G^c`.
- the harness pins the BLAS/OpenMP thread count *before* importing
  numpy and torch. Running it under an externally set `OMP_NUM_THREADS` leaves
  that value in place, so record the effective one.
- Table 7 contains no timing: the copy numbers, the accuracies, the ranks and
  the analytic conditions reproduce on any machine (the run above gave the
  same values, digit for digit, as one on a different machine and stack).
