# Environment of the reported runs

Timings and memory figures are only comparable across runs that share the same
machine, the same thread pinning and the same library versions. This is the
machine the runs deposited in `results_benchmark/` were measured on; the
machine-readable half of it is the `[env]` block of
`results_benchmark/componentA_meta.json` and `componentB_meta.json`, written by
the harness at start-up.

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
| Python | 3.10.12 |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |
| pandas | not recorded per-version by the harness |
| scikit-learn | not recorded per-version by the harness |
| PyTorch | 1.13.1+cu117, `torch.float64`, CPU only |
| BLAS backend (`numpy.show_config()`) | Intel MKL 2020.0.0 for PyTorch; OpenBLAS 0.3.23 for NumPy and SciPy |

The three versions that the runs did record — Python, NumPy and PyTorch — are
pinned in `requirements-repro.txt`, together with SciPy; its header says how to
recreate the environment. The two that were not recorded are absent rather than
invented: a pin nobody measured is worse than an honest gap.

## Run parameters

| item | value |
|---|---|
| `--threads` | 1 (BLAS and OpenMP pinned before numpy and torch are imported) |
| `--reps`, Component A | 5 timed repetitions after one warm-up |
| `--reps`, Component B | 3 timed repetitions after one warm-up |
| `--sweep` / `--nmax`, Component B | N ∈ {250, 500, 1000, 1750, 3000, 5000, 8000}, test set 2000 |
| `--seed` | 42 |
| Date of the run | Component A: 2026-08-24 · Component B: 2026-08-25 |
| Truncation | `BASE_TOL = 1e-6` with `HARMONIZE_TOL = True` |

## Notes

- The three estimators run in `torch.float64` throughout; single precision
  changes both the timings and the retained rank `r_{G^c}` of `G^c`.
- `run_benchmarks.py` pins the BLAS/OpenMP thread count *before* importing
  numpy and torch. Running it under an externally set `OMP_NUM_THREADS` leaves
  that value in place, so record the effective one.
- Table 7 contains no timing, only analytic conditions evaluated on integer
  dataset descriptors; it is reproducible on any machine.
