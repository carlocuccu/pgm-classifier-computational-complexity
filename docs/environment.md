# Environment of the reported runs

Timings and memory figures are only comparable across runs that share the same
machine, the same thread pinning and the same library versions. Fill in the
table below from the `[env]` line that `run_benchmarks.py` prints at start-up —
the same dictionary is written to `results_benchmark/componentA_meta.json` and
`results_benchmark/componentB_meta.json`.

```
[env] {"python": ..., "platform": ..., "cpu_count": ..., "threads": ...,
       "numpy": ..., "torch": ..., "timestamp": ...}
```

## Machine

| item | value |
|---|---|
| CPU model | |
| Physical cores / logical CPUs | |
| RAM | |
| GPU (if any; the reported runs are CPU-only) | |
| Operating system | |

## Software

| item | value |
|---|---|
| Python | |
| NumPy | |
| SciPy | |
| pandas | |
| scikit-learn | |
| PyTorch | |
| BLAS backend (`numpy.show_config()`) | |

## Run parameters

| item | value |
|---|---|
| `--threads` | |
| `--reps`, Component A | |
| `--reps`, Component B | |
| `--sweep` / `--nmax`, Component B | |
| `--seed` | |
| Date of the run | |

## Notes

- The three estimators run in `torch.float64` throughout; single precision
  changes both the timings and the retained Gram rank `r_G`.
- `run_benchmarks.py` pins the BLAS/OpenMP thread count *before* importing
  numpy and torch. Running it under an externally set `OMP_NUM_THREADS` leaves
  that value in place, so record the effective one.
- Table 7 contains no timing, only analytic conditions evaluated on integer
  dataset descriptors; it is reproducible on any machine.
