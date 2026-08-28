"""Thread pinning, which has to happen before numpy or torch are imported.

The reported timings are single-core. BLAS reads its thread count from the
environment at load time, so setting it afterwards has no effect: this module
must therefore run before anything that imports numpy, which is why it lives on
its own and is called at the top of the command-line entry point.
"""

from __future__ import annotations

import os
import sys

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_threads(argv: list[str] | None = None) -> int:
    """Pin every BLAS/OpenMP pool to the value of --threads (default 1).

    `setdefault` rather than assignment: a value already exported by the caller
    wins, so a run under an external OMP_NUM_THREADS is not silently overridden
    -- it is reported instead, in the `env` block of the run metadata.
    """
    argv = sys.argv if argv is None else argv
    n = 1
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(n))
    return n
