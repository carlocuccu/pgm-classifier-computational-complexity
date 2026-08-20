"""Estimator implementations.

Modules
-------
``PGMHQC_gpu_cpu_dtype``
    c-PGM: explicit tensor copies, exported as :class:`qunica.CPGM`.
``KPGMC_Low_Rank``
    k-PGM with low-rank spectral decomposition of the Gram matrix,
    exported as :class:`qunica.KPGM`.
``PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank``
    Rc-PGM: symmetric-subspace reduction, exported as :class:`qunica.RcPGM`.

The three modules import ``torch`` at module level, so they are resolved
lazily here as well.
"""

__all__ = ["CPGM", "KPGM", "RcPGM"]

_LAZY = {
    "CPGM": (".PGMHQC_gpu_cpu_dtype", "PGMHQC_gpu_cpu_dtype"),
    "KPGM": (".KPGMC_Low_Rank", "KPGM"),
    "RcPGM": (".PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank", "PGMHQC_gpu_cpu_dtype"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name, __name__), attr)


def __dir__():
    return sorted(list(globals()) + __all__)
