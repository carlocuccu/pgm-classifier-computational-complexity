"""Quantum-inspired classifiers used in the paper.

*Computational Complexity Analysis of Quantum-Inspired Pretty Good Measurement
Classifiers.*

The package exposes the three estimators compared in the paper:

``CPGM``
    Copy-dependent PGM, i.e. the classifier acting on the explicit tensor
    power :math:`\\ket{x}^{\\otimes c}` of the encoded sample.
``KPGM``
    Kernel PGM, which replaces the explicit tensor power with the entrywise
    :math:`c`-th power of the Gram matrix.
``RcPGM``
    Reduced copy-dependent PGM, i.e. the c-PGM restricted to the symmetric
    subspace of dimension :math:`d_{\\mathrm{sym}} = \\binom{\\tilde d + c - 1}{c}`.

Importing the estimators requires ``torch``; the symbols are therefore
resolved lazily, so that ``import qunica`` alone does not pull torch in.
"""

__all__ = ["CPGM", "KPGM", "RcPGM"]

__version__ = "1.0.0"

_LAZY = {
    "CPGM": ("qunica.classifiers.PGMHQC_gpu_cpu_dtype", "PGMHQC_gpu_cpu_dtype"),
    "KPGM": ("qunica.classifiers.KPGMC_Low_Rank", "KPGM"),
    "RcPGM": ("qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank",
              "PGMHQC_gpu_cpu_dtype"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def __dir__():
    return sorted(list(globals()) + __all__)
