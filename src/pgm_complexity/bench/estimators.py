"""Access to the three published estimators, and the tolerance that pairs them.

Imported lazily on purpose: everything else in this package -- the datasets,
the self-test, the figure, the thresholds -- runs without PyTorch, and only
these factories need it.
"""

from __future__ import annotations

import sys

from pgm_complexity import ROOT
from pgm_complexity.bench.measure import timed
from pgm_complexity.config import BASE_TOL


def ensure_estimators_importable() -> bool:
    """Make `qunica` importable however this code was invoked.

    `qunica` is not an installed package: it is a plain directory at the
    repository root, beside `src/`. Whether it can be imported therefore
    depends on how the process was started, and the difference is easy to
    miss because the two ways that are exercised during development both
    work:

    * `python -m pgm_complexity.cli` and `python script.py` put the working
      directory, or the script's directory, first on `sys.path`;
    * pytest puts the root there because of `pythonpath = ["."]`.

    The installed `pgm` console script does neither. Python gives it the
    directory of the script itself -- `.venv/bin` -- and never the working
    directory, so `import qunica` fails there and only there: in the one
    command the README tells a reader to run.

    Rather than depend on the caller, put the root on `sys.path` ourselves,
    once, and only when it is actually needed. Returns whether `qunica` is
    importable afterwards.
    """
    import importlib.util

    def present() -> bool:
        try:
            return importlib.util.find_spec("qunica") is not None
        except (ImportError, ValueError):
            return False

    if present():
        return True
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
        importlib.invalidate_caches()
    return present()


def get_classifiers(include_cpgm=False):
    import importlib
    import importlib.util

    def load(modname):
        for prefix in ("qunica.classifiers.", "qunica."):
            name = prefix + modname
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError, ModuleNotFoundError):
                found = False
            if found:
                # Import it for real. Anything that goes wrong from here is a
                # genuine error inside the estimator or a missing dependency
                # of its own -- most often PyTorch -- and must surface as
                # itself rather than be reported as a missing module.
                return importlib.import_module(name)
        raise ModuleNotFoundError(
            f"cannot import {modname!r}: expected qunica/classifiers/{modname}.py "
            f"(or qunica/{modname}.py) under the project root {ROOT}"
        )

    ensure_estimators_importable()

    kmod = load("KPGMC_Low_Rank")
    rmod = load("PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank")
    KPGM = kmod.KPGM
    RcPGM = rmod.PGMHQC_gpu_cpu_dtype

    # One pseudo-inverse threshold for the three estimators. All of them
    # truncate the spectrum of sigma = sum_j p_j rho_j: the primal ones
    # directly, the k-PGM through its weighted Gram matrix
    # K = W^{1/2} G^c W^{1/2}, which has the same nonzero spectrum. With the
    # same `tol` they discard the same eigendirections, for any priors and any
    # N. `n_train` is accepted for compatibility and no longer used.
    tol = BASE_TOL

    out = {
        "kpgm": lambda c, n_train=None: KPGM(n_copies=c, encoding="amplit", tol=tol),
        "rcpgm": lambda c, n_train=None: RcPGM(n_copies=c, encoding="amplit", tol=tol),
        # The same k-PGM, with the spectrum of K = (W^{1/2} Y)(W^{1/2} Y)^T
        # taken from the thin SVD of W^{1/2} Y, Y the explicit feature matrix
        # (N x d_sym), instead of the dense N x N matrix: O(N d_sym^2) time
        # and O(N d_sym) memory.
        "kpgm_features": lambda c, n_train=None: KPGM(
            n_copies=c, encoding="amplit", tol=tol, solver="features"
        ),
    }
    if include_cpgm:
        cmod = load("PGMHQC_gpu_cpu_dtype")
        CPGM = cmod.PGMHQC_gpu_cpu_dtype
        out["cpgm"] = lambda c, n_train=None: CPGM(
            n_copies=c, encoding="amplit", tol=tol
        )
    return out


def rc_preprocessing_time(RcFactory, X_train, c, reps):
    """Standalone timing of the Rc-PGM reduction pipeline (basis + factors +
    encoding + mapping), replicated outside fit so it can be reported
    separately. The symmetric basis is built on the encoded dimension
    (self.d = d_raw + 1 when an encoding is used)."""
    est = RcFactory(c, n_train=len(X_train))

    def basis():
        est.d = X_train.shape[1] + 1  # encoded dimension (amplit)
        est.occupation_numbers = sorted(
            est._enumerate_occupation_numbers(), reverse=True
        )
        est.dsym = len(est.occupation_numbers)
        est.multinomial_factors = est._calculate_multinomial_factors()
        return est.dsym

    def mapping():
        Xp = (
            est.X_prime_func(X_train, X_train.shape[0])
            if hasattr(est, "X_prime_func")
            else sys.modules[type(est).__module__].X_prime_func(
                est, X_train, X_train.shape[0]
            )
        )
        return est.map_batch_efficiently(Xp)

    t_basis = timed(basis, reps)[:2]
    t_map = timed(mapping, reps)[:2]
    return t_basis, t_map
