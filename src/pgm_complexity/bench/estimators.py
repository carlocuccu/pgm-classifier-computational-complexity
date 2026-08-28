"""Access to the three published estimators, and the tolerance that pairs them.

Imported lazily on purpose: everything else in this package -- the datasets,
the self-test, the figure, the thresholds -- runs without PyTorch, and only
these factories need it.
"""

from __future__ import annotations

import sys

from pgm_complexity import ROOT
from pgm_complexity.bench.measure import timed
from pgm_complexity.config import BASE_TOL, HARMONIZE_TOL


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

    def rc_tol(n_train):
        # Scale-consistent pseudo-inverse threshold. The k-PGM truncates the
        # spectrum of G^c = Phi Phi^T at BASE_TOL, whereas the reduced and
        # copy-dependent estimators truncate the spectrum of sigma = G^c / N,
        # which is the same nonzero spectrum divided by N. Dividing the
        # threshold by N makes the two truncations coincide, so the estimators
        # agree exactly rather than up to near-threshold eigenvalues.
        #
        # sigma = G^c / N holds because the estimators are used with their
        # default empirical priors (class_weight=None, i.e. p_j = #k_j / N),
        # under which sigma is the plain average of the N pure states. With
        # class_weight='balanced' the relation would pick up class-imbalance
        # factors and this rescaling would no longer be exact.
        return (BASE_TOL / n_train) if (HARMONIZE_TOL and n_train) else BASE_TOL

    out = {
        "kpgm": lambda c, n_train=None: KPGM(
            n_copies=c, encoding="amplit", tol=BASE_TOL
        ),
        "rcpgm": lambda c, n_train=None: RcPGM(
            n_copies=c, encoding="amplit", tol=rc_tol(n_train)
        ),
    }
    if include_cpgm:
        cmod = load("PGMHQC_gpu_cpu_dtype")
        CPGM = cmod.PGMHQC_gpu_cpu_dtype
        out["cpgm"] = lambda c, n_train=None: CPGM(
            n_copies=c, encoding="amplit", tol=rc_tol(n_train)
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
