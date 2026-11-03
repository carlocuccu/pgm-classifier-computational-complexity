"""Numerical diagnostics of the truncated PGM measurement.

The pseudo-inverse sigma^{-1/2} is regularized by discarding the eigenvalues
of sigma below a threshold, i.e. the classifier is built on the truncated
operator sigma_eps. The completed operators

    F_i = sigma_eps^{-1/2} p_i rho_i sigma_eps^{-1/2} + (1/l) P_ker(sigma_eps)

form a POVM exactly when the completion is taken from the same spectral mask
as the pseudo-inverse. This module measures how closely a fitted model meets
that requirement -- completeness ||sum_i F_i - I|| and positivity of every
F_i -- and records where the threshold cuts the spectrum of sigma.

Predicted labels cannot reveal a defective completion, since the completion
term adds the same quantity to every class score: the residuals have to be
measured on the operators themselves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from pgm_complexity import ROOT

DIAG_DIR = ROOT / "results_diagnostics"


@dataclass
class PovmResiduals:
    dim: int  # dimension of the space the operators act on (d_sym or d^c)
    tol: float  # eigenvalue threshold of the pseudo-inverse
    rank_sigma: int  # eigenvalues of sigma retained (lam > tol)
    n_band: int  # strictly positive eigenvalues discarded (zero < lam <= tol)
    lam_min_kept: float  # smallest retained eigenvalue (nan if none)
    lam_max_discarded: float  # largest discarded eigenvalue (nan if none)
    completeness_2: float  # ||sum_i F_i - I||_2
    completeness_fro: float  # ||sum_i F_i - I||_F
    min_eig_F: float  # min over i of the smallest eigenvalue of F_i


def povm_residuals(model) -> PovmResiduals:
    """Residuals of a fitted c-PGM or Rc-PGM (attributes ``pgms_``,
    ``sigma_eigvals_``, ``tol``).

    An eigenvalue counts as numerically zero below ``dim * eps * lam_max``,
    the usual rank tolerance of a symmetric eigensolver; ``n_band`` counts the
    eigenvalues above that floor and at or below ``tol``, i.e. the directions
    in which a separately computed null space would differ from the subspace
    discarded by the pseudo-inverse.
    """
    import torch

    F = model.pgms_.to(torch.float64)
    dim = F.shape[-1]
    eye = torch.eye(dim, dtype=F.dtype, device=F.device)
    defect = F.sum(dim=0) - eye
    completeness_2 = torch.linalg.matrix_norm(defect, ord=2).item()
    completeness_fro = torch.linalg.matrix_norm(defect, ord="fro").item()
    min_eig = min(torch.linalg.eigvalsh((Fi + Fi.T) / 2).min().item() for Fi in F)

    lam = model.sigma_eigvals_.to(torch.float64).numpy()
    tol = float(model.tol)
    kept = lam[lam > tol]
    discarded = lam[lam <= tol]
    floor = dim * np.finfo(np.float64).eps * max(float(lam.max()), 0.0)
    n_band = int(np.sum((lam > floor) & (lam <= tol)))

    return PovmResiduals(
        dim=dim,
        tol=tol,
        rank_sigma=int(kept.size),
        n_band=n_band,
        lam_min_kept=float(kept.min()) if kept.size else float("nan"),
        lam_max_discarded=float(discarded.max()) if discarded.size else float("nan"),
        completeness_2=completeness_2,
        completeness_fro=completeness_fro,
        min_eig_F=min_eig,
    )


def _estimated_bytes(l: int, dsym: int) -> int:
    # Same calibrated estimate as Component A (2l persistent d_sym x d_sym
    # matrices plus ~5 transients, float64), plus one more for the residuals.
    return (2 * l + 6) * dsym * dsym * 8


def povm_table(datasets, copies, seed=42, mem_limit_gb=8.0, outdir=None):
    """Fit the Rc-PGM on the seeded split of every dataset, at every copy
    number, with the harness threshold, and record the residuals.

    Returns the list of rows; also writes ``povm.csv`` and ``povm_meta.json``
    in ``outdir`` (default ``results_diagnostics/``).
    """
    import csv

    from pgm_complexity.bench.estimators import get_classifiers
    from pgm_complexity.bench.io import env_info, load_dataset, stratified_split
    from pgm_complexity.thresholds import dsym_of

    outdir = DIAG_DIR if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)
    factories = get_classifiers()

    rows, skipped = [], []
    for name in datasets:
        X, y = load_dataset(name)
        tr, _ = stratified_split(y, 0.2, seed)
        Xtr, ytr = X[tr], y[tr]
        l = len(np.unique(ytr))
        for c in copies:
            dsym = dsym_of(X.shape[1], c)
            need = _estimated_bytes(l, dsym)
            if need > mem_limit_gb * 1e9:
                print(
                    f"[skip] {name} c={c}: d_sym={dsym}, estimated "
                    f"{need / 1e9:.1f} GB > --mem-limit-gb {mem_limit_gb}"
                )
                skipped.append({"dataset": name, "c": c, "dsym": dsym})
                continue
            model = factories["rcpgm"](c, n_train=len(Xtr)).fit(Xtr, ytr)
            res = povm_residuals(model)
            row = {
                "dataset": name,
                "N": len(Xtr),
                "d_enc": X.shape[1] + 1,
                "c": c,
                "l": l,
                **asdict(res),
            }
            rows.append(row)
            print(
                f"{name:18s} c={c} dsym={res.dim:5d} rank={res.rank_sigma:5d} "
                f"band={res.n_band:3d} ||sumF-I||_2={res.completeness_2:.2e} "
                f"min eig F={res.min_eig_F:.2e}"
            )
            del model

    if rows:
        with open(outdir / "povm.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (outdir / "povm_meta.json").write_text(
        json.dumps(
            {
                "env": env_info(),
                "seed": seed,
                "copies": list(copies),
                "estimator": "rcpgm",
                "skipped": skipped,
            },
            indent=2,
        )
    )
    print(f"\n[diag] {len(rows)} rows -> {outdir / 'povm.csv'}")
    return rows
