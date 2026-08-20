from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import torch
from tqdm import trange
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

__all__ = ["KPGMClassifier"]


class KPGM(BaseEstimator, ClassifierMixin):
    """Kernel Pretty‑Good Measurement classifier (kPGM‑C).

    Parameters
    ----------
    n_copies : int, default=1
        Number of *copies* (tensor‑power encoding) as in Eq. 55.
    encoding : {None, 'amplit', 'stereo'}, default='amplit'
        Encoding to map classical vectors → quantum states.
    rescale : float, default=1.0
        Global scale factor applied prior to encoding (cf. Sec. 6).
    tol : float, default=1e-6
        Eigenvalue threshold for pseudo‑inverse square‑root of *G*.
    dtype : torch.dtype, default=torch.float64
        Computation precision for the encoding routine.
    device : str or torch.device, default='cpu'
        Device for torch tensors in the encoding routine.
    """

    # --------------------------- initialisation ------------------------- #

    def __init__(
        self,
        *,
        n_copies: int = 1,
        encoding: Literal[None, "amplit", "stereo"] = "amplit",
        rescale: float = 1.0,
        tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_copies = n_copies
        self.encoding = encoding
        self.rescale = rescale
        self.tol = tol
        self.dtype = dtype
        self.device = device #torch.device(device)

        # runtime‑populated attributes
        self.X_prime_train: Optional[np.ndarray] = None  # encoded train states
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.G_inv_sqrt: Optional[np.ndarray] = None
        self.w_vec: Optional[np.ndarray] = None

    # --------------------------- encoding ------------------------------ #

    def X_prime_func(self, X, m):
        X_ = torch.tensor(X, dtype = self.dtype).to(self.device)

        X_ = self.rescale * X_

        X_sq_sum = (X_ ** 2).sum(dim=1)

        if not self.encoding:        
            X_prime = X_
        elif self.encoding == "amplit":
            X_prime = torch.nn.functional.normalize(
                torch.cat([X_, torch.ones(m, 1,
                                        dtype=self.dtype,
                                        device=self.device)], dim=1),
                p=2, dim=1)
        elif self.encoding == "stereo":
            factor = (1.0 / (X_sq_sum + 1)).unsqueeze(1)
            X_prime = factor * torch.cat(
                [2 * X_, (X_sq_sum - 1).unsqueeze(1)], dim=1)
        else:
            raise ValueError('encoding must be "amplit", "stereo" or None')

        return X_prime



    # --------------------------- fitting ------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Compute PGM elements from the training data (Sec. 5)."""
        # Save raw labels & classes
        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.y_train = y_int

        # Encoding training data
        
        X_prime = self.X_prime_func(X, X.shape[0])
        

        self.X_prime_train = X_prime 

        n = X_prime.shape[0]

        # Gram Matrix (Eq. 43)
        G = X_prime @ X_prime.T 

        # Tensor copies (Eq. 55)
        if self.n_copies > 1:
            G = G ** self.n_copies

        #print("Making G pseudo-inverse...")
        #self.G_inv_sqrt = self.pinvSqrt(G)

        if G.shape[0] != G.shape[1]:
            raise ValueError(f"Square matrix expected! Got {G.shape} instead.")
                
        # Eq. 50‑52: Pseudo-inverse square root
        G = G.cpu()
        G = G.double()  # float64 (safer with LAPACK)
        G = (G + G.T) / 2  # need to enforce symmetry

        #print("Running eigh...")
        self.lam, self.E = torch.linalg.eigh(G)

        #print("Regularizing...")
        positive_mask = self.lam > self.tol
        self.E = self.E[:, positive_mask]
        self.lam_inv_sqrt = self.lam[positive_mask].pow(-0.5)

        #print("Fitting completed.")

        return self

    # --------------------------- prediction --------------------------- #

    def _scores(self, X_prime: np.ndarray) -> np.ndarray:
        """Compute raw PGM scores f_k(z) for each sample z (Eq. 54)."""
        if self.X_prime_train is None:
            raise ValueError("Model is not fitted.")

        # W vectors (Eq. 52)
        w_vec = self.X_prime_train @ X_prime.T  # shape (n_train, n_test)

        # Tensor copies (Eq. 55)
        if self.n_copies > 1:
            w_vec = w_vec ** self.n_copies
        
        self.w_vec = w_vec
        ###################################################
        v = self.apply_pinvSqrt_lowrank()
        ####################################################

        v = v.cpu().numpy()

        # For each class k, sum v_i² over i ∈ class k  (Eq. 54, semplified)

        scores = np.empty((len(self.classes_), X_prime.shape[0]))
        
        #for k in trange(len(self.classes_), desc="Computing scores"):
        for k in range(len(self.classes_)):
            idx = np.where(self.y_train == k)[0]
            scores[k] = np.sum(v[idx] ** 2, axis=0)

        return scores.T  # shape (n_samples, n_classes)

    def predict_proba(self, X: np.ndarray):
        X_prime = self.X_prime_func(X, X.shape[0])
        scores = self._scores(X_prime)

        # The scores already sum ≤1; optionally renormalise to 1
        probs = scores / scores.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


    # --------------------------- utils ------------------------------- #

    def apply_pinvSqrt_lowrank(self):
        """
        Apply the sqrt low-rank pseudo-inverse to a matrix W (shape: [n, m]).
            E: [n, r] - filtered eigenvectors
            lam_inv_sqrt: [r] - values λ_i^{-1/2}
            W: [n, m] - matrix to transform
        """
        W = self.w_vec
        tmp = self.E.T @ W  # shape [r, m]
        tmp = self.lam_inv_sqrt[:, None] * tmp  # broadcasting on columns
        
        return self.E @ tmp # shape [n, m]

    
    
