from __future__ import annotations
from itertools import combinations_with_replacement
from math import comb, factorial
from typing import Literal, Optional
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["KPGM"]


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
        Eigenvalue threshold of the pseudo-inverse square root, applied to the
        spectrum of the weighted Gram matrix K = W^{1/2} G^c W^{1/2}. The
        nonzero spectrum of K is that of sigma = sum_j p_j rho_j, so the
        threshold has the same meaning as the `tol` of the primal estimators
        (c-PGM, Rc-PGM): with equal `tol` the three truncate the same
        eigendirections. With empirical priors K = G^c / N; the threshold
        1e-6 on G^c used up to v1.0.1 corresponds to tol = 1e-6 / N here.
    class_weight : None, 'balanced' or dict, default=None
        Prior probabilities p_j of the classes. None: empirical priors
        p_j = #k_j / N. 'balanced': uniform priors p_j = 1 / l. A dict maps
        each class label to a nonnegative weight; the weights are normalized
        to sum to one. Each training sample i of class j enters with weight
        w_i = p_j / #k_j, which reproduces the c-PGM with these priors.
    solver : {'dense', 'features', 'auto'}, default='dense'
        How the spectrum of K is obtained in `fit`.
        'dense': form the N x N matrix K and diagonalize it, O(d N^2 + N^3)
        time and O(N^2) memory (the k-PGM of Cruzeiro et al.).
        'features': let Y (N x d_sym) be the feature matrix whose rows are
        the explicit degree-c features of the homogeneous polynomial kernel
        (the reduced vectors of the Rc-PGM), so that G^c = Y Y^T and
        K = (W^{1/2} Y)(W^{1/2} Y)^T. The thin SVD of W^{1/2} Y gives the
        nonzero spectrum of K and its eigenvectors without forming K,
        in O(N d d_sym + N d_sym min(N, d_sym)) time and O(N d_sym) memory.
        'auto': 'features' when d_sym < N, 'dense' otherwise.
        The two solvers give the same classifier; prediction is unchanged.
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
        class_weight=None,
        solver: Literal["dense", "features", "auto"] = "dense",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_copies = n_copies
        self.encoding = encoding
        self.rescale = rescale
        self.tol = tol
        self.class_weight = class_weight
        self.solver = solver
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


        # Priors: sample i of class j enters sigma with weight
        # w_i = p_j / #k_j, i.e. sigma = Y^T W Y with Y the feature matrix
        # (G^c = Y Y^T). The dual of sigma is K = W^{1/2} G^c W^{1/2}, with the
        # same nonzero spectrum; the kernel vector of a test sample is W^{1/2} w.
        self.sample_weight_ = self._sample_weights(y_int)

        n_train, d_enc = X_prime.shape
        dsym = comb(d_enc + self.n_copies - 1, self.n_copies)
        solver = self.solver
        if solver == "auto":
            solver = "features" if dsym < n_train else "dense"
        if solver not in ("dense", "features"):
            raise ValueError('solver should be "dense", "features" or "auto"')
        self.solver_ = solver

        if solver == "dense":
            # Gram Matrix (Eq. 43)
            G = X_prime @ X_prime.T

            # Tensor copies (Eq. 55)
            if self.n_copies > 1:
                G = G ** self.n_copies

            sqrt_w = self.sample_weight_.sqrt().to(device=G.device, dtype=G.dtype)
            G = sqrt_w[:, None] * G * sqrt_w[None, :]

            # Eq. 50-52: Pseudo-inverse square root
            G = G.cpu()
            G = G.double()  # float64 (safer with LAPACK)
            G = (G + G.T) / 2  # need to enforce symmetry

            self.lam, self.E = torch.linalg.eigh(G)
        else:
            # K = (W^{1/2} Y)(W^{1/2} Y)^T: the left singular vectors of the
            # weighted feature matrix Yw = W^{1/2} Y are the eigenvectors of K,
            # its squared singular values the nonzero eigenvalues. K itself is
            # never formed.
            Y = self._symmetric_features(X_prime).cpu().double()
            Yw = self.sample_weight_.sqrt()[:, None] * Y
            del Y
            U, S, _ = torch.linalg.svd(Yw, full_matrices=False)
            self.lam, self.E = S**2, U
            del Yw

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
        
        sqrt_w = self.sample_weight_.sqrt().to(device=w_vec.device, dtype=w_vec.dtype)
        w_vec = sqrt_w[:, None] * w_vec

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

    def _symmetric_features(self, X_prime):
        """The feature matrix Y (N x d_sym): row i holds the explicit degree-c
        features of the homogeneous polynomial kernel at x_i, coordinate n
        being sqrt(c! / prod_t n_t!) * prod_t x_t^{n_t}, one per occupation
        sequence n of c copies over d levels, so that Y Y^T = G^c."""
        n, d = X_prime.shape
        c = self.n_copies
        columns = []
        for levels in combinations_with_replacement(range(d), c):
            counts = np.bincount(np.asarray(levels, dtype=np.int64), minlength=d)
            norm = factorial(c) / np.prod([factorial(int(k)) for k in counts])
            column = torch.full((n,), float(np.sqrt(norm)), dtype=X_prime.dtype,
                                device=X_prime.device)
            for t, k in enumerate(counts):
                if k > 0:
                    column = column * X_prime[:, t] ** int(k)
            columns.append(column)
        return torch.stack(columns, dim=1)

    def _sample_weights(self, y_int):
        """w_i = p_{j(i)} / #k_{j(i)} for the priors selected by class_weight."""
        counts = np.bincount(y_int, minlength=len(self.classes_)).astype(np.float64)
        if self.class_weight is None:
            priors = counts / counts.sum()
        elif isinstance(self.class_weight, str) and self.class_weight == "balanced":
            priors = np.full(len(self.classes_), 1.0 / len(self.classes_))
        elif isinstance(self.class_weight, dict):
            missing = [c for c in self.classes_ if c not in self.class_weight]
            if missing:
                raise ValueError(f"class_weight has no prior for classes {missing}")
            priors = np.array([float(self.class_weight[c]) for c in self.classes_])
            if np.any(priors < 0) or priors.sum() <= 0:
                raise ValueError("class_weight priors must be nonnegative, not all 0")
            priors = priors / priors.sum()
        else:
            raise ValueError('class_weight should be None, "balanced" or a dict')
        self.priors_ = priors
        return torch.tensor(priors[y_int] / counts[y_int], dtype=torch.float64)

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

    
    
