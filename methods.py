
import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA


# Usual MCA (classical MCA) on one-hot encoded categorical data
class ClassicMCA:
    def __init__(self, N, k, eps=1e-12):
        self.N = N
        self.k = k
        self.eps = eps
        self.F = None

    def fit(self):
        result = classical_mca(self.N, k=self.k, eps=self.eps)
        self.S = result["S"]
        self.r = result["r"]
        self.c = result["c"]
        self.Dr_inv_sqrt = result["Dr_inv_sqrt"]
        self.Dc_inv_sqrt = result["Dc_inv_sqrt"]
        self.U = result["U"]
        self.V = result["V"]
        self.singular_values = result["singular_values"]
        self.eigenvalues = result["eigenvalues"]
        self.F = result["F"]
        self.G = result["G"]
        return self

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F


# PCA baseline on one-hot encoded categorical data (indicator PCA)
class IndicatorPCA:
    def __init__(self, N, k):
        self.N = N
        self.k = k
        self.F = None

    def fit(self):
        N = np.asarray(self.N, dtype=float)
        n_components = min(self.k, min(N.shape))
        self.pca = PCA(n_components=n_components, whiten=True)
        self.F = self.pca.fit_transform(N)
        self.eigenvalues = self.pca.explained_variance_[:n_components]
        self.explained_variance_ratio = self.pca.explained_variance_ratio_[:n_components]
        return self

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F

    def as_dict(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return {
            "F": self.F[:, : self.k],
            "eigenvalues": self.eigenvalues,
            "explained_variance_ratio": self.explained_variance_ratio,
        }


def indicator_pca(N, k=10):
    return IndicatorPCA(N, k).fit().as_dict()


# proposed approach 1
class HardFairMCA:
    def __init__(self, N, Z, k, eps=1e-12):
        self.N = N
        self.Z = Z
        self.k = k
        self.eps = eps
        self.F = None

    def fit(self):
        result = hard_fair_mca(self.N, self.Z, k=self.k, eps=self.eps)
        self.S = result["S"]
        self.A_Z = result["A_Z"]
        self.PZ = result["PZ"]
        self.eigenvalues = result["eigenvalues"]
        self.F = result["F"]
        self.G = result["G"]
        return self

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F
    

# proposed approach 2
class SoftFairMCA:
    def __init__(self, N, Z, k, lambda_fair, eps=1e-12):
        self.N = N
        self.Z = Z
        self.k = k
        self.lambda_fair = lambda_fair
        self.eps = eps
        self.F = None

    def fit(self):
        result = soft_fair_mca(self.N, self.Z, k=self.k, lambda_fair=self.lambda_fair, eps=self.eps)
        self.S = result["S"]
        self.A_Z = result["A_Z"]
        self.B = result["B"]
        self.eigenvalues = result["eigenvalues"]
        self.F = result["F"]
        self.G = result["G"]
        return self

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F

# proposed approach 3    
class onlineFairMCA:
    def __init__(self, N, Z, k, lambda_fair):
        self.N = N
        self.Z = Z
        self.k = k
        self.lambda_fair = lambda_fair
        self.F = None
    
    def pca_proc(self, N):
        self.Sigma = np.cov(N.T)
        Pi_N = np.eye(self.d) - self.lambda_fair * self.N @ self.N.T
        Sigma_ = Pi_N @ self.Sigma @ Pi_N
        eigval, eigvec = np.linalg.eigh(Sigma_)
        _indices_V = np.argsort(np.abs(eigval))[-self.k:][::-1]
        self.V = eigvec[:,_indices_V]
    def fair_subspace(self, N, Z):
        self.mean_group = [np.mean([N[Z==a]],0) for a in range(2)]
        self.f = self.mean_group[1] - self.mean_group[0]
        self.f /= (np.linalg.norm(self.f) + 1e-8)
        self.N = np.expand_dims(self.f, -1)

    def fit(self):
        X = np.asarray(self.N)
        A = np.asarray(self.Z)
        self.fair_subspace(X, A)
        ## PCA OPTIMIZATION
        self.pca_proc(X)

        ## evaluate explained variance
        self.eigval_Sigma, self.eigvec_Sigma = np.linalg.eigh(self.Sigma)
        optimal_explained_variance = self.eigval_Sigma.sum()
        self.explained_variance_ratio = np.trace(self.V.T @ self.Sigma @ self.V) / optimal_explained_variance

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F
    


# @pelegrina2023novel
class uFPCA:
    def __init__(self, N, Z, k):
        self.N = N
        self.Z = Z
        self.k = k
        self.F = None

    def fit(self):
        # Implement the u-FPCA algorithm here
        # This is a placeholder for the actual implementation
        self.F = np.random.rand(self.N.shape[0], self.k)  # Random initialization for demonstration

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F
    

# @shen2026fair
class EigOpt:
    def __init__(self, N, Z, k):
        self.N = N
        self.Z = Z
        self.k = k
        self.F = None

    def fit(self):
        # Implement the EigOpt algorithm here
        # This is a placeholder for the actual implementation
        self.F = np.random.rand(self.N.shape[0], self.k)  # Random initialization for demonstration

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F
    


# @kleindessner2023efficient
class FairPCA:
    def __init__(self, N, Z, k, eps=1e-12):
        self.N = N
        self.Z = Z
        self.k = k
        self.eps = eps
        self.F = None

    def fit(self):
        result = fair_pca(self.N, self.Z, k=self.k)
        self.F = result["F"]
        self.eigenvalues = result["eigenvalues"]
        self.explained_variance_ratio = result["explained_variance_ratio"]
        return self

    def transform(self):
        if self.F is None:
            raise ValueError("Model has not been fitted yet.")
        return self.F
    




class FPCA:
    """
    Golden-section Fair PCA (mono-objective constrained version).

    This mirrors the MATLAB logic:
    - Baseline PCA per target dimension j
    - Build dif_cov based on privileged group under baseline PCA
    - Golden-section search on alpha in [0, 1] for:
        X(alpha) = alpha * covM + (1 - alpha) * dif_cov
      with fairness objective (recB - recA)^2 and reconstruction constraints.
    """

    def __init__(self,N, Z,k, tol=1e-6, constraint_eps=1e-6):
        self.N = N
        self.Z = Z
        self.k = k
        self.m_used = k
        self.tol = tol
        self.constraint_eps = constraint_eps
        self._is_fitted = False
        self.F = None

    @staticmethod
    def _reconstruction_error(X, X_proj):
        # Equivalent to re(X, X_proj) in MATLAB if re is squared Frobenius norm.
        D = X - X_proj
        return float(np.sum(D * D))

    @staticmethod
    def _top_eigvecs_desc(X, j):
        # X is symmetric in this method.
        evals, evecs = np.linalg.eigh(X)
        order = np.argsort(evals)[::-1]
        V = evecs[:, order]
        return V[:, :j]

    def _metrics_for_alpha(
        self, alpha, covM, dif_cov, M, A_orig, B_orig, n, na, nb, j
    ):
        Xw = alpha * covM + (1.0 - alpha) * dif_cov
        Vj = self._top_eigvecs_desc(Xw, j)
        proj = Vj @ Vj.T

        recA = self._reconstruction_error(A_orig, A_orig @ proj) / na
        recB = self._reconstruction_error(B_orig, B_orig @ proj) / nb
        rec = self._reconstruction_error(M, M @ proj) / n
        rec_difs = (recB - recA) ** 2
        return recA, recB, rec, rec_difs, proj

    def _golden_section_constrained(
        self, covM, dif_cov, M, A_orig, B_orig, n, na, nb, j, maxRec
    ):
        alpha0, alpha1 = 0.0, 1.0
        g_ratio = (np.sqrt(5.0) + 1.0) / 2.0

        while abs(alpha1 - alpha0) > self.tol:
            a0 = alpha1 - (alpha1 - alpha0) / g_ratio
            a1 = alpha0 + (alpha1 - alpha0) / g_ratio

            recA0, recB0, _, recd0, _ = self._metrics_for_alpha(
                a0, covM, dif_cov, M, A_orig, B_orig, n, na, nb, j
            )
            recA1, recB1, _, recd1, _ = self._metrics_for_alpha(
                a1, covM, dif_cov, M, A_orig, B_orig, n, na, nb, j
            )

            if recd0 < recd1:
                if (
                    recA0 <= maxRec + self.constraint_eps
                    and recB0 <= maxRec + self.constraint_eps
                ):
                    alpha1 = a1
                else:
                    alpha0 = a0
            else:
                alpha0 = a0

        alpha = 0.5 * (alpha0 + alpha1)
        recA, recB, rec, rec_difs, proj = self._metrics_for_alpha(
            alpha, covM, dif_cov, M, A_orig, B_orig, n, na, nb, j
        )
        return alpha, recA, recB, rec, rec_difs, proj

    def fit(self):
        """
        Fit FPCA across j = 1..m_used.

        Parameters
        - M: full data matrix, shape (n, m)
        - Z: binary group indicator matrix, shape (n, 2)
        """
        M = np.asarray(self.N, dtype=float)
        Z = np.asarray(self.Z, dtype=int)

        A_orig = np.asarray(M[Z[:, 0]], dtype=float)
        #A_orig = np.asarray(A_orig, dtype=float)
        B_orig = np.asarray(M[Z[:, 1]], dtype=float)
        #B_orig = np.asarray(B_orig, dtype=float)

        if M.ndim != 2 or A_orig.ndim != 2 or B_orig.ndim != 2:
            raise ValueError("M, A_orig, B_orig must be 2D arrays.")
        if A_orig.shape[1] != M.shape[1] or B_orig.shape[1] != M.shape[1]:
            raise ValueError("A_orig and B_orig must have same number of columns as M.")

        n, m = M.shape
        na = A_orig.shape[0]
        nb = B_orig.shape[0]

        m_used = m if self.m_used is None else int(min(self.m_used, m))
        if m_used <= 0:
            raise ValueError("m_used must be positive.")

        # Baseline PCA coefficients, equivalent role to MATLAB coeff = pca(M)
        pca = PCA(n_components=m)
        pca.fit(M)
        coeff = pca.components_.T  # shape (m, m), columns are principal directions

        covM = (M.T @ M) / n

        self.alpha_ = np.zeros(m_used)
        self.recA_ = np.zeros(m_used)
        self.recB_ = np.zeros(m_used)
        self.rec_ = np.zeros(m_used)
        self.rec_difs_ = np.zeros(m_used)

        self.rec_pca_ = np.zeros(m_used)
        self.recA_pca_ = np.zeros(m_used)
        self.recB_pca_ = np.zeros(m_used)
        self.rec_difs_pca_ = np.zeros(m_used)

        self.proj_pca_ = []
        self.proj_fpca_ = []

        for jj in range(1, m_used + 1):
            # Classical PCA projection for j components
            Vp = coeff[:, :jj]
            proj_pca = Vp @ Vp.T
            self.proj_pca_.append(proj_pca)

            rec_pca = self._reconstruction_error(M, M @ proj_pca) / n
            recA_pca = self._reconstruction_error(A_orig, A_orig @ proj_pca) / na
            recB_pca = self._reconstruction_error(B_orig, B_orig @ proj_pca) / nb
            rec_difs_pca = (recB_pca - recA_pca) ** 2

            self.rec_pca_[jj - 1] = rec_pca
            self.recA_pca_[jj - 1] = recA_pca
            self.recB_pca_[jj - 1] = recB_pca
            self.rec_difs_pca_[jj - 1] = rec_difs_pca

            # Privileged-group logic from MATLAB
            if recA_pca <= recB_pca:
                dif_cov = (B_orig.T @ B_orig) / nb - (A_orig.T @ A_orig) / na
            else:
                dif_cov = (A_orig.T @ A_orig) / na - (B_orig.T @ B_orig) / nb

            maxRec = max(recA_pca, recB_pca)

            alpha, recA, recB, rec, rec_difs, proj = self._golden_section_constrained(
                covM, dif_cov, M, A_orig, B_orig, n, na, nb, jj, maxRec
            )

            self.alpha_[jj - 1] = alpha
            self.recA_[jj - 1] = recA
            self.recB_[jj - 1] = recB
            self.rec_[jj - 1] = rec
            self.rec_difs_[jj - 1] = rec_difs
            self.proj_fpca_.append(proj)

        self.m_used_ = m_used
        self.n_features_in_ = m
        self._M_fit_ = M
        self._is_fitted = True
        print(f"FPCA fit completed for m_used={self.m_used_}.")
        print(f"Projection matrix shape: {self.proj_fpca_[self.m_used_-1].shape}")
        self.F = M @ self.proj_fpca_[self.m_used_-1]  # Store the last projection for convenience
        return self

    def transform(self, X, n_components=None, use_fpca=True):
        """
        Project X using learned projection of chosen dimension.

        Parameters
        - X: data matrix shape (n_samples, m)
        - n_components: target number of components j (default: m_used used in fit)
        - use_fpca: True -> Fair PCA projection, False -> baseline PCA projection
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D with the same number of features as fit data.")

        j = self.m_used_ if n_components is None else int(n_components)
        if j < 1 or j > self.m_used_:
            raise ValueError(f"n_components must be in [1, {self.m_used_}].")

        proj = self.proj_fpca_[j - 1] if use_fpca else self.proj_pca_[j - 1]
        return X @ proj


from scipy.optimize import minimize_scalar


class FPCAviaEigOpt:
    """
    Fair PCA via Eigenvalue Optimization 
    """

    def __init__(self, N, Z, k, tol=1e-6):
        self.r = int(k)
        self.tol = float(tol)
        self.components_ = None   # U in MATLAB (d x r)
        self.t_star_ = None
        self.HA_ = None
        self.HB_ = None
        self._is_fitted = False
        self.F = None
        self.N = N
        self.Z = Z

    @staticmethod
    def _smallest_r_eig_sum(H, r):
        # Sum of the smallest r eigenvalues (phiFun in MATLAB)
        evals = np.linalg.eigvalsh(H)  # ascending for symmetric matrices
        return float(np.sum(evals[:r]))

    @staticmethod
    def _smallest_r_eigvecs(H, r):
        # Return eigenvectors associated with smallest r eigenvalues
        evals, evecs = np.linalg.eigh(H)  # evals ascending
        return evecs[:, :r]

    def fit(self):
        N = np.asarray(self.N, dtype=float)
        Z = np.asarray(self.Z, dtype=int)
        A_orig = np.asarray(N[Z[:, 0]], dtype=float)
        #A_orig = np.asarray(A_orig, dtype=float)
        B_orig = np.asarray(N[Z[:, 1]], dtype=float)
        A = np.asarray(A_orig, dtype=float)
        B = np.asarray(B_orig, dtype=float)

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D arrays.")
        if A.shape[1] != B.shape[1]:
            raise ValueError("A and B must have the same number of features.")

        d = A.shape[1]
        na = A.shape[0]
        nb = B.shape[0]

        if not (1 <= self.r <= d):
            raise ValueError(f"r must be in [1, {d}], got {self.r}.")

        # MATLAB: singular values of A and B
        # Use full SVD singular values (descending order in NumPy)
        sigval_a = np.linalg.svd(A, compute_uv=False)
        sigval_b = np.linalg.svd(B, compute_uv=False)

        # SA = sum of squares of largest r singular values
        SA = float(np.sum(sigval_a[:self.r] ** 2))
        SB = float(np.sum(sigval_b[:self.r] ** 2))

        # HA and HB
        self.HA_ = (SA / self.r * np.eye(d) - A.T @ A) / na
        self.HB_ = (SB / self.r * np.eye(d) - B.T @ B) / nb

        # H(t) = t*HA + (1-t)*HB
        def H_of_t(t):
            return t * self.HA_ + (1.0 - t) * self.HB_

        # phiFun(t) = sum of smallest r eigenvalues of H(t)
        def phi_fun(t):
            return self._smallest_r_eig_sum(H_of_t(t), self.r)

        # MATLAB fminbnd on -phiFun over [0,1] (Brent bounded)
        res = minimize_scalar(
            lambda t: -phi_fun(t),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": self.tol},
        )

        self.t_star_ = float(res.x)

        # U = smallest-r eigenvectors of H(t_star)
        H_star = H_of_t(self.t_star_)
        self.components_ = self._smallest_r_eigvecs(H_star, self.r)
        self._is_fitted = True
        self.F = self.N @ self.components_
        return self

    def transform(self, X):
        """
        Project X onto learned fair subspace.
        Returns shape (n_samples, r).
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit(A, B) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.components_.shape[0]}."
            )

        return X @ self.components_

def inv_sqrt_diag(x, eps=1e-12):
    return np.diag(1.0 / np.sqrt(np.maximum(x, eps)))


def standardized_residuals_from_indicator(N, eps=1e-12):
    N = np.asarray(N, dtype=float)
    P = N / N.sum()
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    Dr_inv_sqrt = inv_sqrt_diag(r, eps)
    Dc_inv_sqrt = inv_sqrt_diag(c, eps)
    S = Dr_inv_sqrt @ (P - np.outer(r, c)) @ Dc_inv_sqrt
    return S, r, c, Dr_inv_sqrt, Dc_inv_sqrt


def classical_mca(N, k=10, eps=1e-12):
    S, r, c, Dr_inv_sqrt, Dc_inv_sqrt = standardized_residuals_from_indicator(N, eps)
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    k = min(k, len(s))
    U = U[:, :k]
    V = Vt.T[:, :k]
    s = s[:k]

    F = Dr_inv_sqrt @ U @ np.diag(s)
    G = Dc_inv_sqrt @ V @ np.diag(s)

    return {
        "S": S, "r": r, "c": c,
        "Dr_inv_sqrt": Dr_inv_sqrt, "Dc_inv_sqrt": Dc_inv_sqrt,
        "U": U, "V": V, "singular_values": s,
        "eigenvalues": s**2, "F": F, "G": G
    }


def indicator_pca(N, k=10):
    return IndicatorPCA(N, k).fit().as_dict()


def make_Z_multi_sensitive(N, sensitive_df, eps=1e-12):
    # Build one-hot encoding across all sensitive attributes
    G_blocks = []
    for col in sensitive_df.columns:
        dummies = pd.get_dummies(sensitive_df[col].astype(str), prefix=col, dtype=float)
        G_blocks.append(dummies)

    G_ind = pd.concat(G_blocks, axis=1).values  # n x s_total

    # Map sensitive information into MCA column/modalities space: m x s_total
    Z = N.T @ G_ind

    # Row-wise normalize to reduce pure frequency scale effects
    Z = Z / np.maximum(Z.sum(axis=1, keepdims=True), eps)
    return Z, G_ind

def make_Z_column_space(N, g_binary, eps=1e-12):
    # G_ind is n x 2 (male, female)
    G_ind = np.column_stack([1.0 - g_binary, g_binary])

    # Map individual sensitive info into column/modalities space: m x 2
    Z = N.T @ G_ind

    # Row-wise normalize to reduce pure frequency scale effects
    Z = Z / np.maximum(Z.sum(axis=1, keepdims=True), eps)
    return Z

def hard_fair_mca(N, Z, k=10, eps=1e-12):
    S, r, c, Dr_inv_sqrt, Dc_inv_sqrt = standardized_residuals_from_indicator(N, eps)
    A_Z = Dr_inv_sqrt @ S @ Z  # n x s

    PZ = np.eye(S.shape[0]) - A_Z @ np.linalg.pinv(A_Z.T @ A_Z) @ A_Z.T
    S_fair = PZ @ S

    U, s, Vt = np.linalg.svd(S_fair, full_matrices=False)
    k = min(k, len(s))
    U = U[:, :k]
    V = Vt.T[:, :k]
    s = s[:k]

    F = Dr_inv_sqrt @ U @ np.diag(s)
    G = Dc_inv_sqrt @ V @ np.diag(s)

    return {
        "S": S_fair, "A_Z": A_Z, "PZ": PZ,
        "eigenvalues": s**2, "F": F, "G": G
    }


def soft_fair_mca(N, Z, k=10, lambda_fair=0.001, eps=1e-12):
    S, r, c, Dr_inv_sqrt, Dc_inv_sqrt = standardized_residuals_from_indicator(N, eps)
    A_Z = Dr_inv_sqrt @ S @ Z  # n x s

    B = S @ S.T - lambda_fair * (A_Z @ A_Z.T)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    pos = evals > eps
    evals = evals[pos]
    evecs = evecs[:, pos]

    k = min(k, len(evals))
    mu = evals[:k]
    U = evecs[:, :k]

    F = Dr_inv_sqrt @ U @ np.diag(np.sqrt(mu))

    denom = np.sqrt(np.maximum(mu, eps))
    V = (S.T @ U) / denom[np.newaxis, :]
    G = Dc_inv_sqrt @ V @ np.diag(np.sqrt(mu))

    return {
        "S": S, "A_Z": A_Z, "B": B,
        "eigenvalues": mu, "F": F, "G": G
    }


def fair_pca(N, Z, k=10):
    N = np.asarray(N, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # Sensitive directions in row space (n x s)
    A_Z = N @ Z

    # Project rows of N onto orthogonal complement of sensitive span
    PZ = np.eye(N.shape[0]) - A_Z @ np.linalg.pinv(A_Z.T @ A_Z) @ A_Z.T
    N_fair = PZ @ N

    pca = PCA(n_components=min(k, min(N_fair.shape)))
    F = pca.fit_transform(N_fair)

    return {
        "F": F[:, :k],
        "eigenvalues": pca.explained_variance_[:k],
        "explained_variance_ratio": pca.explained_variance_ratio_[:k],
    }
