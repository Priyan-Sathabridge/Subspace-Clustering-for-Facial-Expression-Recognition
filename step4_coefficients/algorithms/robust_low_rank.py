"""Robust Low-Rank Subspace Clustering for Step 4.

Matched formulation used during Step 11 parameter search:

    min_{C,E} lambda ||C||_* + gamma ||E||_1
    subject to X = X C + E, diag(C)=0

Step 4 supplies samples as rows (n x d). The class L2-normalises sample rows,
then transposes internally to the mathematical convention X in R^(d x n).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import normalize


class RobustLowRankSubspaceClustering:
    name = "Robust Low-Rank Subspace Clustering (ADMM)"

    def __init__(
        self,
        lambda_reg: float,
        gamma: float,
        mu: float = 1.0,
        rho: float = 1.5,
        max_mu: float = 1e6,
        max_iter: int = 1000,
        tol: float = 1e-6,
        zero_diagonal: bool = True,
        l2_normalize_samples: bool = True,
        verbose: bool = False,
    ):
        if lambda_reg < 0 or gamma < 0:
            raise ValueError("lambda_reg and gamma must be non-negative.")
        if mu <= 0 or rho <= 1 or max_mu <= 0:
            raise ValueError("Require mu>0, rho>1 and max_mu>0.")
        if max_iter < 1 or tol <= 0:
            raise ValueError("Require max_iter>=1 and tol>0.")

        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.rho = float(rho)
        self.max_mu = float(max_mu)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.zero_diagonal = bool(zero_diagonal)
        self.l2_normalize_samples = bool(l2_normalize_samples)
        self.verbose = bool(verbose)

        self.coef_ = None
        self.error_ = None
        self.auxiliary_ = None
        self.n_iter_ = None
        self.converged_ = False
        self.constraint_error_ = np.inf
        self.parameter_change_ = np.inf
        self.final_mu_ = None
        self.input_rows_ = None
        self.normalized_rows_ = None

    @staticmethod
    def soft_threshold(A: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(A) * np.maximum(np.abs(A) - tau, 0.0)

    @staticmethod
    def singular_value_thresholding(A: np.ndarray, tau: float) -> np.ndarray:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s = np.maximum(s - tau, 0.0)
        return (U * s) @ Vt

    def compute_coefficients(self, X_rows: np.ndarray) -> np.ndarray:
        X_rows = np.asarray(X_rows, dtype=float)
        if X_rows.ndim != 2 or X_rows.size == 0:
            raise ValueError("X_rows must be a non-empty 2D array (samples x features).")
        if not np.isfinite(X_rows).all():
            raise ValueError("Input features contain NaN or infinite values.")

        self.input_rows_ = X_rows.copy()
        if self.l2_normalize_samples:
            X_rows = normalize(X_rows, norm="l2", axis=1)
        self.normalized_rows_ = X_rows.copy()

        X = X_rows.T  # d x n, matching the Step 11 formulation
        d, n = X.shape

        # J carries the regularised copy of C; E captures sparse corruption.
        # Y1 and Y2 are multipliers for reconstruction and C=J constraints.
        C = np.zeros((n, n), dtype=float)
        J = np.zeros((n, n), dtype=float)
        E = np.zeros((d, n), dtype=float)
        Y1 = np.zeros((d, n), dtype=float)
        Y2 = np.zeros((n, n), dtype=float)

        mu = self.mu
        XtX = X.T @ X
        # The linear system is constant across iterations, so factor it once.
        factor = cho_factor(XtX + np.eye(n), lower=True, check_finite=False)

        constraint_error = np.inf
        parameter_change = np.inf
        self.converged_ = False

        for iteration in range(self.max_iter):
            C_old = C.copy()
            E_old = E.copy()

            # Low-rank coefficient proximal update.
            J = self.singular_value_thresholding(C + Y2 / mu, self.lambda_reg / mu)
            if self.zero_diagonal:
                np.fill_diagonal(J, 0.0)

            # Sparse corruption proximal update.
            E = self.soft_threshold(X - X @ C + Y1 / mu, self.gamma / mu)

            # Quadratic C update.
            B = X.T @ (X - E + Y1 / mu) + J - Y2 / mu
            C = cho_solve(factor, B, check_finite=False)
            if self.zero_diagonal:
                np.fill_diagonal(C, 0.0)

            residual1 = X - X @ C - E
            residual2 = C - J
            Y1 += mu * residual1
            Y2 += mu * residual2

            constraint_error = max(
                np.linalg.norm(residual1, ord="fro"),
                np.linalg.norm(residual2, ord="fro"),
            )
            parameter_change = max(
                np.linalg.norm(C - C_old, ord="fro"),
                np.linalg.norm(E - E_old, ord="fro"),
            )

            if not np.isfinite(constraint_error) or not np.isfinite(parameter_change):
                raise FloatingPointError("Robust LRSC ADMM became non-finite.")

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(
                    f"LRSC iter={iteration:4d} constraint={constraint_error:.3e} "
                    f"change={parameter_change:.3e} mu={mu:.3e}"
                )

            # Require both feasible constraints and stabilised parameters.
            if constraint_error < self.tol and parameter_change < self.tol:
                self.converged_ = True
                break

            mu = min(self.rho * mu, self.max_mu)

        self.coef_ = C
        self.error_ = E
        self.auxiliary_ = J
        self.n_iter_ = iteration + 1
        self.constraint_error_ = float(constraint_error)
        self.parameter_change_ = float(parameter_change)
        self.final_mu_ = float(mu)
        return C
