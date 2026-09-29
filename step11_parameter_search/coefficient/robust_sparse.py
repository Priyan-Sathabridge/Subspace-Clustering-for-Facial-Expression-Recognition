"""
Robust Sparse Subspace Clustering (Robust SSC) using ADMM.

Matched formulation to the Robust LRSC implementation:

    Robust LRSC:
        min lambda ||C||_* + gamma ||E||_1
        s.t. X = X C + E

    Robust SSC:
        min lambda ||C||_1 + gamma ||E||_1
        s.t. X = X C + E, diag(C) = 0

The nuclear-norm proximal step used by Robust LRSC is replaced by an
entrywise L1 proximal step, so the coefficient matrix is encouraged to be
sparse rather than low rank. The same sparse-error model is retained.

Matrix convention
-----------------
X : (d, n)  features x samples
C : (n, n)  self-representation coefficients
E : (d, n)  sparse corruption/error matrix
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class RobustSparseSubspaceClustering:
    """
    Robust Sparse Subspace Clustering using ADMM.

    Solves
        min_{C,E} lambda_reg * ||C||_1 + gamma * ||E||_1
        subject to X = X C + E,
                   diag(C) = 0.

    An auxiliary variable J separates the non-smooth L1 penalty from the
    quadratic C update:
        min lambda_reg ||J||_1 + gamma ||E||_1
        s.t. X = X C + E,
             C = J.

    Parameters
    ----------
    lambda_reg : float, default=1.0
        Sparsity penalty applied to the self-representation coefficients.
        This is the SSC analogue of the nuclear-norm lambda used in Robust
        LRSC. Larger values promote a sparser representation matrix.

    gamma : float, default=0.1
        Entrywise L1 penalty on the sparse error matrix E. Larger values make
        it more expensive to explain observations through E.

    mu, rho, max_mu : float
        ADMM penalty settings. mu is increased geometrically by rho until
        max_mu is reached.

    max_iter : int
        Maximum ADMM iterations.

    tol : float
        Convergence tolerance applied to both constraint residuals and the
        maximum parameter change.

    zero_diagonal : bool
        Enforce diag(C)=diag(J)=0 to prevent trivial self-representation.
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        gamma: float = 0.1,
        mu: float = 1.0,
        rho: float = 1.5,
        max_mu: float = 1e6,
        max_iter: int = 1000,
        tol: float = 1e-6,
        zero_diagonal: bool = True,
        verbose: bool = False,
    ):
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
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
        self.verbose = bool(verbose)

        self.coef_ = None
        self.error_ = None
        self.auxiliary_ = None
        self.n_iter_ = None
        self.converged_ = False
        self.constraint_error_ = np.inf
        self.parameter_change_ = np.inf
        self.final_mu_ = None

    @staticmethod
    def soft_threshold(A: np.ndarray, tau: float) -> np.ndarray:
        """Entrywise proximal operator for the L1 norm."""
        return np.sign(A) * np.maximum(np.abs(A) - tau, 0.0)

    def fit(self, X: np.ndarray):
        """
        Fit Robust SSC.

        Parameters
        ----------
        X : ndarray of shape (d, n)
            d features/principal components by n samples/images.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array of shape (d, n).")
        if X.size == 0:
            raise ValueError("X must not be empty.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        d, n = X.shape

        C = np.zeros((n, n), dtype=float)
        J = np.zeros((n, n), dtype=float)
        E = np.zeros((d, n), dtype=float)
        Y1 = np.zeros((d, n), dtype=float)
        Y2 = np.zeros((n, n), dtype=float)

        mu = self.mu
        XtX = X.T @ X
        A = XtX + np.eye(n)

        # A is constant and positive definite; factorise once for speed.
        factor = cho_factor(A, lower=True, check_finite=False)

        self.converged_ = False
        constraint_error = np.inf
        parameter_change = np.inf

        for iteration in range(self.max_iter):
            C_old = C.copy()
            E_old = E.copy()

            # ----------------------------------------------------
            # 1. J update: proximal L1 norm on coefficients.
            # ----------------------------------------------------
            J = self.soft_threshold(
                C + Y2 / mu,
                self.lambda_reg / mu,
            )
            if self.zero_diagonal:
                np.fill_diagonal(J, 0.0)

            # ----------------------------------------------------
            # 2. E update: proximal L1 norm on sparse corruption.
            # ----------------------------------------------------
            residual_for_E = X - X @ C + Y1 / mu
            E = self.soft_threshold(
                residual_for_E,
                self.gamma / mu,
            )

            # ----------------------------------------------------
            # 3. C update.
            #
            # (X^T X + I) C
            #   = X^T (X - E + Y1/mu) + J - Y2/mu
            # ----------------------------------------------------
            B = X.T @ (X - E + Y1 / mu) + J - Y2 / mu
            C = cho_solve(factor, B, check_finite=False)
            if self.zero_diagonal:
                np.fill_diagonal(C, 0.0)

            # ----------------------------------------------------
            # 4. Constraint residuals and multipliers.
            # ----------------------------------------------------
            residual1 = X - X @ C - E
            residual2 = C - J

            Y1 = Y1 + mu * residual1
            Y2 = Y2 + mu * residual2

            constraint_error = max(
                np.linalg.norm(residual1, ord="fro"),
                np.linalg.norm(residual2, ord="fro"),
            )
            parameter_change = max(
                np.linalg.norm(C - C_old, ord="fro"),
                np.linalg.norm(E - E_old, ord="fro"),
            )

            if not np.isfinite(constraint_error) or not np.isfinite(parameter_change):
                raise FloatingPointError("ADMM diverged to non-finite values.")

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(
                    f"Iteration {iteration:4d} | "
                    f"constraint={constraint_error:.6e} | "
                    f"change={parameter_change:.6e} | mu={mu:.3e}"
                )

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

        return self

    def fit_predict_coefficients(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.coef_
