import numpy as np
from scipy.linalg import cho_factor, cho_solve


class RobustLowRankSubspaceClustering:
    """
    Robust Low-Rank Subspace Clustering using ADMM.

    Solves
        min_{C,E} lambda_reg ||C||_* + gamma ||E||_1
        subject to X = X C + E,

    with X of shape (d, n), C of shape (n, n), and E of shape (d, n).
    """

    def __init__(
        self,
        lambda_reg=1.0,
        gamma=0.1,
        mu=1.0,
        rho=1.5,
        max_mu=1e6,
        max_iter=1000,
        tol=1e-6,
        zero_diagonal=True,
        verbose=False,
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
        self.n_iter_ = None
        self.converged_ = False
        self.constraint_error_ = np.inf
        self.parameter_change_ = np.inf
        self.final_mu_ = None

    @staticmethod
    def soft_threshold(A, tau):
        return np.sign(A) * np.maximum(np.abs(A) - tau, 0.0)

    @staticmethod
    def singular_value_thresholding(A, tau):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_thresholded = np.maximum(s - tau, 0.0)
        return (U * s_thresholded) @ Vt

    def fit(self, X):
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

        # A is symmetric positive-definite. Factorising it once is much faster
        # than refactorising it with np.linalg.solve at every ADMM iteration.
        factor = cho_factor(A, lower=True, check_finite=False)

        self.converged_ = False

        for iteration in range(self.max_iter):
            C_old = C.copy()
            E_old = E.copy()

            # J update: proximal nuclear norm.
            J = self.singular_value_thresholding(
                C + Y2 / mu,
                self.lambda_reg / mu,
            )
            if self.zero_diagonal:
                np.fill_diagonal(J, 0.0)

            # E update: proximal entrywise L1 norm.
            residual_for_E = X - X @ C + Y1 / mu
            E = self.soft_threshold(residual_for_E, self.gamma / mu)

            # C update.
            B = X.T @ (X - E + Y1 / mu) + J - Y2 / mu
            C = cho_solve(factor, B, check_finite=False)
            if self.zero_diagonal:
                np.fill_diagonal(C, 0.0)

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
        self.n_iter_ = iteration + 1
        self.constraint_error_ = float(constraint_error)
        self.parameter_change_ = float(parameter_change)
        self.final_mu_ = float(mu)

        return self

    def fit_predict_coefficients(self, X):
        self.fit(X)
        return self.coef_
