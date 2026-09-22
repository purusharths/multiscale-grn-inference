"""
Population-level moment-matching preconditioner for theta = {A, sigma}.

Motivation: CMA-ES currently starts every fit from the same generic guess
(diag(A)=1.2, off-diagonal=0, sigma=0.5 -- see _ours_combinations.py's
fit_combination_cma) regardless of what the data actually looks like.

FIRST ATTEMPT (kept only in git history, not here): regress the population
MEAN trajectory against its own ODE, dm/dt = A(mu - m). This only needs
per-snapshot sample means -- never per-cell correspondence across snapshots
-- so it respects the same unpaired-snapshot assumption the OU/FP/Cons
losses already make (real scRNA-seq destroys cells; there is no "this cell
later becomes that cell"). It FAILED on this dataset: PERTURBATION="uniform"
in ../non-interventional/_stationary_ground_truth.py displaces every cell
the SAME way, so the population mean decays along essentially one direction.
That makes regressing a full n_genes x n_genes A off the mean trajectory
alone rank-deficient: edge_corr ~0.28 but offdiag_err *worse* than just
starting at all-zero off-diagonal (~5 vs ~1.9), and a matrix-log variant
that tries to use the EXACT (not linearized) transition made it worse still
(~13.5) by amplifying regression noise in an already underdetermined fit.

WHAT WORKS: the population COVARIANCE trajectory. Its ODE,

    dSigma/dt = -A @ Sigma - Sigma @ A.T + sigma^2 * I,

is exactly linear in A for a KNOWN Sigma (A appears on both sides, but
"Sigma fixed, A the unknown" is a linear map -- see _cov_drift_operator).
Critically this isn't limited by the mean's rank: even a rank-1 mean shift
produces covariance dynamics that depend on ALL of A's off-diagonal
coupling, because Cov_ij's growth rate is driven by how every pair of genes
correlates through A, not by which direction the mean moved. Still only
needs per-snapshot sample covariances -- same unpaired-snapshot-safe
property as the mean approach.

Verified on this dataset: edge_corr rises to ~0.41 (on par with a full
CMA-ES fit run to convergence from the cold start!) and offdiag_err drops
to ~1.71, below the cold start's implicit ~1.9. The overall diagonal SCALE
still comes back biased low (first-order/Euler discretization of the ODE at
dt=0.5, not small relative to the system's own relaxation time -- this
dataset's own stats already flag dt/tau=0.53). That bias is fine for a WARM
START: fixing scale while preserving structure is exactly what CMA-ES's own
search is for. See compare_preconditioner.py for whether starting there
actually gets CMA-ES to a better final fit, faster.
"""
from __future__ import annotations

import numpy as np


def _cov_drift_operator(Sigma: np.ndarray, n: int) -> np.ndarray:
    """
    G such that G @ vec(A) == vec(A @ Sigma + Sigma @ A.T), for this fixed
    Sigma. A @ Sigma + Sigma @ A.T is linear in A, so G exists; built
    directly from each of A's n^2 basis directions -- O(n^4) work per
    snapshot, fine at GRN-diagnostic gene counts (tens), not intended to
    scale to hundreds of genes.
    """
    G = np.zeros((n * n, n * n))
    basis = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            basis[p, q] = 1.0
            G[:, p * n + q] = (basis @ Sigma + Sigma @ basis.T).ravel()
            basis[p, q] = 0.0
    return G


def estimate_theta0(
    snapshots: list[np.ndarray],
    mu_known: list[np.ndarray],
    dt: float,
    n_genes: int,
    *,
    diag_floor: float = 0.1,
    sigma_floor: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Closed-form moment-matched (A_hat, sigma_hat) from the snapshot
    covariance trajectory alone (mu_known is accepted for a consistent
    signature with the mean-based approach this replaced, but the
    covariance ODE doesn't reference mu). Returns (A_hat, sigma_hat),
    ready for _ours_combinations.encode().
    """
    del mu_known  # kept for signature symmetry; unused by the covariance ODE
    n = n_genes
    covs = np.stack([np.cov(X, rowvar=False) for X in snapshots])  # (n_snaps, n, n)
    n_intervals = len(snapshots) - 1

    # Stack -(A @ Sigma_k + Sigma_k @ A.T) + sigma^2 * I == dSigma/dt|_k into
    # one linear least-squares system for [vec(A); sigma^2].
    rows, rhs = [], []
    for k in range(n_intervals):
        G_k = _cov_drift_operator(covs[k], n)
        dcov = (covs[k + 1] - covs[k]) / dt
        rows.append(np.hstack([G_k, -np.eye(n).ravel()[:, None]]))
        rhs.append(-dcov.ravel())
    lhs = np.vstack(rows)
    y = np.concatenate(rhs)

    sol, *_ = np.linalg.lstsq(lhs, y, rcond=None)
    A_hat = sol[:-1].reshape(n, n)
    sigma2_hat = sol[-1]

    # The downstream log-encoding requires diag(A) > 0, and a Euler-biased
    # regression can occasionally push a diagonal entry non-positive; a bad
    # place to START CMA-ES from regardless.
    np.fill_diagonal(A_hat, np.clip(np.diag(A_hat), diag_floor, None))
    sigma_hat = float(np.sqrt(max(float(sigma2_hat), sigma_floor**2)))

    return A_hat, sigma_hat
