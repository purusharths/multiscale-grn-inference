"""
Algorithm 1, function ComputeLoss (paper lines 40-49):

    function ComputeLoss(theta, {chi_tk})
        L_OU   <- sum_{k=1}^{T-1} W2( KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}} )
        L_FP   <- sum_{k=1}^{T-1} W2( FP(chi_tk; theta), chi_{t_{k+1}} )
        L_cons <- sum_{k=1}^{T-1} W2( KDE(Phi_OU(X_tk; theta)), FP(chi_tk; theta) )
        return L_OU + L_FP + L_cons

Phi_OU is OUGeneExpression (micro-scale, needs per-cell tracking); FP is
FPCellPopulation (macro-scale, needs only the estimated density). Each term
is exposed as its own function (loss_ou, loss_fp, loss_cons) so they can be
computed/tested independently; compute_loss sums them, matching the paper.

W2 is approximated via sliced Wasserstein-2 (random 1D projections, order
statistics compared along each) for tractability in > 1 dimension.

loss_fp and loss_cons both call FPCellPopulation, which is research-scope
(JKO, not implemented -- see fp_cell_population.py) so they raise
NotImplementedError transitively, as does compute_loss as a whole. loss_ou
only needs OUGeneExpression + Preprocessing, both implemented, and works now.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta


def _sliced_w2(X: np.ndarray, Y: np.ndarray, n_proj: int, rng: np.random.Generator) -> float:
    """Sliced Wasserstein-2^2 between particle clouds X (N, G) and Y (M, G)."""
    G = X.shape[1]
    dirs = rng.standard_normal((n_proj, G))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    n_min = min(len(X), len(Y))
    t_x = np.linspace(0, 1, len(X))
    t_y = np.linspace(0, 1, len(Y))
    t_q = np.linspace(0, 1, n_min)

    total = 0.0
    for d in dirs:
        xq = np.interp(t_q, t_x, np.sort(X @ d))
        yq = np.interp(t_q, t_y, np.sort(Y @ d))
        total += float(np.mean((xq - yq) ** 2))
    return total / n_proj


def loss_ou(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_OU = sum_k W2( KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}} )."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(X_snapshots) - 1):
        propagated = ou_gene_expression(X_snapshots[k], theta, k, dt, rng=rng)
        propagated_density = preprocessing(propagated)

        n = len(propagated)
        cloud_a = propagated_density.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = chi_snapshots[k + 1].resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)
    return float(total)


def loss_fp(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_FP = sum_k W2( FP(chi_tk; theta), chi_{t_{k+1}} )."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(chi_snapshots) - 1):
        nu_star = fp_cell_population(chi_snapshots[k], theta, k, dt, rng=rng)

        n = len(X_snapshots[k])
        cloud_a = nu_star.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = chi_snapshots[k + 1].resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)
    return float(total)


def loss_cons(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_cons = sum_k W2( KDE(Phi_OU(X_tk; theta)), FP(chi_tk; theta) )."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(X_snapshots) - 1):
        propagated = ou_gene_expression(X_snapshots[k], theta, k, dt, rng=rng)
        propagated_density = preprocessing(propagated)
        nu_star = fp_cell_population(chi_snapshots[k], theta, k, dt, rng=rng)

        n = len(X_snapshots[k])
        cloud_a = propagated_density.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = nu_star.resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)
    return float(total)


def compute_loss(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> dict:
    """
    Returns {"L_OU": ..., "L_FP": ..., "L_cons": ..., "total": L_OU+L_FP+L_cons}.

    X_snapshots   : list of (N, G) observed cell snapshots, length T
    chi_snapshots : list of fitted densities (Preprocessing output), length T
    """
    l_ou = loss_ou(theta, X_snapshots, chi_snapshots, dt, n_proj=n_proj, seed=seed)
    l_fp = loss_fp(theta, X_snapshots, chi_snapshots, dt, n_proj=n_proj, seed=seed)
    l_cons = loss_cons(theta, X_snapshots, chi_snapshots, dt, n_proj=n_proj, seed=seed)
    return {"L_OU": l_ou, "L_FP": l_fp, "L_cons": l_cons, "total": l_ou + l_fp + l_cons}
