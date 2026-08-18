"""
Three-way multiscale GRN loss functions (Algorithm 1).

  L_OU  : W₂(KDE(Φ_OU(X_tₖ; θ)), χ_{tₖ₊₁})
  L_FP  : W₂(FP(χ_tₖ; θ), χ_{tₖ₊₁})
  L_cons: W₂(KDE(Φ_OU(X_tₖ; θ)), FP(χ_tₖ; θ))

X_tₖ   – observed snapshot at tₖ   (N, G) array
χ_tₖ   – population density at tₖ, estimated via Gaussian KDE
Φ_OU   – Euler–Maruyama integration of dc = A(μ − c) dt + σ dW
FP     – particle approximation of the Fokker–Planck push-forward:
         sample from KDE(X_tₖ), then run Φ_OU
W₂     – approximated via sliced Wasserstein distance
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Internal primitives
# ---------------------------------------------------------------------------

def _ou_euler_maruyama(
    X0: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    dt: float,
    n_substeps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler–Maruyama step for dc = A(μ − c) dt + σ dW."""
    X = X0.copy()
    sub_dt = dt / n_substeps
    sqrt_sub_dt = np.sqrt(sub_dt)
    for _ in range(n_substeps):
        drift = (mu - X) @ A.T          # (N, G): A(μ − c) for each particle
        noise = sigma * sqrt_sub_dt * rng.standard_normal(X.shape)
        X = np.maximum(X + drift * sub_dt + noise, 0.0)
    return X


def _kde_sample(
    X: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples from the Gaussian KDE fitted to particle cloud X (N, G)."""
    kde = gaussian_kde(X.T)
    seed_val = int(rng.integers(0, 2**31))
    return np.maximum(kde.resample(n_samples, seed=seed_val).T, 0.0)


def _sliced_w2(
    X: np.ndarray,
    Y: np.ndarray,
    n_proj: int,
    rng: np.random.Generator,
) -> float:
    """Sliced Wasserstein² between particle clouds X (N, G) and Y (M, G)."""
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


# ---------------------------------------------------------------------------
# Single-step helpers (advance state one interval [tₖ, tₖ₊₁])
# ---------------------------------------------------------------------------

def _ou_step(X_tk, X_next, A, mu, sigma, dt, n_substeps, n_proj, rng):
    X_sim = _ou_euler_maruyama(X_tk, A, mu, sigma, dt, n_substeps, rng)
    return _sliced_w2(X_sim, X_next, n_proj, rng)


def _fp_step(X_tk, X_next, A, mu, sigma, dt, n_substeps, n_proj, rng):
    X_kde = _kde_sample(X_tk, len(X_tk), rng)
    X_fp = _ou_euler_maruyama(X_kde, A, mu, sigma, dt, n_substeps, rng)
    return _sliced_w2(X_fp, X_next, n_proj, rng)


def _cons_step(X_tk, A, mu, sigma, dt, n_substeps, n_proj, rng):
    X_ou = _ou_euler_maruyama(X_tk, A, mu, sigma, dt, n_substeps, rng)
    X_kde = _kde_sample(X_tk, len(X_tk), rng)
    X_fp = _ou_euler_maruyama(X_kde, A, mu, sigma, dt, n_substeps, rng)
    return _sliced_w2(X_ou, X_fp, n_proj, rng)


# ---------------------------------------------------------------------------
# Public loss functions  (sum over all intervals [tₖ, tₖ₊₁])
# ---------------------------------------------------------------------------

def ou_loss(
    snapshots: list[np.ndarray],
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    dt: float,
    *,
    n_substeps: int = 1,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_OU = Σₖ W₂(KDE(Φ_OU(X_tₖ; θ)), χ_{tₖ₊₁})."""
    rng = np.random.default_rng(seed)
    return float(sum(
        _ou_step(snapshots[k], snapshots[k + 1], A, mu, sigma, dt, n_substeps, n_proj, rng)
        for k in range(len(snapshots) - 1)
    ))


def fp_loss(
    snapshots: list[np.ndarray],
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    dt: float,
    *,
    n_substeps: int = 1,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_FP = Σₖ W₂(FP(χ_tₖ; θ), χ_{tₖ₊₁})."""
    rng = np.random.default_rng(seed)
    return float(sum(
        _fp_step(snapshots[k], snapshots[k + 1], A, mu, sigma, dt, n_substeps, n_proj, rng)
        for k in range(len(snapshots) - 1)
    ))


def consistency_loss(
    snapshots: list[np.ndarray],
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    dt: float,
    *,
    n_substeps: int = 1,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_cons = Σₖ W₂(KDE(Φ_OU(X_tₖ; θ)), FP(χ_tₖ; θ))."""
    rng = np.random.default_rng(seed)
    return float(sum(
        _cons_step(snapshots[k], A, mu, sigma, dt, n_substeps, n_proj, rng)
        for k in range(len(snapshots) - 1)
    ))


def ou_fp_loss(
    snapshots: list[np.ndarray],
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    dt: float,
    *,
    n_substeps: int = 1,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_OU + L_FP."""
    return (
        ou_loss(snapshots, A, mu, sigma, dt, n_substeps=n_substeps, n_proj=n_proj, seed=seed)
        + fp_loss(snapshots, A, mu, sigma, dt, n_substeps=n_substeps, n_proj=n_proj, seed=seed)
    )


def total_loss(
    snapshots: list[np.ndarray],
    A: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    dt: float,
    *,
    n_substeps: int = 1,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """L_OU + L_FP + L_cons."""
    return (
        ou_loss(snapshots, A, mu, sigma, dt, n_substeps=n_substeps, n_proj=n_proj, seed=seed)
        + fp_loss(snapshots, A, mu, sigma, dt, n_substeps=n_substeps, n_proj=n_proj, seed=seed)
        + consistency_loss(snapshots, A, mu, sigma, dt, n_substeps=n_substeps, n_proj=n_proj, seed=seed)
    )
