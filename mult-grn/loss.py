"""
Loss computation for multiscale GRN inference.

L(theta) = L_OU  +  L_FP  +  L_cons

L_OU   = sum_{k=1}^{T-1}  W2( KDE(Phi_OU(X_{t_k}; theta)),  chi_hat_{t_{k+1}} )
L_FP   = sum_{k=1}^{T-1}  W2( FP(chi_hat_{t_k}; theta),     chi_hat_{t_{k+1}} )
L_cons = sum_{k=1}^{T-1}  W2( KDE(Phi_OU(X_{t_k}; theta)),  FP(chi_hat_{t_k}; theta) )

W2 is approximated via geomloss SamplesLoss("sinkhorn") by default,
or via POT exact discrete OT when config.w2_backend == "exact".

Swapping the backend requires only changing GRNConfig.w2_backend.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gaussian_kde

from config import GRNConfig
from jko import fp
from preprocess import sample_chi_hat
from simulate import phi_ou


# ---------------------------------------------------------------------------
# W2 backend abstraction
# ---------------------------------------------------------------------------

def _make_w2_fn(config: GRNConfig):
    """Return a callable  w2(X, Y) → scalar tensor  based on config.w2_backend."""

    if config.w2_backend == "sinkhorn":
        try:
            from geomloss import SamplesLoss
        except ImportError as e:
            raise ImportError(
                "geomloss is required for w2_backend='sinkhorn'. "
                "Install via: pip install geomloss"
            ) from e
        _loss_fn = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur)

        def w2_sinkhorn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            return _loss_fn(X, Y)

        return w2_sinkhorn

    elif config.w2_backend == "exact":
        try:
            import ot
        except ImportError as e:
            raise ImportError(
                "POT is required for w2_backend='exact'. "
                "Install via: pip install POT"
            ) from e

        def w2_exact(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            X_np = X.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()
            n, m = len(X_np), len(Y_np)
            a = np.ones(n) / n
            b = np.ones(m) / m
            M = ot.dist(X_np, Y_np, metric="sqeuclidean")
            val = float(ot.emd2(a, b, M)) ** 0.5
            return torch.tensor(val, dtype=X.dtype, device=X.device)

        return w2_exact

    else:
        raise ValueError(f"Unknown w2_backend: '{config.w2_backend}'. Use 'sinkhorn' or 'exact'.")


# ---------------------------------------------------------------------------
# Per-interval loss components
# ---------------------------------------------------------------------------

def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Draw n rows from X (with replacement if needed) and return as float32 tensor."""
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32)


def compute_loss(
    A: torch.Tensor,
    log_sigma: torch.Tensor,
    dataset: dict[int, np.ndarray],
    chi_hat: dict[int, gaussian_kde],
    mu_list: list[torch.Tensor],
    config: GRNConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the full multiscale loss L(theta).

    Parameters
    ----------
    A         : (G, G) drift matrix — nn.Parameter
    log_sigma : ()     log scalar diffusion — nn.Parameter
    dataset   : timepoint → (N_k, G) snapshot matrices (numpy)
    chi_hat   : timepoint → scipy KDE
    mu_list   : list of (G,) tensors, mu_list[k] = known intervention mean at t_k
    config    : GRNConfig
    rng       : numpy RNG

    Returns
    -------
    total_loss : scalar tensor (differentiable w.r.t. A, log_sigma)
    breakdown  : dict with float values for L_OU, L_FP, L_cons
    """
    w2 = _make_w2_fn(config)
    device = config.device

    timepoints = sorted(dataset.keys())
    T = len(timepoints)

    L_ou = torch.tensor(0.0, device=device)
    L_fp = torch.tensor(0.0, device=device)
    L_cons = torch.tensor(0.0, device=device)

    for i in range(T - 1):
        k0, k1 = timepoints[i], timepoints[i + 1]
        mu_k = mu_list[i].to(device)

        # --- OU forward pass ---
        X0 = _subsample(dataset[k0], config.n_particles, rng).to(device)
        X_ou = phi_ou(X0, A, log_sigma, mu_k, config.dt, config.n_em_steps)

        # --- target: samples from chi_hat_{k+1} ---
        X_next = torch.tensor(
            sample_chi_hat(chi_hat[k1], config.n_particles, rng),
            dtype=torch.float32,
            device=device,
        )

        # --- FP / JKO step (stop-gradient through inner argmin) ---
        X_fp = fp(chi_hat[k0], A, log_sigma, mu_k, config, rng, w2).to(device)

        L_ou   = L_ou   + w2(X_ou, X_next)
        L_fp   = L_fp   + w2(X_fp, X_next)
        L_cons = L_cons + w2(X_ou, X_fp)

    total = L_ou + L_fp + L_cons
    breakdown = {
        "L_OU":   float(L_ou.detach()),
        "L_FP":   float(L_fp.detach()),
        "L_cons": float(L_cons.detach()),
    }
    return total, breakdown
