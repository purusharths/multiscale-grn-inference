"""
Micro-scale simulation: Euler-Maruyama integration of the OU SDE.

    dc(t) = -A (c(t) - mu_k) dt + sigma dW_t,   t ∈ [t_k, t_{k+1}]

Parameters A and sigma are torch.nn.Parameters so gradients flow back
through the deterministic part of the SDE via the reparameterisation trick.

Noise is pre-sampled and fixed for each forward call; gradients do NOT flow
through the noise draws (pathwise estimator — variance reduction is a TODO).
"""
from __future__ import annotations

import torch


def phi_ou(
    X_tk: torch.Tensor,
    A: torch.Tensor,
    log_sigma: torch.Tensor,
    mu_k: torch.Tensor,
    dt: float,
    n_steps: int,
) -> torch.Tensor:
    """
    Euler-Maruyama forward simulation.

    Parameters
    ----------
    X_tk     : (N, G) initial particle positions (rows = cells)
    A        : (G, G) drift matrix (nn.Parameter)
    log_sigma: ()     log scalar diffusion — σ = exp(log_sigma)
    mu_k     : (G,)   intervention mean for this interval
    dt       : float  step size
    n_steps  : int    number of EM steps

    Returns
    -------
    torch.Tensor of shape (N, G) — particle cloud at t_{k+1}
    """
    sigma = torch.exp(log_sigma)
    x = X_tk.clone()
    sqrt_dt = dt ** 0.5

    for _ in range(n_steps):
        # drift: -A (x - mu)  →  shape (N, G)
        drift = -(x - mu_k) @ A.T          # (N, G) @ (G, G)^T
        noise = torch.randn_like(x) * sigma * sqrt_dt
        x = x + drift * dt + noise

    return x
