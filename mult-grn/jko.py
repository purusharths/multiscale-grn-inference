"""
Macro-scale Fokker-Planck / JKO step (particle approximation).

JKO proximal step:

    nu* = argmin_{nu ∈ P_2}  [ W_2^2(nu, chi_hat_k) + tau * F[nu; theta] ]

    F[nu; theta] = ∫ nu log nu dc  +  ∫ nu(c) Psi(c; theta) dc

    Psi(c; theta) = 1/2 (c - mu_k)^T A (c - mu_k)

We approximate nu as a finite particle cloud and solve the inner minimisation
with n_jko_steps gradient steps on the particle positions.

Notes
-----
* A is NOT assumed symmetric.  Psi uses the full A, so it is not a true
  gradient-flow potential unless A is positive-definite.  We use it as a
  heuristic proxy regardless.
* Stop-gradient is applied: the outer loss does not differentiate through
  the JKO inner argmin.  This is intentional and sufficient for the warm-start
  / regularisation role of the FP term.
  # TODO: implement implicit differentiation through JKO if needed
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gaussian_kde

from config import GRNConfig


def _potential(x: torch.Tensor, A: torch.Tensor, mu_k: torch.Tensor) -> torch.Tensor:
    """
    Compute Psi(x; A, mu_k) = 0.5 * sum_i (x_i - mu)^T A (x_i - mu).

    Parameters
    ----------
    x    : (N, G) particle positions
    A    : (G, G) drift matrix
    mu_k : (G,)   intervention mean

    Returns
    -------
    Scalar tensor — total potential across all particles.
    """
    d = x - mu_k                  # (N, G)
    return 0.5 * (d @ A * d).sum()


def _entropy_kde(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximate entropy  H[nu] ≈ -sum_i log nu_hat(x_i)  via kernel density.

    Uses a simple Gaussian kernel with bandwidth chosen by Scott's rule so
    the computation stays on-graph for x.
    """
    N, G = x.shape
    bw = N ** (-1.0 / (G + 4))   # Scott's rule

    # log-sum-exp over N kernel evaluations at each of the N points
    diff = x.unsqueeze(0) - x.unsqueeze(1)          # (N, N, G)
    sq = (diff ** 2).sum(-1) / (2 * bw ** 2)         # (N, N)
    log_k = -sq - 0.5 * G * np.log(2 * np.pi * bw ** 2)
    log_nu = torch.logsumexp(log_k, dim=1) - np.log(N)   # (N,)

    return -log_nu.mean()


def fp(
    chi_hat_tk: gaussian_kde,
    A: torch.Tensor,
    log_sigma: torch.Tensor,
    mu_k: torch.Tensor,
    config: GRNConfig,
    rng: np.random.Generator,
    w2_fn,
) -> torch.Tensor:
    """
    Particle JKO step: approximate nu* given chi_hat at timepoint k.

    Parameters
    ----------
    chi_hat_tk : fitted scipy KDE at timepoint k
    A          : (G, G) drift matrix (used for Psi only; stop-gradient applied)
    log_sigma  : () log diffusion (not used in JKO potential, kept for API symmetry)
    mu_k       : (G,) intervention mean
    config     : GRNConfig
    rng        : numpy RNG for reproducible particle sampling
    w2_fn      : callable(X, Y) → scalar — W2 distance function

    Returns
    -------
    torch.Tensor of shape (n_particles, G) — the JKO-stepped particle cloud.
    The result is detached from the computation graph (stop-gradient).
    """
    N = config.n_particles
    device = config.device

    # sample initial particles from chi_hat_k
    seed = int(rng.integers(0, 2**31))
    x0 = torch.tensor(
        chi_hat_tk.resample(N, seed=seed).T,
        dtype=torch.float32,
        device=device,
    )

    # reference samples (fixed) for the W2 fidelity term
    ref = x0.detach().clone()

    # stop-gradient on A for the inner loop
    A_sg = A.detach()
    mu_sg = mu_k.detach()

    # particle positions to optimise
    particles = x0.clone().requires_grad_(True)
    inner_opt = torch.optim.Adam([particles], lr=config.jko_lr)

    for _ in range(config.n_jko_steps):
        inner_opt.zero_grad()

        # fidelity: W2^2(nu_particles, chi_hat_k)
        fidelity = w2_fn(particles, ref) ** 2

        # free energy: entropy + potential
        free_energy = (
            _entropy_kde(particles)
            + _potential(particles, A_sg, mu_sg) / max(N, 1)
        )

        loss = fidelity + config.tau * free_energy
        loss.backward()
        inner_opt.step()

    return particles.detach()
