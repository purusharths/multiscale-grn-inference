"""
Bayesian Inverse Problem (BIP) warm start for the outer optimisation.

Uses the *linearised* OU model only — consistent with the micro-scale EM
simulator but NOT with the FP/JKO macro-scale loop.  This is intentional:
the BIP provides a good initial A rather than a fully consistent initialiser.

Linearised OU (mean-level):
    m_{k+1} - m_k  ≈  -A (m_k - mu_k) * delta_t

Given T-1 consecutive mean transitions we solve the MAP:
    A_map = argmin_A  ||Y - A C||_F^2 + lambda * ||A||_F^2

where
    C[:, k] = (m_k  - mu_k)          — driving signal, shape (G, T-1)
    Y[:, k] = (m_{k+1} - m_k) / delta_t  — response,       shape (G, T-1)

Closed form:  A_map = Y C^T (C C^T + lambda I)^{-1}

The result is returned as numpy arrays and then converted to torch Parameters
by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from config import GRNConfig


# ---------------------------------------------------------------------------
# Prior (stub — replace with sparsity-inducing prior)
# ---------------------------------------------------------------------------

class Prior(Protocol):
    def log_prob(self, A: np.ndarray) -> float: ...
    def regularisation(self, lam: float) -> float: ...


class GaussianPrior:
    """Standard isotropic Gaussian prior on each entry of A."""

    def log_prob(self, A: np.ndarray) -> float:
        return -0.5 * float(np.sum(A ** 2))

    def regularisation(self, lam: float) -> float:
        # TODO: replace with sparsity-inducing prior (e.g. Laplace / ARD)
        return lam


def make_prior(G: int) -> GaussianPrior:
    """
    Return a prior over the G×G drift matrix A.

    Currently a standard Gaussian prior.
    # TODO: replace with sparsity-inducing prior (e.g. Laplace / ARD)
    """
    return GaussianPrior()


# ---------------------------------------------------------------------------
# MAP estimation
# ---------------------------------------------------------------------------

def bip_initialise(
    dataset: dict[int, np.ndarray],
    mu_list: list[np.ndarray],
    config: GRNConfig,
) -> dict:
    """
    Compute the MAP estimate of A via the linearised OU model.

    Parameters
    ----------
    dataset  : timepoint → (N_k, G) snapshot matrices
    mu_list  : list of G-vectors, mu_list[k] = known intervention mean at t_k
    config   : GRNConfig

    Returns
    -------
    dict with keys:
        "A"         : (G, G) ndarray — MAP drift matrix
        "log_sigma" : float          — log scalar diffusion (initialised to 0)
    """
    timepoints = sorted(dataset.keys())
    T = len(timepoints)
    G = next(iter(dataset.values())).shape[1]

    prior = make_prior(G)

    # mean expressions at each timepoint
    means = [dataset[k].mean(axis=0) for k in timepoints]   # list of (G,) arrays

    # build driving (C) and response (Y) matrices
    C_cols, Y_cols = [], []
    for i in range(T - 1):
        k0, k1 = timepoints[i], timepoints[i + 1]
        m0, m1 = means[i], means[i + 1]
        mu0 = mu_list[i] if mu_list else m0   # fall back to empirical mean

        c = m0 - mu0                              # (G,)
        y = (m1 - m0) / config.bip_delta_t       # (G,)
        C_cols.append(c)
        Y_cols.append(y)

    C = np.column_stack(C_cols)   # (G, T-1)
    Y = np.column_stack(Y_cols)   # (G, T-1)

    lam = prior.regularisation(config.bip_lambda)
    # Ridge: A_map = Y C^T (C C^T + lam I)^{-1}
    CCt = C @ C.T                                         # (G, G)
    A_map = Y @ C.T @ np.linalg.solve(CCt + lam * np.eye(G), np.eye(G))

    return {
        "A": A_map,
        "log_sigma": 0.0,   # diffusion initialised at sigma=1
    }


def theta_to_params(
    theta0: dict,
    G: int,
    device: str,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Convert BIP dict output to torch nn.Parameters."""
    A = torch.nn.Parameter(
        torch.tensor(theta0["A"], dtype=torch.float32, device=device)
    )
    log_sigma = torch.nn.Parameter(
        torch.tensor(theta0["log_sigma"], dtype=torch.float32, device=device)
    )
    return A, log_sigma
