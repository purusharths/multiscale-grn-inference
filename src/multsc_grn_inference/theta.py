"""
theta = {A, sigma, {mu_k}} -- Algorithm 1's parameter set (paper, line 4):

    theta_hat = {A_hat, sigma_hat^2, {mu_k}_{k=1}^T}

A     : (G, G) interaction/regulation matrix (not assumed symmetric)
mu    : list of (G,) vectors, one per interval -- the "known intervention
        means" {mu_k} (paper Data, line 2). A stationary system is just the
        same vector repeated for every k.
sigma : scalar diffusion coefficient
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Theta:
    A: np.ndarray
    mu: list[np.ndarray]
    sigma: float

    @property
    def G(self) -> int:
        return self.A.shape[0]

    def mu_at(self, k: int) -> np.ndarray:
        """mu_k for interval [t_k, t_{k+1}] -- the drift target for that interval."""
        return self.mu[k]
