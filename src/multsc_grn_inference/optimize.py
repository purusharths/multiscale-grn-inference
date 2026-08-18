"""
Algorithm 1, function Optimize (paper lines 53-57):

    function Optimize(theta0, {chi_tk})
        # Outer Bilevel optimization over theta
        theta_hat <- argmin_theta ComputeLoss(theta, {chi_tk}), initialised at theta0
        return theta_hat
"""
from __future__ import annotations

import numpy as np


def optimize(
    theta0,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
    maxiter: int = 3000,
):
    """theta_hat <- argmin_theta ComputeLoss(theta, ...), started at theta0."""
    raise NotImplementedError
