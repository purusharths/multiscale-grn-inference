"""
Algorithm 1, function OUGeneExpression (paper lines 20-28):

    function OUGeneExpression(X_tk; theta)
        # Micro scale: simulates OU forward one timestep via Euler-Maruyama
        # dc(t) = A(mu_k - c(t)) dt + sigma dW_t,  t in [t_k, t_{k+1}]
        for i <= N do
            c_j(t_k) <- X_tk[j, :]     initialise from observed cell j
            Integrate SDE forward to t_{k+1}
        return {c_j(t_{k+1})}_{j=1}^N

Advances every observed cell in X_tk one interval forward under the OU SDE,
independently per particle (no cross-particle interaction). This is the
micro-scale forward model Phi_OU referenced throughout ComputeLoss.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.theta import Theta


def ou_gene_expression(
    X_tk: np.ndarray,
    theta: Theta,
    k: int,
    dt: float,
    *,
    n_substeps: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    {c_j(t_{k+1})}_{j=1}^N <- Euler-Maruyama(X_tk; A, mu_k, sigma, dt).

    X_tk : (N, G) cells observed at t_k, used as initial condition c_j(t_k)
    k    : interval index -- selects theta.mu_at(k) as the drift target mu_k
    """
    raise NotImplementedError
