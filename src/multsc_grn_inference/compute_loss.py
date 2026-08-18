"""
Algorithm 1, function ComputeLoss (paper lines 40-49):

    function ComputeLoss(theta, {chi_tk})
        L_OU   <- sum_{k=1}^{T-1} W2( KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}} )
        L_FP   <- sum_{k=1}^{T-1} W2( FP(chi_tk; theta), chi_{t_{k+1}} )
        L_cons <- sum_{k=1}^{T-1} W2( KDE(Phi_OU(X_tk; theta)), FP(chi_tk; theta) )
        return L_OU + L_FP + L_cons

Phi_OU is OUGeneExpression; FP is FPCellPopulation. L_FP and L_cons both
depend on FPCellPopulation, which is research-scope (JKO, not implemented
-- see fp_cell_population.py) so they cannot be computed for real yet.
L_OU only needs OUGeneExpression + Preprocessing, both implementable now.
"""
from __future__ import annotations

import numpy as np


def compute_loss(
    theta,
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
    raise NotImplementedError
