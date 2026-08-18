"""
Algorithm 1, function FPCellPopulation (paper lines 30-38):

    function FPCellPopulation(chi_tk; theta)
        # Macro scale: evolve population density forward via JKO scheme
        # d/dt p = div[A(c - mu_k) p] + sigma^2 * Delta p
        nu* <- argmin_{nu in P_2(R^G)} [W2^2(nu, chi_tk) + tau * F[nu; theta]]
        # Numerical solution: JKO-ICNN or particle JKO
        return nu*

    where F[nu; theta] = int nu log(nu) dc + int nu(c) Psi[c; theta] dc
          Psi[c; theta] = (1/2)(c - mu_k)^T A (c - mu_k)   [OU potential;
                                                             A not assumed symmetric]
          tau = JKO step size (hyperparameter, not part of theta)
          nu  = optimisation variable: a probability measure in P_2(R^G)
                (finite second moment)

Research-scope: genuine JKO requires solving a Wasserstein-regularised
proximal minimisation (JKO-ICNN or particle-JKO) each step -- not the
KDE-resample-then-integrate shortcut used elsewhere in this codebase, which
is only a particle *approximation* of this scheme. Not implemented yet.
See tests/algorithm/test_fp_cell_population.py for the skip-marked specs
this should satisfy once built.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.theta import Theta


def fp_cell_population(
    chi_tk,
    theta: Theta,
    k: int,
    dt: float,
    *,
    tau: float = 0.1,
    rng: np.random.Generator | None = None,
):
    """nu* <- JKO-step(chi_tk; A, mu_k, sigma, tau). See module docstring."""
    raise NotImplementedError
