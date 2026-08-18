"""
Spec for Algorithm 1's FPCellPopulation (paper lines 30-38):

    function FPCellPopulation(chi_tk; theta)
        # d/dt p = div[A(c - mu_k) p] + sigma^2 * Delta p
        nu* <- argmin_{nu in P_2(R^G)} [W2^2(nu, chi_tk) + tau * F[nu; theta]]
        return nu*

    F[nu; theta] = int nu log(nu) dc + int nu(c) Psi[c; theta] dc
    Psi[c; theta] = (1/2)(c - mu_k)^T A (c - mu_k)

Research-scope, not implemented (see fp_cell_population.py's docstring):
genuine JKO needs a Wasserstein-regularised proximal solve (JKO-ICNN or
particle-JKO) each step, not the KDE-resample-then-integrate shortcut used
elsewhere in this codebase (which only approximates it). Skip-marked so
tests/algorithm/ collects cleanly; un-skip and fill in once
fp_cell_population() is implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.skip(
    reason="FPCellPopulation (genuine JKO Wasserstein-gradient-flow step) is "
           "research-scope, not implemented yet"
)


def test_nu_star_objective_is_no_worse_than_staying_at_chi_tk():
    """nu* minimises W2^2(nu, chi_tk) + tau*F[nu;theta] over all nu in P_2(R^G);
    chi_tk itself is a feasible candidate, so the objective at nu* must be
    <= the objective at nu=chi_tk. True for ANY correct argmin regardless of
    theta -- the most basic invariant a JKO solver must satisfy."""
    raise NotImplementedError


def test_nu_star_mean_matches_ou_gene_expression_ensemble_mean():
    """Jordan-Kinderlehrer-Otto: iterating the JKO step is a time-discretisation
    of the Fokker-Planck PDE that OUGeneExpression's SDE is the particle
    formulation of. For small tau, nu*'s mean should track the same target as
    ou_gene_expression()'s ensemble mean over one interval."""
    raise NotImplementedError


def test_evolution_over_time_matches_euler_maruyama_ensemble_moments():
    """Repeated FPCellPopulation steps across several intervals should track
    the ensemble mean/covariance produced by direct OUGeneExpression
    simulation of the same dynamics (mirrors ComputeLoss's L_cons intuition)."""
    raise NotImplementedError


def test_smaller_tau_converges_faster_to_the_fokker_planck_solution():
    """JKO's classical convergence result: as the step size tau -> 0, the
    scheme's iterates converge to the true Fokker-Planck solution. A coarser
    tau should show more discretisation error than a finer one, for a fixed
    total elapsed time."""
    raise NotImplementedError
