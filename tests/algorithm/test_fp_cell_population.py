"""
Spec for Algorithm 1's FPCellPopulation (paper lines 30-38):

    function FPCellPopulation(chi_tk; theta)
        # d/dt p = div[A(c - mu_k) p] + sigma^2 * Delta p
        nu* <- argmin_{nu in P_2(R^G)} [W2^2(nu, chi_tk) + tau * F[nu; theta]]
        return nu*

    F[nu; theta] = int nu log(nu) dc + int nu(c) Psi[c; theta] dc
    Psi[c; theta] = (1/2)(c - mu_k)^T A (c - mu_k)

Implemented as particle-JKO (see fp_cell_population.py's docstring for the
full justification and citations): since Psi is exactly the known OU
potential (not learned), nu* coincides with resampling chi_tk and
propagating under the same Euler-Maruyama step OUGeneExpression uses. That
makes the moment-matching properties below directly testable now. The two
tests that need the free-energy functional F[nu;theta] evaluated on a
particle cloud (not implemented -- would need a differential-entropy
estimator) stay skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _ground_truth import draw_cross_section, make_stationary_sim

N_CELLS = 2000
DT = 0.3
N_SNAPS = 4

_SIM = make_stationary_sim(seed=42)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15


def _theta(n_intervals: int = 1) -> Theta:
    return Theta(A=TRUE_A, mu=[TRUE_MU] * n_intervals, sigma=TRUE_SIGMA)


@pytest.mark.skip(
    reason="Needs F[nu;theta] (differential entropy + potential integral) "
           "evaluated on a particle cloud -- not implemented"
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
    formulation of. nu*'s mean should track the same target as
    ou_gene_expression()'s ensemble mean over one interval, up to the extra
    KDE-resampling noise this implementation introduces."""
    X_tk = draw_cross_section(_SIM, N_CELLS)
    chi_tk = preprocessing(X_tk)
    theta = _theta()

    nu_star = fp_cell_population(chi_tk, theta, k=0, dt=DT, rng=np.random.default_rng(1))
    direct = ou_gene_expression(X_tk, theta, k=0, dt=DT, rng=np.random.default_rng(1))

    nu_star_mean = nu_star.resample(N_CELLS, seed=2).T.mean(axis=0)
    np.testing.assert_allclose(nu_star_mean, direct.mean(axis=0), atol=0.05)


def test_evolution_over_time_matches_euler_maruyama_ensemble_moments():
    """Repeated FPCellPopulation steps across several intervals should track
    the ensemble mean/covariance produced by direct OUGeneExpression
    simulation of the same dynamics (mirrors ComputeLoss's L_cons intuition)."""
    X0 = draw_cross_section(_SIM, N_CELLS)
    theta = _theta(n_intervals=N_SNAPS - 1)

    chi = preprocessing(X0)
    fp_rng = np.random.default_rng(3)
    for k in range(N_SNAPS - 1):
        chi = fp_cell_population(chi, theta, k, DT, rng=fp_rng)

    X = X0
    direct_rng = np.random.default_rng(4)
    for k in range(N_SNAPS - 1):
        X = ou_gene_expression(X, theta, k, DT, rng=direct_rng)

    fp_sample = chi.resample(N_CELLS, seed=5).T
    np.testing.assert_allclose(fp_sample.mean(axis=0), X.mean(axis=0), atol=0.1)
    np.testing.assert_allclose(np.diag(np.cov(fp_sample.T)), np.diag(np.cov(X.T)), rtol=0.5)


@pytest.mark.skip(
    reason="No separate JKO step-size (tau) exists in this particle-JKO "
           "implementation -- the proximal step size coincides with dt "
           "(single Euler-Maruyama pass per interval, see "
           "fp_cell_population.py's docstring). Substep convergence for the "
           "underlying SDE is already covered by "
           "test_ou_gene_expression.py::test_converges_to_analytic_solution_as_substeps_increase"
)
def test_smaller_tau_converges_faster_to_the_fokker_planck_solution():
    """JKO's classical convergence result: as the step size tau -> 0, the
    scheme's iterates converge to the true Fokker-Planck solution."""
    raise NotImplementedError
