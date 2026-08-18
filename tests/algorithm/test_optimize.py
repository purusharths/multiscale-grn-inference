"""
Spec for Algorithm 1's Optimize (paper lines 53-57):

    function Optimize(theta0, {chi_tk})
        theta_hat <- argmin_theta ComputeLoss(theta, {chi_tk}), initialised at theta0
        return theta_hat

TDD red phase: multsc_grn_inference.optimize.optimize currently
raises NotImplementedError. Since ComputeLoss's L_FP/L_cons terms depend on
FPCellPopulation (research-scope, not implemented), Optimize cannot run for
real yet either -- these tests document the target behaviour regardless.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.optimize import optimize
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.compute_loss import compute_loss
from multsc_grn_inference.theta import Theta

from _ground_truth import make_snapshots, make_stationary_sim

N_CELLS = 500
DT = 0.3
N_SNAPS = 4

_SIM = make_stationary_sim(seed=42)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15


def _snapshots():
    return make_snapshots(_SIM, N_CELLS, N_SNAPS, DT, seed=7)


def _chi(snaps):
    return [preprocessing(X) for X in snaps]


def _wrong_start() -> Theta:
    """Deliberately off starting point: half the true diagonal rate, mu shifted."""
    return Theta(A=0.5 * TRUE_A, mu=[TRUE_MU * 0.7] * (N_SNAPS - 1), sigma=0.3)


def test_returns_theta_with_correct_shapes():
    snaps = _snapshots()
    theta_hat = optimize(_wrong_start(), snaps, _chi(snaps), DT)
    assert theta_hat.A.shape == TRUE_A.shape
    assert len(theta_hat.mu) == N_SNAPS - 1
    assert theta_hat.mu[0].shape == TRUE_MU.shape
    assert isinstance(theta_hat.sigma, float)


def test_final_loss_lower_than_initial_loss():
    snaps = _snapshots()
    chi = _chi(snaps)
    theta0 = _wrong_start()
    loss_before = compute_loss(theta0, snaps, chi, DT, seed=0)["total"]

    theta_hat = optimize(theta0, snaps, chi, DT, seed=0)
    loss_after = compute_loss(theta_hat, snaps, chi, DT, seed=0)["total"]

    assert loss_after < loss_before


def test_recovers_true_params_approximately_from_a_nearby_start():
    """Loose recovery check, starting close to the truth so a local optimiser
    has a realistic chance of converging within test-appropriate iterations."""
    snaps = _snapshots()
    chi = _chi(snaps)
    near_true_start = Theta(A=TRUE_A * 1.3, mu=[TRUE_MU * 1.1] * (N_SNAPS - 1), sigma=0.25)

    theta_hat = optimize(near_true_start, snaps, chi, DT, seed=0)

    np.testing.assert_allclose(np.diag(theta_hat.A), np.diag(TRUE_A), rtol=0.5)
    np.testing.assert_allclose(theta_hat.mu[0], TRUE_MU, atol=0.3)


def test_deterministic_given_fixed_seed():
    snaps = _snapshots()
    chi = _chi(snaps)
    theta0 = _wrong_start()
    theta_a = optimize(theta0, snaps, chi, DT, seed=3)
    theta_b = optimize(theta0, snaps, chi, DT, seed=3)
    np.testing.assert_allclose(theta_a.A, theta_b.A)
