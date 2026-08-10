"""
Tests for the SDE integrator itself: `_ou_euler_maruyama` in loss.py.

dc = A(mu - c) dt + sigma dW, integrated with Euler-Maruyama substeps.

Ground truth (A, mu) comes from the datagen "stationary" simulator (see
_stationary_destructive_data.py) with network_density=0.0, which forces a
diagonal A. For diagonal A, the noise-free (sigma=0) case has a closed-form
solution
    X(t) = mu + (X0 - mu) * exp(-diag(A) * t)
which lets us check the integrator converges to the right answer as the
number of substeps increases (Euler-Maruyama's defining property), and that
the noisy ensemble mean tracks the same analytic curve.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _ou_euler_maruyama

from _stationary_destructive_data import make_stationary_sim

# ---------------------------------------------------------------------------
# Ground-truth parameters, sourced from the datagen stationary simulator
# (diagonal A => closed-form ODE solution)
# ---------------------------------------------------------------------------

_SIM = make_stationary_sim(seed=42)
A_DIAG = np.diag(_SIM.A)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15  # scalar, isotropic -- distinct noise model from the sim's per-gene D


def _analytic_mean(X0: np.ndarray, t: float) -> np.ndarray:
    """Closed-form E[X(t)] for the noise-free OU ODE with diagonal A."""
    return TRUE_MU + (X0 - TRUE_MU) * np.exp(-A_DIAG * t)  # exact soln


# ---------------------------------------------------------------------------
# Convergence of Euler-Maruyama as substeps increase (noise-free case)
# ---------------------------------------------------------------------------

def test_euler_maruyama_converges_to_analytic_solution_as_substeps_increase():
    # As the timestep becomes smaller, the numerical solution should converge toward the true solution.
    X0 = np.array([[0.5, 0.5]])
    t = 0.5
    analytic = _analytic_mean(X0[0], t)

    errors = []
    for n_substeps in (1, 5, 50, 500):
        result = _ou_euler_maruyama(
            X0, TRUE_A, TRUE_MU, sigma=0.0, dt=t,
            n_substeps=n_substeps, rng=np.random.default_rng(0),
        )
        errors.append(np.linalg.norm(result[0] - analytic))

    assert errors == sorted(errors, reverse=True), (
        f"Expected monotonically decreasing error as substeps increase, got {errors}"
    )
    assert errors[-1] < 2e-3, f"Fine-grained integration should nearly match analytic solution, got err={errors[-1]}"
    assert errors[0] > 1e-2, f"Coarse (1-substep) integration should show visible discretisation error, got err={errors[0]}"


def test_euler_maruyama_matches_single_deterministic_step_exactly():
    """With n_substeps=1 and sigma=0, the update is exactly X0 + dt*A(mu-X0) by construction."""
    X0 = np.array([[1.0, 4.0]])
    dt = 0.1
    expected = X0 + dt * ((TRUE_MU - X0) @ TRUE_A.T)
    result = _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, sigma=0.0, dt=dt,
        n_substeps=1, rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(result, np.maximum(expected, 0.0))


# ---------------------------------------------------------------------------
# Noisy ensemble behaviour
# ---------------------------------------------------------------------------

def test_ensemble_mean_tracks_analytic_ou_mean_under_noise():
    # Across n noisy trajectories, the empirical mean should approximately follow the analytical OU mean.
    n_cells = 5000
    X0 = np.tile(np.array([0.5, 0.5]), (n_cells, 1))
    dt = 0.5
    analytic = _analytic_mean(X0[0], dt)

    result = _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, TRUE_SIGMA, dt,
        n_substeps=10, rng=np.random.default_rng(42),
    )
    # Standard error of the ensemble mean is small for n_cells=5000
    np.testing.assert_allclose(result.mean(axis=0), analytic, atol=0.05)


def test_diffusion_increases_ensemble_variance_relative_to_noise_free():
    # Adding diffusion (sigma > 0) should produce greater ensemble variance than the deterministic case.
    n_cells = 5000
    X0 = np.tile(np.array([0.5, 0.5]), (n_cells, 1))
    dt = 0.5

    noiseless = _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, sigma=0.0, dt=dt,
        n_substeps=10, rng=np.random.default_rng(1),
    )
    noisy = _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, TRUE_SIGMA, dt,
        n_substeps=10, rng=np.random.default_rng(1),
    )
    assert np.all(noisy.var(axis=0) > noiseless.var(axis=0))


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------

def test_output_shape_matches_input():
    # The output shape should match the input shape, regardless of the number of substeps or the random seed.
    X0 = np.full((37, 2), 1.0)
    result = _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, TRUE_SIGMA, dt=0.1,
        n_substeps=3, rng=np.random.default_rng(0),
    )
    assert result.shape == X0.shape


def test_output_is_clipped_nonnegative():
    # With large diffusion that would generate negative values, the output must remain X>=0.
    X0 = np.zeros((200, 2))
    result = _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, sigma=5.0, dt=0.1,
        n_substeps=1, rng=np.random.default_rng(0),
    )
    assert np.all(result >= 0.0)


def test_input_array_is_not_mutated():
    # The function must not modify the original X0 array.
    X0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X0_copy = X0.copy()
    _ou_euler_maruyama(
        X0, TRUE_A, TRUE_MU, TRUE_SIGMA, dt=0.1,
        n_substeps=3, rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(X0, X0_copy)
