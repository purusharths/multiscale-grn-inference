"""
Evolution of a single OU particle over multiple time steps under
Euler-Maruyama, checked against the closed-form OU solution (diagonal A):

    noise-free ODE : X(t) = mu + (X0 - mu) * exp(-diag(A) * t)
    long-time SDE  : stationary variance = sigma^2 / (2 * diag(A))

A lone particle's noisy trajectory can't be checked against a distributional
target by itself, so the long-time / stationary-variance checks use an
ensemble of independent single-particle replicates (each replicate is one
particle, evolved on its own -- there is no cross-particle interaction in
the OU SDE, so this is equivalent to N independent single-particle runs).

Ground truth (A, mu) is sourced from the datagen stationary simulator (see
_stationary_destructive_data.py); network_density=0.0 there forces a
diagonal A, which the closed-form checks below need.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _ou_euler_maruyama

from _stationary_destructive_data import make_stationary_sim

_SIM = make_stationary_sim(seed=42)
A_DIAG = np.diag(_SIM.A)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15  # scalar, isotropic -- distinct noise model from the sim's per-gene D


def test_single_particle_noise_free_trajectory_matches_analytic_decay():
    """Step a single particle forward several intervals (sigma=0) and compare
    its position at each step to the analytic exponential decay towards mu."""
    X = np.array([[0.5, 0.5]])
    dt = 0.2
    n_steps = 10
    rng = np.random.default_rng(0)

    for step in range(1, n_steps + 1):
        X = _ou_euler_maruyama(X, TRUE_A, TRUE_MU, sigma=0.0, dt=dt, n_substeps=20, rng=rng)
        t = step * dt
        analytic = TRUE_MU + (np.array([0.5, 0.5]) - TRUE_MU) * np.exp(-A_DIAG * t)
        np.testing.assert_allclose(X[0], analytic, atol=0.01)


def test_single_particle_noise_free_trajectory_converges_to_mu():
    X = np.array([[0.0, 0.0]])
    dt = 5.0  # several time constants (1/A_DIAG ~ 0.7-0.8)
    X = _ou_euler_maruyama(X, TRUE_A, TRUE_MU, sigma=0.0, dt=dt, n_substeps=200, rng=np.random.default_rng(0))
    np.testing.assert_allclose(X[0], TRUE_MU, atol=0.01)


def test_ensemble_of_independent_single_particles_approaches_stationary_variance():
    """N independent single-particle trajectories, run long enough, should have
    ensemble variance approaching the OU stationary variance sigma^2/(2A)."""
    n_particles = 20_000
    X = np.tile(np.array([0.5, 0.5]), (n_particles, 1))
    rng = np.random.default_rng(1)

    dt_small = 0.05
    n_steps = 200  # total time = 10, several stationary relaxation times
    for _ in range(n_steps):
        X = _ou_euler_maruyama(X, TRUE_A, TRUE_MU, TRUE_SIGMA, dt_small, n_substeps=1, rng=rng)

    stationary_var = TRUE_SIGMA**2 / (2 * A_DIAG)
    np.testing.assert_allclose(X.var(axis=0), stationary_var, rtol=0.2)


def test_finer_substeps_reduce_euler_maruyama_bias_for_single_particle():
    """Checking Euler-Maruyama: coarser substep counts should show more bias
    against the analytic (noise-free) trajectory than finer ones, at fixed dt."""
    X0 = np.array([[0.0, 0.0]])
    dt = 1.0
    analytic = TRUE_MU + (X0[0] - TRUE_MU) * np.exp(-A_DIAG * dt)

    err_coarse = np.linalg.norm(
        _ou_euler_maruyama(X0, TRUE_A, TRUE_MU, sigma=0.0, dt=dt, n_substeps=1, rng=np.random.default_rng(0))[0]
        - analytic
    )
    err_fine = np.linalg.norm(
        _ou_euler_maruyama(X0, TRUE_A, TRUE_MU, sigma=0.0, dt=dt, n_substeps=100, rng=np.random.default_rng(0))[0]
        - analytic
    )
    assert err_fine < err_coarse
