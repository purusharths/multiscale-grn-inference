"""
Spec for Algorithm 1's OUGeneExpression (paper lines 20-28):

    function OUGeneExpression(X_tk; theta)
        # dc(t) = A(mu_k - c(t)) dt + sigma dW_t,  t in [t_k, t_{k+1}]
        for i <= N do
            c_j(t_k) <- X_tk[j, :]
            Integrate SDE forward to t_{k+1}
        return {c_j(t_{k+1})}_{j=1}^N

TDD red phase: multsc_grn_inference.ou_gene_expression.ou_gene_expression
currently raises NotImplementedError. These tests describe the behaviour
it must satisfy, using a diagonal A (closed-form OU solution) sourced from
the datagen stationary simulator (see _ground_truth.py).
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.theta import Theta

from _ground_truth import make_stationary_sim

_SIM = make_stationary_sim(seed=42)
A_DIAG = np.diag(_SIM.A)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15  # scalar, isotropic


def _theta(mu_k: np.ndarray, n_intervals: int = 1) -> Theta:
    return Theta(A=TRUE_A, mu=[mu_k] * n_intervals, sigma=TRUE_SIGMA)


def _analytic_mean(X0: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
    """Closed-form E[c(t)] for the noise-free OU ODE with diagonal A."""
    return mu + (X0 - mu) * np.exp(-A_DIAG * t)


def test_output_shape_matches_input():
    X_tk = np.full((37, 2), 1.0)
    result = ou_gene_expression(X_tk, _theta(TRUE_MU), k=0, dt=0.1, rng=np.random.default_rng(0))
    assert result.shape == X_tk.shape


def test_converges_to_analytic_solution_as_substeps_increase():
    """Checking Euler-Maruyama: finer substeps should track the closed-form
    noise-free OU trajectory more closely."""
    X0 = np.array([[0.5, 0.5]])
    t = 0.5
    theta = Theta(A=TRUE_A, mu=[TRUE_MU], sigma=0.0)
    analytic = _analytic_mean(X0[0], TRUE_MU, t)

    errors = []
    for n_substeps in (1, 5, 50, 500):
        result = ou_gene_expression(
            X0, theta, k=0, dt=t, n_substeps=n_substeps, rng=np.random.default_rng(0)
        )
        errors.append(np.linalg.norm(result[0] - analytic))

    assert errors == sorted(errors, reverse=True), (
        f"Expected monotonically decreasing error as substeps increase, got {errors}"
    )
    assert errors[-1] < 2e-3
    assert errors[0] > 1e-2


def test_ensemble_mean_tracks_analytic_mean_under_noise():
    n_cells = 5000
    X0 = np.tile(np.array([0.5, 0.5]), (n_cells, 1))
    dt = 0.5
    theta = _theta(TRUE_MU)
    analytic = _analytic_mean(X0[0], TRUE_MU, dt)

    result = ou_gene_expression(X0, theta, k=0, dt=dt, n_substeps=10, rng=np.random.default_rng(42))
    np.testing.assert_allclose(result.mean(axis=0), analytic, atol=0.05)


def test_output_is_nonnegative():
    X0 = np.zeros((200, 2))
    theta = Theta(A=TRUE_A, mu=[TRUE_MU], sigma=5.0)
    result = ou_gene_expression(X0, theta, k=0, dt=0.1, n_substeps=1, rng=np.random.default_rng(0))
    assert np.all(result >= 0.0)


def test_uses_mu_at_k_not_a_fixed_interval():
    """theta.mu is per-interval {mu_k} (paper Data, line 2) -- OUGeneExpression
    must select mu_at(k) for the interval being integrated, not e.g. always mu[0]."""
    n_cells = 3000
    X0 = np.tile(np.array([0.0, 0.0]), (n_cells, 1))
    dt = 3.0  # several relaxation times, noise-free so mean == deterministic target
    mu_0 = np.array([1.0, 1.0])
    mu_1 = np.array([5.0, 5.0])
    theta = Theta(A=TRUE_A, mu=[mu_0, mu_1], sigma=0.0)

    result_k0 = ou_gene_expression(X0, theta, k=0, dt=dt, n_substeps=200, rng=np.random.default_rng(0))
    result_k1 = ou_gene_expression(X0, theta, k=1, dt=dt, n_substeps=200, rng=np.random.default_rng(0))

    np.testing.assert_allclose(result_k0.mean(axis=0), mu_0, atol=0.05)
    np.testing.assert_allclose(result_k1.mean(axis=0), mu_1, atol=0.05)


def test_input_array_is_not_mutated():
    X0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X0_copy = X0.copy()
    ou_gene_expression(X0, _theta(TRUE_MU), k=0, dt=0.1, n_substeps=3, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(X0, X0_copy)
