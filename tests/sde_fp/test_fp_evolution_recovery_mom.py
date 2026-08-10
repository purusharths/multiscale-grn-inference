"""
FP evolution over many timesteps (repeated KDE-resample + Euler-Maruyama,
see loss.py's `_fp_step`), checked via method-of-moments (MoM) recovery.

MoM (`mom_estimate` in mom_init.py) only looks at cross-sectional moments of
a snapshot sequence -- it has no notion of *how* those snapshots were
produced. So generating the sequence via the FP mechanism (KDE-resample at
each step rather than tracking the same particles) and recovering (A, mu,
sigma) with MoM is a check that FP evolution preserves the moment structure
Euler-Maruyama would have produced directly.

The sequence is run long enough (K * dt >> 1 / diag(A)) for the last
snapshot to sit near the stationary mean, since MoM uses it as a mu proxy.
"""
from __future__ import annotations

import numpy as np
import pytest

from multsc_grn_inference.loss import _kde_sample, _ou_euler_maruyama
from multsc_grn_inference.mom_init import mom_estimate

G = 2
N_CELLS = 2000
TRUE_A = np.array([[1.5, 0.0], [0.0, 1.2]])
TRUE_MU = np.array([2.5, 3.0])
TRUE_SIGMA = 0.15
DT = 0.3
N_SNAPSHOTS = 20  # total horizon = 5.7, several relaxation times


def _simulate_fp_evolution(seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal(TRUE_MU * 0.4, 0.25 * np.eye(G), size=N_CELLS)
    X0 = np.maximum(X0, 0.05)

    fp_rng = np.random.default_rng(seed + 100)
    snapshots = [X0]
    X = X0
    for _ in range(N_SNAPSHOTS - 1):
        X_kde = _kde_sample(X, len(X), fp_rng)
        X = _ou_euler_maruyama(X_kde, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_substeps=5, rng=fp_rng)
        snapshots.append(X)
    return snapshots


def test_mom_recovers_mu_from_fp_evolved_snapshots():
    # recovered equilirium mean. 
    snapshots = _simulate_fp_evolution()
    _, mu_hat, _ = mom_estimate(snapshots, DT)
    np.testing.assert_allclose(mu_hat, TRUE_MU, atol=0.05)


def test_mom_recovers_sigma_from_fp_evolved_snapshots():
    # recovery of diffusion strength.the estimated sigma should be close to the true 
    # sigma (0.15, with 30%  error tolerance)
    snapshots = _simulate_fp_evolution()
    _, _, sigma_hat = mom_estimate(snapshots, DT)
    assert sigma_hat == pytest.approx(TRUE_SIGMA, rel=0.3)


def test_mom_recovers_A_diagonal_structure_from_fp_evolved_snapshots():
    # diagonal entries of A should be positive and close to 1.5,1.2; 
    # off-diagonal entries should remain small 
    # because the true system has no gene–gene coupling.
    
    snapshots = _simulate_fp_evolution()
    A_hat, _, _ = mom_estimate(snapshots, DT)

    # Diagonal entries: same order of magnitude and sign as ground truth
    diag_hat = np.diag(A_hat)
    assert np.all(diag_hat > 0)
    np.testing.assert_allclose(diag_hat, np.diag(TRUE_A), rtol=0.5)

    # Off-diagonal entries should stay small relative to the diagonal, since
    # the true system has no cross-gene coupling
    off_diag_magnitude = max(abs(A_hat[0, 1]), abs(A_hat[1, 0]))
    assert off_diag_magnitude < 0.5 * diag_hat.mean()
