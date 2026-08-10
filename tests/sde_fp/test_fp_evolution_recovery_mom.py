"""
FP evolution over many timesteps (repeated KDE-resample + Euler-Maruyama,
see loss.py's `_fp_step`), checked via method-of-moments (MoM) recovery.

MoM (`mom_estimate` in mom_init.py) only looks at cross-sectional moments of
a snapshot sequence -- it has no notion of *how* those snapshots were
produced. So generating the sequence via the FP mechanism (KDE-resample at
each step rather than tracking the same particles) and recovering (A, mu,
sigma) with MoM is a check that FP evolution preserves the moment structure
Euler-Maruyama would have produced directly.

Ground truth (A, mu) and the starting cross-section come from the datagen
stationary simulator (_stationary_destructive_data.py). The starting
cross-section uses `draw_perturbed_cross_section` (not a plain t0 draw):
mom_estimate's mean-velocity regression (step 2) needs the population mean
to actually move over the snapshot sequence, but the simulator's own t0
state already sits at mu, which gives a degenerate (near-zero-displacement)
regression. Starting away from mu and letting the FP evolution relax
towards mu over the sequence is what the destructive-measurement /
gene-perturbation datasets in datagen/ model too.

The sequence is run long enough (K * dt >> 1 / diag(A)) for the last
snapshot to sit near the stationary mean, since MoM uses it as a mu proxy.
"""
from __future__ import annotations

import numpy as np
import pytest

from multsc_grn_inference.loss import _kde_sample, _ou_euler_maruyama
from multsc_grn_inference.mom_init import mom_estimate

from _stationary_destructive_data import draw_perturbed_cross_section, make_stationary_sim

N_CELLS = 2000
DT = 0.3
N_SNAPSHOTS = 20  # total horizon = 5.7, several relaxation times

_SIM = make_stationary_sim(seed=42)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15  # scalar, isotropic -- distinct noise model from the sim's per-gene D


def _simulate_fp_evolution(seed: int = 42) -> list[np.ndarray]:
    X0 = draw_perturbed_cross_section(make_stationary_sim(seed=seed), N_CELLS, shift=-0.6 * TRUE_MU)

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
    # diagonal entries of A should be positive and close to the true diagonal;
    # off-diagonal entries should remain small
    # because the true system has no gene-gene coupling.

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
