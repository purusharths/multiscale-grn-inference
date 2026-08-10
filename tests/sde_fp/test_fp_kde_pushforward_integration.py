"""
Tests for the combined FP push-forward: KDE-resample then Euler-Maruyama
integrate, i.e. the same mechanism as `_fp_step` in loss.py, exercised
directly (without going through the sliced-W2 loss wrapper).

FP(chi_tk; theta) = Phi_OU(sample(KDE(X_tk)); theta)

This checks the push-forward is well-formed on its own (shape, non-negative)
and that, starting from the same cross-sectional snapshot, it stays close in
distribution to directly integrating the SDE on the original particles
(the two paths coincide in expectation; KDE resampling is what makes the
FP path a valid stand-in when no per-cell trajectory is available).

The source snapshot X_t0 is a destructive-measurement-style cross-section
drawn from the datagen stationary simulator (_stationary_destructive_data.py),
and ground truth (A, mu) comes from that same simulator.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _kde_sample, _ou_euler_maruyama

from _stationary_destructive_data import (
    draw_destructive_t0_cross_section,
    draw_perturbed_cross_section,
    make_stationary_sim,
)

N_CELLS = 2000
DT = 0.1

_SIM = make_stationary_sim(seed=42)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15  # scalar, isotropic -- distinct noise model from the sim's per-gene D


def _make_source_cloud(seed: int = 42) -> np.ndarray:
    return draw_destructive_t0_cross_section(make_stationary_sim(seed=seed), N_CELLS)


def _fp_pushforward(X_tk: np.ndarray, rng: np.random.Generator, n_substeps: int = 5) -> np.ndarray:
    X_kde = _kde_sample(X_tk, len(X_tk), rng)
    return _ou_euler_maruyama(X_kde, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_substeps, rng)


def test_fp_pushforward_output_shape_and_nonnegative():
    X_tk = _make_source_cloud()
    result = _fp_pushforward(X_tk, np.random.default_rng(1))
    assert result.shape == X_tk.shape
    assert np.all(result >= 0.0)


def test_fp_pushforward_mean_close_to_direct_sde_mean():
    """Same source snapshot, two propagation paths: FP (KDE-resample then integrate) vs
    direct SDE (integrate the original particles). Their ensemble means should agree
    up to KDE resampling noise, since both apply identical OU dynamics."""
    X_tk = _make_source_cloud()

    fp_result = _fp_pushforward(X_tk, np.random.default_rng(5))
    sde_result = _ou_euler_maruyama(
        X_tk, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_substeps=5, rng=np.random.default_rng(5)
    )

    np.testing.assert_allclose(fp_result.mean(axis=0), sde_result.mean(axis=0), atol=0.05)


def test_fp_pushforward_true_params_closer_to_next_snapshot_than_wrong_params():
    """Sanity check mirroring fp_loss: FP push-forward under the true dynamics should land
    closer (in mean) to the actual next-timestep data than under clearly wrong dynamics.

    Uses a perturbed cross-section (see draw_perturbed_cross_section) rather than a plain
    t0 draw: the simulator's own t0 state sits at mu already, where true vs. wrong dynamics
    produce near-identical (near-zero) drift and can't be told apart."""
    X_t0 = draw_perturbed_cross_section(make_stationary_sim(seed=42), N_CELLS, shift=-0.6 * TRUE_MU)
    drift = (TRUE_MU - X_t0) @ TRUE_A.T
    noise = TRUE_SIGMA * np.sqrt(DT) * np.random.default_rng(9).standard_normal(X_t0.shape)
    X_t1 = np.maximum(X_t0 + drift * DT + noise, 0.05)

    def pushforward(A):
        rng = np.random.default_rng(3)
        X_kde = _kde_sample(X_t0, len(X_t0), rng)
        return _ou_euler_maruyama(X_kde, A, TRUE_MU, TRUE_SIGMA, DT, n_substeps=5, rng=rng)

    wrong_A = np.array([[0.1, 0.0], [0.0, 0.1]])
    true_result = pushforward(TRUE_A)
    wrong_result = pushforward(wrong_A)

    d_true = np.linalg.norm(true_result.mean(axis=0) - X_t1.mean(axis=0))
    d_wrong = np.linalg.norm(wrong_result.mean(axis=0) - X_t1.mean(axis=0))
    assert d_true < d_wrong
