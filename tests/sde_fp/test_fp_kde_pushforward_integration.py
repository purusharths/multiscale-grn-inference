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
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _kde_sample, _ou_euler_maruyama

G = 2
N_CELLS = 2000
TRUE_A = np.array([[1.5, 0.0], [0.0, 1.2]])
TRUE_MU = np.array([2.5, 3.0])
TRUE_SIGMA = 0.15
DT = 0.1


def _make_source_cloud(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(TRUE_MU * 0.4, 0.25 * np.eye(G), size=N_CELLS)
    return np.maximum(X, 0.05)


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
    closer (in mean) to the actual next-timestep data than under clearly wrong dynamics."""
    X_t0 = _make_source_cloud()
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
