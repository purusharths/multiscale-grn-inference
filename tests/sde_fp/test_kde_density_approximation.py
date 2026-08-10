"""
Tests for the Gaussian-KDE resampling primitive: `_kde_sample` in loss.py.

`_kde_sample` fits a Gaussian KDE to a particle cloud and draws new samples
from it. This underlies the FP push-forward (chi_tk -> KDE -> resample ->
Euler-Maruyama), so it needs to reproduce the source cloud's distribution
(mean, covariance, requested sample count) closely enough for that
downstream use to make sense.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _kde_sample

G = 2
N_CELLS = 2000
TRUE_MU = np.array([2.5, 3.0])


def _make_source_cloud(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(TRUE_MU * 0.4, 0.25 * np.eye(G), size=N_CELLS)
    return np.maximum(X, 0.05)


def test_kde_resample_recovers_source_mean():
    X = _make_source_cloud()
    resampled = _kde_sample(X, N_CELLS, np.random.default_rng(1))
    np.testing.assert_allclose(resampled.mean(axis=0), X.mean(axis=0), atol=0.05)


def test_kde_resample_recovers_source_covariance():
    X = _make_source_cloud()
    resampled = _kde_sample(X, N_CELLS, np.random.default_rng(1))
    # KDE bandwidth smoothing slightly inflates covariance; allow generous tolerance
    np.testing.assert_allclose(np.cov(resampled.T), np.cov(X.T), atol=0.08)


def test_kde_resample_returns_requested_sample_count():
    X = _make_source_cloud()
    for n in (1, 50, 3000):
        resampled = _kde_sample(X, n, np.random.default_rng(1))
        assert resampled.shape == (n, G)


def test_kde_resample_is_clipped_nonnegative():
    # Source cloud hugging zero => some KDE draws would go negative without clipping
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(0.02, 0.05, size=(N_CELLS, G)))
    resampled = _kde_sample(X, N_CELLS, np.random.default_rng(1))
    assert np.all(resampled >= 0.0)


def test_kde_resample_deterministic_given_same_seed():
    X = _make_source_cloud()
    r1 = _kde_sample(X, N_CELLS, np.random.default_rng(7))
    r2 = _kde_sample(X, N_CELLS, np.random.default_rng(7))
    np.testing.assert_array_equal(r1, r2)


def test_kde_resample_varies_with_different_seed():
    X = _make_source_cloud()
    r1 = _kde_sample(X, N_CELLS, np.random.default_rng(7))
    r2 = _kde_sample(X, N_CELLS, np.random.default_rng(8))
    assert not np.array_equal(r1, r2)
