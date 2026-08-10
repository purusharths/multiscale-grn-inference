"""
Tests for the Gaussian-KDE resampling primitive: `_kde_sample` in loss.py.

`_kde_sample` fits a Gaussian KDE to a particle cloud and draws new samples
from it. This underlies the FP push-forward (chi_tk -> KDE -> resample ->
Euler-Maruyama), so it needs to reproduce the source cloud's distribution
(mean, covariance, requested sample count) closely enough for that
downstream use to make sense.

The source cloud is a destructive-measurement-style t0 snapshot drawn from
the datagen stationary simulator (see _stationary_destructive_data.py),
rather than a hand-rolled synthetic distribution.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _kde_sample

from _stationary_destructive_data import draw_destructive_t0_cross_section, make_stationary_sim

N_CELLS = 2000


def _make_source_cloud(seed: int = 42) -> np.ndarray:
    sim = make_stationary_sim(seed=seed)
    return draw_destructive_t0_cross_section(sim, N_CELLS)


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
        assert resampled.shape == (n, X.shape[1])


def test_kde_resample_is_clipped_nonnegative():
    # Source cloud hugging zero => some KDE draws would go negative without clipping
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(0.02, 0.05, size=(N_CELLS, 2)))
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
