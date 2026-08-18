"""
Spec for Algorithm 1's Preprocessing (paper lines 5-9):

    function Preprocessing(X_tk)
        chi_tk <- KDE(X_tk)
        return chi_tk

TDD red phase: multsc_grn_inference.preprocessing.preprocessing
currently raises NotImplementedError (see that module's docstring for the
target contract). These tests describe the behaviour it must satisfy.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from multsc_grn_inference.preprocessing import preprocessing

from _ground_truth import draw_cross_section, make_stationary_sim

N_CELLS = 2000


def _source_cloud(seed: int = 42) -> np.ndarray:
    return draw_cross_section(make_stationary_sim(seed=seed), N_CELLS)


def test_preprocessing_returns_fitted_density():
    """chi_tk must support both resampling (for W2 loss terms) and pointwise
    evaluation (for FPCellPopulation's entropy functional) -- gaussian_kde
    satisfies both."""
    X = _source_cloud()
    chi = preprocessing(X)
    assert isinstance(chi, gaussian_kde)


def test_preprocessing_resample_recovers_source_mean():
    X = _source_cloud()
    chi = preprocessing(X)
    resampled = chi.resample(N_CELLS, seed=1).T
    np.testing.assert_allclose(resampled.mean(axis=0), X.mean(axis=0), atol=0.05)


def test_preprocessing_resample_recovers_source_covariance():
    X = _source_cloud()
    chi = preprocessing(X)
    resampled = chi.resample(N_CELLS, seed=1).T
    np.testing.assert_allclose(np.cov(resampled.T), np.cov(X.T), atol=0.08)


def test_preprocessing_density_is_positive_and_finite_on_support():
    X = _source_cloud()
    chi = preprocessing(X)
    density = chi.pdf(X.T)
    assert np.all(density > 0)
    assert np.all(np.isfinite(density))


def test_preprocessing_is_deterministic_for_the_same_input():
    """Fitting is deterministic; only resampling needs a seed."""
    X = _source_cloud()
    chi_a = preprocessing(X)
    chi_b = preprocessing(X)
    np.testing.assert_allclose(chi_a.covariance, chi_b.covariance)
    np.testing.assert_allclose(
        chi_a.resample(500, seed=7).T, chi_b.resample(500, seed=7).T
    )
