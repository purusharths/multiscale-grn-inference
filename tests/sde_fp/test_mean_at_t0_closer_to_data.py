"""
"At time t0, the [KDE-resampled] mean is closer to the data described [at t0]."

The FP push-forward seeds its t0 particle cloud by resampling from
KDE(X_t0) (see `_kde_sample` in loss.py, used by `_fp_step`/`_cons_step`).
For that substitution to be meaningful, the resampled cloud's mean must sit
close to the true t0 data mean -- and, discriminatively, closer to it than
to some other, unrelated reference distribution's mean.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.loss import _kde_sample

G = 2
N_CELLS = 1000
TRUE_MU = np.array([2.5, 3.0])


def _make_data_at_t0(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(TRUE_MU * 0.4, 0.25 * np.eye(G), size=N_CELLS)
    return np.maximum(X, 0.05)


def test_kde_seeded_t0_mean_close_to_described_data_mean():
    X_t0 = _make_data_at_t0()
    resampled = _kde_sample(X_t0, N_CELLS, np.random.default_rng(1))
    np.testing.assert_allclose(resampled.mean(axis=0), X_t0.mean(axis=0), atol=0.05)


def test_kde_seeded_t0_mean_closer_to_described_data_than_to_wrong_reference():
    X_t0 = _make_data_at_t0()
    resampled = _kde_sample(X_t0, N_CELLS, np.random.default_rng(1))

    true_mean = X_t0.mean(axis=0)
    wrong_reference_mean = true_mean + np.array([2.0, 2.0])  # a clearly different population

    d_true = np.linalg.norm(resampled.mean(axis=0) - true_mean)
    d_wrong = np.linalg.norm(resampled.mean(axis=0) - wrong_reference_mean)
    assert d_true < d_wrong, (
        f"Expected KDE-resampled t0 mean to sit closer to the described data mean "
        f"({d_true:.4f}) than to an unrelated reference ({d_wrong:.4f})"
    )
