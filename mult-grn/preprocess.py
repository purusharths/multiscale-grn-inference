"""
KDE pre-processing: compute empirical population densities chi_hat_k for each
timepoint from the raw snapshot matrices.

scipy.stats.gaussian_kde is used with Scott's bandwidth rule by default.
The returned KDE objects support:
    .resample(n)  → (G, n) float64 array
    .logpdf(x)    → log-density at x  (x shape: (G, n_pts))
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def compute_chi_hat(
    dataset: dict[int, np.ndarray],
    bw_method: str | float | None = "scott",
) -> dict[int, gaussian_kde]:
    """
    Parameters
    ----------
    dataset    : output of load_dataset — maps k → X_{t_k} (N, G)
    bw_method  : bandwidth selector passed to scipy.stats.gaussian_kde

    Returns
    -------
    dict[int, gaussian_kde]
        Maps timepoint index k → fitted KDE over R^G.
        KDE data matrix has shape (G, N) as required by scipy.
    """
    chi_hat: dict[int, gaussian_kde] = {}
    for k, X in dataset.items():
        # scipy expects (G, N): transpose
        chi_hat[k] = gaussian_kde(X.T, bw_method=bw_method)
    return chi_hat


def sample_chi_hat(
    kde: gaussian_kde,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw n particles from the KDE and return them as (n, G) float64 array.
    """
    # gaussian_kde.resample returns (G, n); transpose to (n, G)
    samples = kde.resample(n, seed=int(rng.integers(0, 2**31))).T
    return samples.astype(np.float64)
