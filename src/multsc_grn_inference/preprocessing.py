"""
Algorithm 1, function Preprocessing (paper lines 5-9):

    function Preprocessing(X_tk)
        # Estimate population density at each time-point.
        chi_tk <- KDE(X_tk)
        return chi_tk

Fits a Gaussian KDE to one cross-sectional snapshot X_tk (N, G), giving the
estimated population density chi_tk. Downstream consumers need two things
from it: particle-cloud samples (for the sliced-W2 terms in ComputeLoss)
and pointwise (log-)density evaluation (for FPCellPopulation's entropy
functional) -- scipy.stats.gaussian_kde supports both, so that's the
return type here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def preprocessing(X_tk: np.ndarray) -> gaussian_kde:
    """chi_tk <- KDE(X_tk). X_tk: (N, G) cross-sectional snapshot."""
    return gaussian_kde(X_tk.T)
