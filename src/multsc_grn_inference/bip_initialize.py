"""
Algorithm 1, function BIPInitilize (paper lines 11-18):

    function BIPInitilize(D)
        # Used as preconditioner for theta.
        # Warm-starts theta via Bayesian inverse problem on linearised OU model
        # Specify prior: pi(theta)
        # Specify likelihood: pi(D|theta) (gaussian under linearized OU dynamics)
        # Compute Posterior: pi(theta | D) prop_to p(D | theta) * pi(theta)
        # theta0 <- MAP(pi(theta | D)) or posterior mean
        return theta0

Research-scope: a proper Bayesian-inverse-problem warm-start (explicit
prior + Gaussian likelihood under the *linearised* OU model + MAP or
posterior-mean extraction) isn't implemented in this codebase yet -- the
closest existing idea (method-of-moments init, archived) solves a
different, non-Bayesian problem. See
tests/algorithm/test_bip_initialize.py for the skip-marked specs this
should satisfy once built.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.theta import Theta


def bip_initialize(
    D: list[np.ndarray],
    mu: list[np.ndarray],
    dt: float,
) -> Theta:
    """theta0 <- BIP warm-start(D; mu, dt). See module docstring."""
    raise NotImplementedError
