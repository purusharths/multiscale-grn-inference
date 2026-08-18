"""
Spec for Algorithm 1's BIPInitilize (paper lines 11-18):

    function BIPInitilize(D)
        # Warm-starts theta via Bayesian inverse problem on linearised OU model
        # prior pi(theta); gaussian likelihood pi(D|theta) under linearised OU
        # theta0 <- MAP(pi(theta | D)) or posterior mean
        return theta0

Research-scope, not implemented (see bip_initialize.py's docstring): a
proper Bayesian-inverse-problem warm-start needs an explicit prior, a
Gaussian likelihood under the *linearised* OU model, and MAP/posterior-mean
extraction. Skip-marked so tests/algorithm/ collects cleanly; un-skip and
fill in the numerical checks once bip_initialize() is implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.skip(
    reason="BIPInitialize (Bayesian warm-start on linearised OU model) is "
           "research-scope, not implemented yet"
)


def test_theta0_shapes_match_data():
    """theta0.A is (G,G), theta0.mu has one vector per interval, theta0.sigma is scalar."""
    raise NotImplementedError


def test_theta0_closer_to_true_params_than_a_naive_generic_start():
    """A Bayesian warm start informed by D should land closer to the true
    (A, mu, sigma) than an uninformed generic guess (e.g. identity A,
    data mean mu, sigma=1) -- otherwise it isn't earning its place as a
    preconditioner for Optimize."""
    raise NotImplementedError


def test_theta0_matches_closed_form_map_for_a_linear_gaussian_special_case():
    """For a model that IS linear-Gaussian (constant A, mu, sigma; Euler-Maruyama
    likelihood is exactly Gaussian in theta given fixed X_t), the MAP/posterior
    mean has a closed form (Bayesian linear regression normal equations). Compare
    bip_initialize()'s output against that closed form directly -- this is the
    one case where "correct" isn't ambiguous."""
    raise NotImplementedError


def test_theta0_is_deterministic_given_fixed_seed():
    raise NotImplementedError
