"""
FP evolution via a JKO (Jordan-Kinderlehrer-Otto) Wasserstein-gradient-flow
step is not implemented yet -- see JKONet (https://github.com/bunnech/jkonet,
in particular https://github.com/bunnech/jkonet/pull/8) as the intended
reference implementation to adapt.

These tests are placeholders describing the behaviour a JKO-based FP solver
should satisfy once it lands, mirroring the KDE-particle FP evolution tests
in test_fp_evolution_recovery_mom.py and test_fp_kde_pushforward_integration.py.
Un-skip and fill in once `jko_step` (or equivalent) exists.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="JKO evolution not implemented yet; tracked against JKONet "
           "(https://github.com/bunnech/jkonet/pull/8)"
)


def test_jko_step_matches_analytic_ou_mean_for_gaussian_initial_density():
    """A single JKO proximal step on a Gaussian density under the OU energy
    functional should move the mean towards mu at (approximately) the same
    rate as the Euler-Maruyama SDE mean (`_analytic_mean` in
    test_euler_maruyama_sde_integration.py)."""
    raise NotImplementedError


def test_jko_evolution_matches_euler_maruyama_ensemble_moments_over_time():
    """Repeated JKO steps over several intervals should track the ensemble
    mean/covariance produced by direct Euler-Maruyama simulation of the same
    OU dynamics (same role as test_fp_pushforward_mean_close_to_direct_sde_mean
    in test_fp_kde_pushforward_integration.py, but for the JKO solver)."""
    raise NotImplementedError


def test_mom_recovers_true_params_from_jko_evolved_snapshots():
    """Same recovery check as test_fp_evolution_recovery_mom.py, but with
    snapshots produced by the JKO solver instead of KDE-resample + Euler-Maruyama."""
    raise NotImplementedError
