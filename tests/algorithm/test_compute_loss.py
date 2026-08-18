"""
Spec for Algorithm 1's ComputeLoss (paper lines 40-49):

    function ComputeLoss(theta, {chi_tk})
        L_OU   <- sum_k W2( KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}} )
        L_FP   <- sum_k W2( FP(chi_tk; theta), chi_{t_{k+1}} )
        L_cons <- sum_k W2( KDE(Phi_OU(X_tk; theta)), FP(chi_tk; theta) )
        return L_OU + L_FP + L_cons

TDD red phase: multsc_grn_inference.compute_loss.compute_loss
currently raises NotImplementedError unconditionally, so every test below
fails red for now -- L_FP and L_cons in particular cannot be computed for
real until FPCellPopulation (JKO, research-scope) exists. These tests
document the full target behaviour regardless, per Algorithm 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from multsc_grn_inference.compute_loss import compute_loss
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _ground_truth import make_snapshots, make_stationary_sim

N_CELLS = 500
DT = 0.3
N_SNAPS = 4

_SIM = make_stationary_sim(seed=42)
TRUE_A = _SIM.A
TRUE_MU = _SIM.mu0
TRUE_SIGMA = 0.15

# Eigenvalues 10x smaller -> dynamics barely move cells -> clearly wrong
WRONG_A = np.array([[0.1, 0.0], [0.0, 0.1]])


def _snapshots():
    return make_snapshots(_SIM, N_CELLS, N_SNAPS, DT, seed=7)


def _theta(A) -> Theta:
    return Theta(A=A, mu=[TRUE_MU] * (N_SNAPS - 1), sigma=TRUE_SIGMA)


def _chi(snaps):
    return [preprocessing(X) for X in snaps]


def test_returns_dict_with_expected_keys():
    snaps = _snapshots()
    result = compute_loss(_theta(TRUE_A), snaps, _chi(snaps), DT)
    assert set(result.keys()) >= {"L_OU", "L_FP", "L_cons", "total"}


def test_total_equals_sum_of_components():
    snaps = _snapshots()
    result = compute_loss(_theta(TRUE_A), snaps, _chi(snaps), DT)
    assert result["total"] == pytest.approx(
        result["L_OU"] + result["L_FP"] + result["L_cons"], rel=1e-10
    )


def test_all_components_nonneg_finite():
    snaps = _snapshots()
    result = compute_loss(_theta(TRUE_A), snaps, _chi(snaps), DT)
    for key in ("L_OU", "L_FP", "L_cons", "total"):
        assert result[key] >= 0.0
        assert np.isfinite(result[key])


def test_L_OU_true_params_lower_than_wrong_params():
    """L_OU = sum_k W2(KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}}) should be
    smaller under the true dynamics than under clearly wrong ones."""
    snaps = _snapshots()
    chi = _chi(snaps)
    l_true = compute_loss(_theta(TRUE_A), snaps, chi, DT, seed=0)["L_OU"]
    l_wrong = compute_loss(_theta(WRONG_A), snaps, chi, DT, seed=0)["L_OU"]
    assert l_true < l_wrong


def test_L_cons_small_when_dynamics_agree():
    """Under the true params, OUGeneExpression and FPCellPopulation are two
    approximations of the same forward dynamics from (approximately) the
    same starting distribution, so their W2 distance should be small."""
    snaps = _snapshots()
    result = compute_loss(_theta(TRUE_A), snaps, _chi(snaps), DT, seed=0)
    assert result["L_cons"] < 0.1


def test_deterministic_given_same_seed():
    snaps = _snapshots()
    chi = _chi(snaps)
    v1 = compute_loss(_theta(TRUE_A), snaps, chi, DT, seed=7)
    v2 = compute_loss(_theta(TRUE_A), snaps, chi, DT, seed=7)
    assert v1 == v2
