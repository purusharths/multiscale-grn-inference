"""
Tests for all five loss combinations on simple OU data with multiple timesteps.

Data: same N cells are propagated across T=4 snapshots via Euler–Maruyama
      (continuous / non-destructive measurement).
Timesteps: T=4 (three intervals [t0→t1], [t1→t2], [t2→t3]).

Loss functions tested:
  1. ou_loss           – L_OU   (sum over 3 intervals)
  2. fp_loss           – L_FP   (sum over 3 intervals)
  3. consistency_loss  – L_cons (sum over 3 intervals)
  4. ou_fp_loss        – L_OU + L_FP
  5. total_loss        – L_OU + L_FP + L_cons
"""
from __future__ import annotations

import numpy as np
import pytest

from multsc_grn_inference.loss import (
    consistency_loss,
    fp_loss,
    ou_fp_loss,
    ou_loss,
    total_loss,
)

# ---------------------------------------------------------------------------
# Ground-truth parameters (2-gene system for speed)
# ---------------------------------------------------------------------------

G = 2
N_CELLS = 500
N_STEPS = 4          # number of snapshots (T = 4  →  3 intervals)
SEED = 42

TRUE_A = np.array([[1.5, 0.0], [0.0, 1.2]])
TRUE_MU = np.array([2.5, 3.0])
TRUE_SIGMA = 0.15
DT = 0.1

WRONG_A = np.array([[0.1, 0.0], [0.0, 0.1]])

N_PROJ = 100


# ---------------------------------------------------------------------------
# Data generation: simple OU (same cells tracked across all timesteps)
# ---------------------------------------------------------------------------

def _make_ou_snapshots(
    A=TRUE_A, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT,
    n_steps=N_STEPS, seed=SEED,
):
    """N_STEPS snapshots from OU simulation with the same N cells tracked."""
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mu * 0.4, 0.25 * np.eye(G), size=N_CELLS)
    X = np.maximum(X, 0.05)

    snapshots = [X.copy()]
    for _ in range(n_steps - 1):
        drift = (mu - X) @ A.T
        noise = sigma * np.sqrt(dt) * rng.standard_normal(X.shape)
        X = np.maximum(X + drift * dt + noise, 0.05)
        snapshots.append(X.copy())
    return snapshots


# ---------------------------------------------------------------------------
# 1. OU loss
# ---------------------------------------------------------------------------

def test_ou_loss_is_nonneg_finite():
    snaps = _make_ou_snapshots()
    val = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_ou_loss_accumulates_over_steps():
    """Multi-step L_OU should be larger than single-step L_OU (more intervals summed)."""
    snaps_multi = _make_ou_snapshots(n_steps=4)
    snaps_single = snaps_multi[:2]   # just [t0, t1]
    l_multi = ou_loss(snaps_multi, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_single = ou_loss(snaps_single, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_multi > l_single, (
        f"Multi-step L_OU={l_multi:.4f} should exceed single-step L_OU={l_single:.4f}"
    )


def test_ou_loss_true_params_lower_than_wrong():
    snaps = _make_ou_snapshots()
    l_true = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = ou_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong, (
        f"Expected L_OU(true)={l_true:.4f} < L_OU(wrong)={l_wrong:.4f}"
    )


def test_ou_loss_deterministic():
    snaps = _make_ou_snapshots()
    v1 = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    v2 = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    assert v1 == v2


# ---------------------------------------------------------------------------
# 2. FP loss
# ---------------------------------------------------------------------------

def test_fp_loss_is_nonneg_finite():
    snaps = _make_ou_snapshots()
    val = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_fp_loss_accumulates_over_steps():
    snaps_multi = _make_ou_snapshots(n_steps=4)
    snaps_single = snaps_multi[:2]
    l_multi = fp_loss(snaps_multi, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_single = fp_loss(snaps_single, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_multi > l_single


def test_fp_loss_true_params_lower_than_wrong():
    snaps = _make_ou_snapshots()
    l_true = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = fp_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong


def test_fp_loss_deterministic():
    snaps = _make_ou_snapshots()
    v1 = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    v2 = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    assert v1 == v2


# ---------------------------------------------------------------------------
# 3. Consistency loss
# ---------------------------------------------------------------------------

def test_consistency_loss_is_nonneg_finite():
    snaps = _make_ou_snapshots()
    val = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_consistency_loss_accumulates_over_steps():
    snaps_multi = _make_ou_snapshots(n_steps=4)
    snaps_single = snaps_multi[:2]
    l_multi = consistency_loss(snaps_multi, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_single = consistency_loss(snaps_single, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_multi > l_single


def test_consistency_loss_deterministic():
    snaps = _make_ou_snapshots()
    v1 = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    v2 = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    assert v1 == v2


# ---------------------------------------------------------------------------
# 4. OU + FP loss
# ---------------------------------------------------------------------------

def test_ou_fp_loss_is_nonneg_finite():
    snaps = _make_ou_snapshots()
    val = ou_fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_ou_fp_loss_equals_sum_of_components():
    snaps = _make_ou_snapshots()
    seed = 0
    combined = ou_fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_ou = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_fp = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    assert combined == pytest.approx(l_ou + l_fp, rel=1e-10)


def test_ou_fp_loss_true_params_lower_than_wrong():
    snaps = _make_ou_snapshots()
    l_true = ou_fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = ou_fp_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong


# ---------------------------------------------------------------------------
# 5. Total loss (OU + FP + Consistency)
# ---------------------------------------------------------------------------

def test_total_loss_is_nonneg_finite():
    snaps = _make_ou_snapshots()
    val = total_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_total_loss_equals_sum_of_components():
    snaps = _make_ou_snapshots()
    seed = 0
    combined = total_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_ou = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_fp = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_cons = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    assert combined == pytest.approx(l_ou + l_fp + l_cons, rel=1e-10)


def test_total_loss_true_params_lower_than_wrong():
    snaps = _make_ou_snapshots()
    l_true = total_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = total_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong
