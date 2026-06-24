"""
Tests for all five loss combinations on destructive-measurement data, single timestep.

Data: at each of T=2 timepoints a fresh, independent set of N cells is drawn
      from the population (cells destroyed after measurement — no trajectories).
      Cells at t0 and t1 are different individuals from the same OU dynamics.
Timesteps: T=2 (one interval [t0, t1]).

Loss functions tested:
  1. ou_loss           – L_OU
  2. fp_loss           – L_FP
  3. consistency_loss  – L_cons
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
SEED = 42

TRUE_A = np.array([[1.5, 0.0], [0.0, 1.2]])
TRUE_MU = np.array([2.5, 3.0])
TRUE_SIGMA = 0.15
DT = 0.1

WRONG_A = np.array([[0.1, 0.0], [0.0, 0.1]])

N_PROJ = 100


# ---------------------------------------------------------------------------
# Data generation: destructive measurements
#
# Simulate M >> N cells from t0 to t1. At each timepoint, select a disjoint
# set of N cells (no cell is used at more than one timepoint).
# ---------------------------------------------------------------------------

def _make_destructive_snapshots(
    A=TRUE_A, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT,
    n_steps=2, seed=SEED,
):
    """
    T=n_steps independent cross-sectional snapshots (destructive measurements).

    Each cell is observed at exactly one timepoint: cells at t0 are different
    from cells at t1, mirroring a scRNA-seq destructive protocol.
    """
    rng = np.random.default_rng(seed)
    M = N_CELLS * n_steps * 2   # large pool; each cell observed at most once

    # Initialise pool far from mu so dynamics are clearly visible
    X_pool = rng.multivariate_normal(mu * 0.4, 0.25 * np.eye(G), size=M)
    X_pool = np.maximum(X_pool, 0.05)

    # Evolve the pool forward: store state at each timestep
    pool_states = [X_pool.copy()]
    X = X_pool.copy()
    for _ in range(n_steps - 1):
        drift = (mu - X) @ A.T
        noise = sigma * np.sqrt(dt) * rng.standard_normal(X.shape)
        X = np.maximum(X + drift * dt + noise, 0.05)
        pool_states.append(X.copy())

    # Destructive sampling: each cell used at exactly one timepoint
    all_indices = rng.permutation(M)
    snapshots = []
    for k in range(n_steps):
        idx = all_indices[k * N_CELLS : (k + 1) * N_CELLS]
        snapshots.append(pool_states[k][idx])
    return snapshots


# ---------------------------------------------------------------------------
# 1. OU loss
# ---------------------------------------------------------------------------

def test_ou_loss_is_nonneg_finite():
    snaps = _make_destructive_snapshots()
    val = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_ou_loss_true_params_lower_than_wrong():
    snaps = _make_destructive_snapshots()
    l_true = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = ou_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong, (
        f"Expected L_OU(true)={l_true:.4f} < L_OU(wrong)={l_wrong:.4f}"
    )


def test_ou_loss_deterministic():
    snaps = _make_destructive_snapshots()
    v1 = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    v2 = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    assert v1 == v2


# ---------------------------------------------------------------------------
# 2. FP loss
# ---------------------------------------------------------------------------

def test_fp_loss_is_nonneg_finite():
    snaps = _make_destructive_snapshots()
    val = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_fp_loss_true_params_lower_than_wrong():
    snaps = _make_destructive_snapshots()
    l_true = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = fp_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong, (
        f"Expected L_FP(true)={l_true:.4f} < L_FP(wrong)={l_wrong:.4f}"
    )


def test_fp_loss_deterministic():
    snaps = _make_destructive_snapshots()
    v1 = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    v2 = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    assert v1 == v2


# ---------------------------------------------------------------------------
# 3. Consistency loss
# ---------------------------------------------------------------------------

def test_consistency_loss_is_nonneg_finite():
    snaps = _make_destructive_snapshots()
    val = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_consistency_loss_deterministic():
    snaps = _make_destructive_snapshots()
    v1 = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    v2 = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=7)
    assert v1 == v2


def test_consistency_loss_true_params_small():
    """
    At true params, OU and FP use identical dynamics so the only discrepancy
    comes from KDE approximation of the starting distribution; with N=500 this
    should be small.
    """
    snaps = _make_destructive_snapshots()
    val = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert val < 0.1, f"Consistency loss unexpectedly large: {val:.4f}"


# ---------------------------------------------------------------------------
# 4. OU + FP loss
# ---------------------------------------------------------------------------

def test_ou_fp_loss_is_nonneg_finite():
    snaps = _make_destructive_snapshots()
    val = ou_fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_ou_fp_loss_equals_sum_of_components():
    snaps = _make_destructive_snapshots()
    seed = 0
    combined = ou_fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_ou = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_fp = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    assert combined == pytest.approx(l_ou + l_fp, rel=1e-10)


def test_ou_fp_loss_true_params_lower_than_wrong():
    snaps = _make_destructive_snapshots()
    l_true = ou_fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = ou_fp_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong


# ---------------------------------------------------------------------------
# 5. Total loss (OU + FP + Consistency)
# ---------------------------------------------------------------------------

def test_total_loss_is_nonneg_finite():
    snaps = _make_destructive_snapshots()
    val = total_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ)
    assert val >= 0.0
    assert np.isfinite(val)


def test_total_loss_equals_sum_of_components():
    snaps = _make_destructive_snapshots()
    seed = 0
    combined = total_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_ou = ou_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_fp = fp_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    l_cons = consistency_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=seed)
    assert combined == pytest.approx(l_ou + l_fp + l_cons, rel=1e-10)


def test_total_loss_true_params_lower_than_wrong():
    snaps = _make_destructive_snapshots()
    l_true = total_loss(snaps, TRUE_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    l_wrong = total_loss(snaps, WRONG_A, TRUE_MU, TRUE_SIGMA, DT, n_proj=N_PROJ, seed=0)
    assert l_true < l_wrong
