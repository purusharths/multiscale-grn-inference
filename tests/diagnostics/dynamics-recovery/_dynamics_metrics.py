"""
Metrics on the RECOVERED DYNAMICAL SYSTEM itself, not just on theta's raw
parameter values or a single teacher-forced prediction step.

Everything else in this investigation (edge_corr, AUPRC, A_err, sigma_err,
one_step_ahead_w2) either compares theta_hat's numbers pointwise against
theta_true's, or predicts exactly one interval ahead from the TRUE
snapshot at t_k (so errors never compound). Neither answers: if you take
theta_hat and actually run it forward as a generative model from the
first observed snapshot, does it reproduce the observed trajectory over
the full horizon? That's what full_ou_trajectory_w2 and
full_fp_trajectory_w2 below do -- a compounding, open-loop rollout,
scored against every remaining snapshot, not just the next one.

Both are scored with the exact OU transition (exact_ou.py), not
Euler-Maruyama, regardless of which forward model theta_hat was FIT
with. This is deliberate: ../exact-ou-test/'s one_step_ahead_w2 scored
both the Euler-fit and exact-fit thetas with the (biased) Euler rollout,
which is a confound flagged but not resolved there. Scoring every theta
here with the unbiased exact transition removes that confound.

spectral_recovery compares the EIGENVALUES of A_hat against A_true's --
a compact summary of the recovered LINEAR DYNAMICAL SYSTEM's qualitative
behaviour (relaxation rates = real parts, oscillation = nonzero imaginary
parts) that A_err (a flat Frobenius norm) doesn't surface: two A's can
have similar A_err but very different dynamics if the errors land on
different eigen-directions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "exact-ou-test"))
sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from exact_ou import exact_fp_cell_population, exact_ou_transition  # noqa: E402


def full_ou_trajectory_w2(
    theta_hat: Theta,
    eval_snapshots: list[np.ndarray],
    dt: float,
    *,
    n_proj: int = 30,
    seed: int = 0,
) -> float:
    """Compounding OU rollout: start at the true snapshot 0, propagate
    forward through EVERY interval under theta_hat (feeding each step's
    OUTPUT into the next, never resetting to ground truth), and score
    sliced-W2 against the true snapshot at every step. Mean over steps."""
    rng = np.random.default_rng(seed)
    cloud = eval_snapshots[0]
    errs = []
    for k in range(len(eval_snapshots) - 1):
        cloud = exact_ou_transition(cloud, theta_hat, k, dt, rng=rng)
        errs.append(_sliced_w2(cloud, eval_snapshots[k + 1], n_proj, rng))
    return float(np.mean(errs))


def full_fp_trajectory_w2(
    theta_hat: Theta,
    eval_snapshots: list[np.ndarray],
    dt: float,
    *,
    n_proj: int = 30,
    seed: int = 0,
) -> float:
    """Same compounding idea as full_ou_trajectory_w2, but propagating the
    macro-scale population density (FP map) instead of a single cloud."""
    rng = np.random.default_rng(seed)
    chi = preprocessing(eval_snapshots[0])
    errs = []
    for k in range(len(eval_snapshots) - 1):
        chi = exact_fp_cell_population(chi, theta_hat, k, dt, rng=rng)
        n = len(eval_snapshots[k + 1])
        cloud = chi.resample(n, seed=int(rng.integers(0, 2**31))).T
        errs.append(_sliced_w2(cloud, eval_snapshots[k + 1], n_proj, rng))
    return float(np.mean(errs))


def spectral_recovery(A_true: np.ndarray, A_hat: np.ndarray) -> dict:
    """Compare the eigenvalues of A_true and A_hat -- the recovered LINEAR
    DYNAMICAL SYSTEM's relaxation rates (real parts) and oscillation
    (nonzero imaginary parts), not just a flat parameter-error norm."""
    eig_true = np.linalg.eigvals(A_true)
    eig_hat = np.linalg.eigvals(A_hat)

    # Sort by real part (then imaginary part) so the two vectors are
    # compared mode-by-mode in a consistent order; not a true matching
    # (no correspondence between eigenvectors is assumed), just a
    # deterministic, comparable ordering.
    eig_true_sorted = eig_true[np.lexsort((eig_true.imag, eig_true.real))]
    eig_hat_sorted = eig_hat[np.lexsort((eig_hat.imag, eig_hat.real))]

    vec_true = np.concatenate([eig_true_sorted.real, eig_true_sorted.imag])
    vec_hat = np.concatenate([eig_hat_sorted.real, eig_hat_sorted.imag])

    return {
        "eig_l2_dist": float(np.linalg.norm(vec_true - vec_hat)),
        "slowest_mode_true": float(eig_true.real.min()),
        "slowest_mode_hat": float(eig_hat.real.min()),
        "fastest_mode_true": float(eig_true.real.max()),
        "fastest_mode_hat": float(eig_hat.real.max()),
        "any_oscillatory_true": bool(np.any(np.abs(eig_true.imag) > 1e-9)),
        "any_oscillatory_hat": bool(np.any(np.abs(eig_hat.imag) > 1e-9)),
        "stable_true": bool(np.all(eig_true.real > 0)),
        "stable_hat": bool(np.all(eig_hat.real > 0)),
    }
