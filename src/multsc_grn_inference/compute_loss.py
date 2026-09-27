"""
Algorithm 1, function ComputeLoss (paper lines 40-49):

    function ComputeLoss(theta, {chi_tk})
        L_OU   <- sum_{k=1}^{T-1} W2( KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}} )
        L_FP   <- sum_{k=1}^{T-1} W2( FP(chi_tk; theta), chi_{t_{k+1}} )
        L_cons <- sum_{k=1}^{T-1} W2( KDE(Phi_OU(X_tk; theta)), FP(chi_tk; theta) )
        return L_OU + L_FP + L_cons

Phi_OU is OUGeneExpression (micro-scale, needs per-cell tracking); FP is
FPCellPopulation (macro-scale, needs only the estimated density). Each term
is exposed as its own function (loss_ou, loss_fp, loss_cons) so they can be
computed/tested independently; compute_loss sums them, matching the paper.

W2 is approximated via sliced Wasserstein-2 (random 1D projections, order
statistics compared along each) for tractability in > 1 dimension.

On the KDE round-trip (removed 2026-09). The formulas above are written in
terms of densities, but sliced-W2 is a SAMPLE-based distance and every density
here is estimated from a particle cloud we already hold. Earlier versions fitted
a KDE to those particles and then redrew samples from it -- points -> density ->
points -- on both sides of every comparison. That round-trip bought nothing
(_sliced_w2 already handles unequal sample sizes by quantile interpolation) and
cost two things: a systematic (1 + h^2) covariance inflation, h being Scott's
bandwidth, so the bias GROWS with gene count (1.28x at G=5, 1.43x at G=15); and
a large share of the objective's run-to-run noise, which mattered because the
signal distinguishing individual network edges sits below that noise floor.

So the terms are now computed on the clouds themselves. The one resampling that
remains is chi_tk -> particles INSIDE the FP map (fp_particles), because that is
what makes L_FP a macro-scale term at all: remove it and L_FP collapses onto
L_OU and L_cons becomes identically zero.

All three terms are implemented and working. (An earlier version of this
docstring said loss_fp/loss_cons raised NotImplementedError pending a real JKO
solver; the particle-JKO step in fp_cell_population.py supersedes that.)
"""
from __future__ import annotations

import os

import numpy as np

from multsc_grn_inference.exact_ou import exact_fp_particles, exact_ou_transition
from multsc_grn_inference.fp_cell_population import fp_particles
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.theta import Theta

# Which forward model the loss terms propagate with.
#
# "exact" (default) uses the closed-form OU transition (exact_ou.py). "euler"
# uses ou_gene_expression's single Euler-Maruyama step, which is what every
# result before 2026-09-21 was computed under.
#
# This is not a tuning knob -- Euler-Maruyama at the configs used here is
# qualitatively wrong, and the default was changed because of it. Measured on
# the 8-gene non-interventional dataset (dt=0.5), summed sliced-W2 over all
# intervals, true A versus the same A with every edge deleted:
#
#     forward model            L(true A)   L(no edges)   edge signal
#     Euler-Maruyama n_sub=1     0.46684       0.15188      -0.31496
#     Euler-Maruyama n_sub=5     0.00319       0.08871      +0.08551
#     exact (Van Loan)           0.00932       0.11125      +0.10193
#
# With a single Euler step the edge signal is NEGATIVE: the loss fits an
# edge-free matrix three times better than the truth, so its minimum is not at
# the true parameters and no optimizer can find them. The mechanism is that
# edges raise A's eigenvalues and make them complex; at dt=0.5 the largest
# reaches lambda*dt = 1.07, where Euler's contraction (1 - lambda*dt) turns
# NEGATIVE and the mode overshoots past equilibrium, while the exact
# exp(-lambda*dt) = 0.34. A diagonal A has smaller, purely real eigenvalues and
# escapes this -- so the integrator systematically rewards deleting edges.
#
# That single default is why edge recovery measured at chance across every run
# in this project, interventional and non-interventional alike.
FORWARD_MODEL = os.environ.get("GRN_FORWARD_MODEL", "exact")


def _propagate(X, theta, k, dt, rng):
    if FORWARD_MODEL == "euler":
        return ou_gene_expression(X, theta, k, dt, rng=rng)
    return exact_ou_transition(X, theta, k, dt, rng=rng)


def _fp_propagate(chi, theta, k, dt, n, rng):
    if FORWARD_MODEL == "euler":
        return fp_particles(chi, theta, k, dt, n_particles=n, rng=rng)
    return exact_fp_particles(chi, theta, k, dt, n_particles=n, rng=rng)


def _sliced_w2(X: np.ndarray, Y: np.ndarray, n_proj: int, rng: np.random.Generator) -> float:
    """Sliced Wasserstein-2^2 between particle clouds X (N, G) and Y (M, G)."""
    G = X.shape[1]
    dirs = rng.standard_normal((n_proj, G))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    n_min = min(len(X), len(Y))
    t_x = np.linspace(0, 1, len(X))
    t_y = np.linspace(0, 1, len(Y))
    t_q = np.linspace(0, 1, n_min)

    total = 0.0
    for d in dirs:
        xq = np.interp(t_q, t_x, np.sort(X @ d))
        yq = np.interp(t_q, t_y, np.sort(Y @ d))
        total += float(np.mean((xq - yq) ** 2))
    return total / n_proj


def loss_ou(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """
    L_OU = sum_k W2( KDE(Phi_OU(X_tk; theta)), chi_{t_{k+1}} ).

    Implemented on the particle clouds the two densities are estimated FROM,
    not on samples redrawn from them -- see the module docstring's note on the
    KDE round-trip. chi_snapshots is unused here as a result; it stays in the
    signature so the three loss terms remain interchangeable to callers.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(X_snapshots) - 1):
        propagated = _propagate(X_snapshots[k], theta, k, dt, rng)
        total += _sliced_w2(propagated, X_snapshots[k + 1], n_proj, rng)
    return float(total)


def loss_fp(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """
    L_FP = sum_k W2( FP(chi_tk; theta), chi_{t_{k+1}} ).

    Uses fp_particles, not fp_cell_population: the FP map's INPUT resampling
    (chi_tk -> particles) is kept, since that is what makes this the macro-scale
    term, but its output is compared as particles rather than re-fitted to a KDE
    and redrawn.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(chi_snapshots) - 1):
        nu_star = _fp_propagate(chi_snapshots[k], theta, k, dt, len(X_snapshots[k]), rng)
        total += _sliced_w2(nu_star, X_snapshots[k + 1], n_proj, rng)
    return float(total)


def loss_cons(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> float:
    """
    L_cons = sum_k W2( KDE(Phi_OU(X_tk; theta)), FP(chi_tk; theta) ).

    Both sides compared as particle clouds. Note what this term is actually
    measuring: both arms apply the SAME Phi_OU with the SAME theta, and differ
    only in whether the input was the observed cells or a KDE-resampled
    surrogate of them. So L_cons scores how much the FP map's input resampling
    perturbs the result -- which is why it is degenerate alone, and why it is
    numerically tiny next to L_OU and L_FP.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(X_snapshots) - 1):
        propagated = _propagate(X_snapshots[k], theta, k, dt, rng)
        nu_star = _fp_propagate(chi_snapshots[k], theta, k, dt, len(X_snapshots[k]), rng)
        total += _sliced_w2(propagated, nu_star, n_proj, rng)
    return float(total)


def compute_loss(
    theta: Theta,
    X_snapshots: list[np.ndarray],
    chi_snapshots: list,
    dt: float,
    *,
    n_proj: int = 100,
    seed: int = 0,
) -> dict:
    """
    Returns {"L_OU": ..., "L_FP": ..., "L_cons": ..., "total": L_OU+L_FP+L_cons}.

    X_snapshots   : list of (N, G) observed cell snapshots, length T
    chi_snapshots : list of fitted densities (Preprocessing output), length T
    """
    l_ou = loss_ou(theta, X_snapshots, chi_snapshots, dt, n_proj=n_proj, seed=seed)
    l_fp = loss_fp(theta, X_snapshots, chi_snapshots, dt, n_proj=n_proj, seed=seed)
    l_cons = loss_cons(theta, X_snapshots, chi_snapshots, dt, n_proj=n_proj, seed=seed)
    return {"L_OU": l_ou, "L_FP": l_fp, "L_cons": l_cons, "total": l_ou + l_fp + l_cons}
