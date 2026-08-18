"""
Multi-timestep OU loss check.

Loss functions are loss.py's ou_fp_loss / total_loss (sum over all
consecutive snapshot pairs, sliced-W2 on KDE-resampled clouds) -- the same
implementation covered by tests/loss/ and tests/sde_fp/. This script adds
a `weighted_loss` (lam-weighted consistency term, no equivalent in loss.py)
and a Nelder-Mead `optimize()` wrapper around all of them, plus diagnostic
plots (housekeeping/ou_check_plots.py) and a loss-landscape sweep.

Usage:
    uv run python -m multsc_grn_inference.multi_step_ou_loss_check
"""
from __future__ import annotations

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from multsc_grn_inference.loss import (
    _ou_euler_maruyama,
    consistency_loss,
    fp_loss,
    ou_loss,
)
from multsc_grn_inference.loss import ou_fp_loss as _ou_fp_loss
from multsc_grn_inference.loss import total_loss as _total_loss
from multsc_grn_inference.housekeeping.ou_check_plots import (
    _ARROW_COLOR,
    plot_A_heatmap,
    plot_comparison,
    plot_convergence,
    plot_phase_portrait,
    plot_recovery,
    plot_snapshots_line,
)


# ---------------------------------------------------------------------------
# Parameter (en/de)coding: theta = [log_a0,...,log_aG (diag), off-diag
# row-major, mu0,...,muG, log_sigma]
# ---------------------------------------------------------------------------

def encode(A, mu, sigma):
    """theta = [log_a00,...,log_aGG (diag), off-diag row-major, mu0,...,muG, log_sigma]."""
    G = len(mu)
    log_diag = [np.log(A[g, g]) for g in range(G)]
    off_diag = [A[i, j] for i in range(G) for j in range(G) if i != j]
    return np.array(log_diag + off_diag + list(mu) + [np.log(sigma)])


def decode(theta):
    """Inverse of encode. Infers G from len(theta) = G**2 + G + 1."""
    n = len(theta)
    G = int(round((-1 + np.sqrt(1 + 4 * (n - 1))) / 2))
    A = np.zeros((G, G))
    for g in range(G):
        A[g, g] = np.exp(theta[g])
    off_idx = G
    for i in range(G):
        for j in range(G):
            if i != j:
                A[i, j] = theta[off_idx]
                off_idx += 1
    mu    = np.array(theta[G * G: G * G + G])
    sigma = float(np.exp(theta[-1]))
    return A, mu, sigma


# ---------------------------------------------------------------------------
# Multi-step losses (sum over all K-1 consecutive intervals) -- thin
# wrappers over loss.py, matching this script's (n_steps) call signature to
# loss.py's (n_substeps, n_proj)
# ---------------------------------------------------------------------------

def ou_fp_loss(snapshots, A, mu, sigma, dt, n_steps=1, seed=0, n_proj=100):
    """L_OU + L_FP summed over all intervals (loss.py implementation)."""
    return _ou_fp_loss(snapshots, A, mu, sigma, dt, n_substeps=n_steps, n_proj=n_proj, seed=seed)


def total_loss(snapshots, A, mu, sigma, dt, n_steps=1, seed=0, n_proj=100):
    """L_OU + L_FP + L_cons summed over all intervals (loss.py implementation)."""
    return _total_loss(snapshots, A, mu, sigma, dt, n_substeps=n_steps, n_proj=n_proj, seed=seed)


def weighted_loss(snapshots, A, mu, sigma, dt, n_steps=1, seed=0, n_proj=100, lam=0.5):
    """L_OU + L_FP + lam * L_cons summed over all intervals."""
    kw = dict(n_substeps=n_steps, n_proj=n_proj, seed=seed)
    return (
        ou_loss(snapshots, A, mu, sigma, dt, **kw)
        + fp_loss(snapshots, A, mu, sigma, dt, **kw)
        + lam * consistency_loss(snapshots, A, mu, sigma, dt, **kw)
    )


# ---------------------------------------------------------------------------
# Optimizer for multi-step losses
# ---------------------------------------------------------------------------

def optimize(loss_fn, snapshots, dt, init_theta, n_steps=50, seed=0, n_proj=100):
    """
    Minimize loss_fn(snapshots, A, mu, sigma, dt, ...) over
    theta = [log_a0, log_a1, mu0, mu1, log_sigma] via Nelder-Mead.

    Returns
    -------
    (A_opt, mu_opt, sigma_opt), res, history
    history : list of objective values at every function evaluation
    """
    history = []

    def objective(theta):
        A, mu, sigma = decode(theta)
        val = loss_fn(snapshots, A, mu, sigma, dt,
                      n_steps=n_steps, seed=seed, n_proj=n_proj)
        history.append(val)
        return val

    res = scipy.optimize.minimize(
        objective, init_theta, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    return decode(res.x), res, history


# ---------------------------------------------------------------------------
# Loss landscape (2D heatmap, vary a0 × mu0)
# ---------------------------------------------------------------------------

def plot_loss_landscape(snapshots, dt, true_A, true_mu, true_sigma,
                        out_path="loss_landscape.png",
                        n_grid=20, seed=0, lam=0.5):
    """
    2D loss heatmap over a0 × mu0 for the three combined multi-step losses.
    All other parameters held at their true values.
    """
    a0_true  = true_A[0, 0]
    mu0_true = true_mu[0]

    a0_vals  = np.linspace(max(a0_true  * 0.2, 0.1), a0_true  * 2.5, n_grid)
    mu0_vals = np.linspace(max(mu0_true * 0.3, 0.1), mu0_true * 1.7, n_grid)

    loss_fns = {
        "OU+FP":              ou_fp_loss,
        "OU+FP+Cons":         total_loss,
        f"OU+FP+{lam}·Cons":  lambda snaps, A, mu, sigma, dt, **kw: weighted_loss(
                                   snaps, A, mu, sigma, dt, lam=lam, **kw),
    }
    grids = {name: np.zeros((n_grid, n_grid)) for name in loss_fns}

    total_evals = n_grid * n_grid * len(loss_fns)
    done = 0
    for j, a0 in enumerate(a0_vals):
        for i, mu0 in enumerate(mu0_vals):
            A_       = true_A.copy(); A_[0, 0] = a0
            mu_      = true_mu.copy(); mu_[0]  = mu0
            for name, fn in loss_fns.items():
                grids[name][i, j] = fn(
                    snapshots, A_, mu_, true_sigma, dt,
                    seed=seed,
                )
            done += len(loss_fns)
            if done % (n_grid * len(loss_fns)) == 0:
                print(f"  landscape: {done}/{total_evals}")

    step = max(1, n_grid // 10)
    qa   = a0_vals[::step]
    qmu  = mu0_vals[::step]
    QA, QMU = np.meshgrid(qa, qmu)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, grid) in zip(axes, grids.items()):
        im = ax.contourf(a0_vals, mu0_vals, grid, levels=20, cmap="RdYlGn_r")
        fig.colorbar(im, ax=ax, label="loss")

        dL_dmu, dL_da = np.gradient(grid, mu0_vals, a0_vals)
        ga  = -dL_da[::step, ::step]
        gmu = -dL_dmu[::step, ::step]
        mag = np.sqrt(ga**2 + gmu**2) + 1e-12
        ax.quiver(QA, QMU, ga / mag, gmu / mag,
                  color=_ARROW_COLOR, alpha=0.8, scale=25, width=0.004)

        ax.axvline(a0_true,  color="white", lw=1.5, ls="--")
        ax.axhline(mu0_true, color="white", lw=1.5, ls="--")
        ax.scatter([a0_true], [mu0_true], color="white", s=80,
                   marker="*", zorder=5, label="true")
        ax.set_xlabel("a0  (rate gene 0)")
        ax.set_ylabel("μ₀  (target gene 0)")
        ax.set_title(name)
        ax.legend(fontsize=8)

    K = len(snapshots)
    fig.suptitle(
        f"Loss landscape: a0 vs μ₀  ({K} snapshots, {K-1} intervals, "
        "all other params fixed at true)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    OUT = Path("output/multi_step_ou_check")
    OUT.mkdir(parents=True, exist_ok=True)

    TRUE_A     = np.array([[1.5, 0.3, 0.0],
                           [0.2, 1.2, 0.1],
                           [0.0, 0.2, 0.9]])
    TRUE_MU    = np.array([3.0, 2.5, 2.0])
    TRUE_SIGMA = 0.3
    DT         = 0.5
    N_SNAPS    = 4
    N_CELLS    = 300
    SIM_STEPS  = 50
    OPT_STEPS  = 10
    G          = len(TRUE_MU)

    # --- generate snapshots ---------------------------------------------------
    rng = np.random.default_rng(42)
    X = rng.multivariate_normal(TRUE_MU * 0.3, 0.1 * np.eye(G), size=N_CELLS)
    X = np.maximum(X, 0.05)

    snapshots = [X.copy()]
    for _ in range(N_SNAPS - 1):
        X = _ou_euler_maruyama(X, TRUE_A, TRUE_MU, TRUE_SIGMA, DT,
                               n_substeps=SIM_STEPS, rng=rng)
        snapshots.append(X.copy())

    times = [k * DT for k in range(N_SNAPS)]
    print(f"Generated {N_SNAPS} snapshots at t = {times}")

    # --- line plot ------------------------------------------------------------
    plot_snapshots_line(snapshots, DT, out_path=OUT / "line_plot.png",
                        true_A=TRUE_A, true_mu=TRUE_MU)

    # --- optimise: ou+fp vs ou+fp+cons ---------------------------------------
    init_theta = encode(np.diag([1.0] * G), np.array([2.0] * G), 0.5)

    loss_fns = {
        "ou+fp":      ou_fp_loss,
        "ou+fp+cons": total_loss,
    }

    results   = {}
    histories = {}
    for name, fn in loss_fns.items():
        print(f"Optimising [{name}] ...")
        (A_opt, mu_opt, sigma_opt), res, hist = optimize(
            fn, snapshots, DT, init_theta.copy(),
            n_steps=OPT_STEPS,
        )
        results[name] = {
            "A": A_opt,
            "mu": mu_opt,
            "sigma": sigma_opt,
            **{f"a{i}{j}": A_opt[i, j] for i in range(G) for j in range(G)},
            **{f"mu{g}": mu_opt[g] for g in range(G)},
        }
        histories[name] = hist
        print(f"  converged={res.success}  fun={res.fun:.5f}")
        print(f"  A={A_opt.diagonal().round(3)}  mu={mu_opt.round(3)}  sigma={sigma_opt:.3f}")

    print(f"\nTrue A:\n{TRUE_A}")
    print(f"True mu={TRUE_MU}  sigma={TRUE_SIGMA}")

    # --- plots ----------------------------------------------------------------
    plot_convergence(histories, out_path=OUT / "convergence.png")
    plot_comparison(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                    out_path=OUT / "comparison.png")
    plot_recovery(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                  out_path=OUT / "recovery.png")
    plot_phase_portrait(snapshots, TRUE_A, TRUE_MU, DT,
                        out_path=OUT / "phase_portrait.svg",
                        results=results)
    plot_A_heatmap(results, TRUE_A, out_path=OUT / "A_heatmap.png")

    print(f"\nAll outputs in {OUT.resolve()}")
