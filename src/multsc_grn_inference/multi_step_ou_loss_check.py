"""
Multi-timestep OU loss check.

Reuses primitives from simple_ou_loss_check.py.
Losses sum over all consecutive snapshot pairs [t0,t1], [t1,t2], ...

Usage:
    uv run python -m multsc_grn_inference.multi_step_ou_loss_check
"""
from __future__ import annotations

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from multsc_grn_inference.simple_ou_loss_check import (
    simulate_ou,
    ou_fp_loss_single_step,
    total_loss_single_step,
    weighted_loss_single_step,
    encode,
    decode,
    plot_snapshots_line,
    plot_phase_portrait,
    plot_recovery,
)


# ---------------------------------------------------------------------------
# Multi-step losses  (sum over all K-1 consecutive intervals)
# ---------------------------------------------------------------------------

def ou_fp_loss(snapshots, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """L_OU + L_FP summed over all intervals."""
    return sum(
        ou_fp_loss_single_step(
            snapshots[k], snapshots[k + 1], A, mu, sigma, dt,
            n_steps=n_steps, epsilon=epsilon, seed=seed + k,
        )
        for k in range(len(snapshots) - 1)
    )


def total_loss(snapshots, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """L_OU + L_FP + L_cons summed over all intervals."""
    return sum(
        total_loss_single_step(
            snapshots[k], snapshots[k + 1], A, mu, sigma, dt,
            n_steps=n_steps, epsilon=epsilon, seed=seed + k,
        )
        for k in range(len(snapshots) - 1)
    )


def weighted_loss(snapshots, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0, lam=0.5):
    """L_OU + L_FP + lam * L_cons summed over all intervals."""
    return sum(
        weighted_loss_single_step(
            snapshots[k], snapshots[k + 1], A, mu, sigma, dt,
            n_steps=n_steps, epsilon=epsilon, seed=seed + k, lam=lam,
        )
        for k in range(len(snapshots) - 1)
    )


# ---------------------------------------------------------------------------
# Optimizer for multi-step losses
# ---------------------------------------------------------------------------

def optimize(loss_fn, snapshots, dt, init_theta, n_steps=50, epsilon=0.01, seed=0):
    """
    Minimize loss_fn(snapshots, A, mu, sigma, dt, ...) over
    θ = [log_a0, log_a1, mu0, mu1, log_sigma] via Nelder-Mead.

    Returns
    -------
    (A_opt, mu_opt, sigma_opt), res
    """
    def objective(theta):
        A, mu, sigma = decode(theta)
        return loss_fn(snapshots, A, mu, sigma, dt,
                       n_steps=n_steps, epsilon=epsilon, seed=seed)

    res = scipy.optimize.minimize(
        objective, init_theta, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    return decode(res.x), res


# ---------------------------------------------------------------------------
# Loss landscape (2D heatmap, vary a0 × mu0)
# ---------------------------------------------------------------------------

def plot_loss_landscape(snapshots, dt, true_A, true_mu, true_sigma,
                        out_path="loss_landscape.png",
                        n_grid=20, epsilon=0.01, seed=0, lam=0.5):
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
            A_  = np.diag([a0, true_A[1, 1]])
            mu_ = np.array([mu0, true_mu[1]])
            for name, fn in loss_fns.items():
                grids[name][i, j] = fn(
                    snapshots, A_, mu_, true_sigma, dt,
                    epsilon=epsilon, seed=seed,
                )
            done += len(loss_fns)
            if done % (n_grid * len(loss_fns)) == 0:
                print(f"  landscape: {done}/{total_evals}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, grid) in zip(axes, grids.items()):
        im = ax.contourf(a0_vals, mu0_vals, grid, levels=20, cmap="RdYlGn_r")
        fig.colorbar(im, ax=ax, label="loss")
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

    TRUE_A     = np.diag([1.5, 1.2])
    TRUE_MU    = np.array([3.0, 2.5])
    TRUE_SIGMA = 0.3
    DT         = 0.5          # interval between consecutive snapshots
    N_SNAPS    = 4             # t = 0, 0.5, 1.0, 1.5
    N_CELLS    = 300
    SIM_STEPS  = 50
    OPT_STEPS  = 10
    LAM        = 0.5

    # --- generate snapshots ---------------------------------------------------
    rng = np.random.default_rng(42)
    X = rng.multivariate_normal(TRUE_MU * 0.3, 0.1 * np.eye(2), size=N_CELLS)
    X = np.maximum(X, 0.05)

    snapshots = [X.copy()]
    for _ in range(N_SNAPS - 1):
        X = simulate_ou(X, TRUE_A, TRUE_MU, TRUE_SIGMA, DT,
                        n_steps=SIM_STEPS, rng=rng)
        snapshots.append(X.copy())

    times = [k * DT for k in range(N_SNAPS)]
    print(f"Generated {N_SNAPS} snapshots at t = {times}")

    # --- line plot (first) ----------------------------------------------------
    plot_snapshots_line(snapshots, DT, out_path=OUT / "line_plot.png",
                        true_A=TRUE_A, true_mu=TRUE_MU)

    # --- loss landscape -------------------------------------------------------
    print("Computing loss landscape (may take a few minutes) ...")
    plot_loss_landscape(snapshots, DT, TRUE_A, TRUE_MU, TRUE_SIGMA,
                        out_path=OUT / "loss_landscape.png",
                        n_grid=18, epsilon=0.05, lam=LAM)

    # --- optimise combined losses ---------------------------------------------
    init_theta = encode(
        np.diag([1.0, 1.0]),
        np.array([2.0, 2.0]),
        0.5,
    )

    loss_fns = {
        "ou+fp":              ou_fp_loss,
        "ou+fp+cons":         total_loss,
        f"ou+fp+{LAM}*cons":  lambda snaps, A, mu, sigma, dt, **kw: weighted_loss(
                                  snaps, A, mu, sigma, dt, lam=LAM, **kw),
    }

    results = {}
    for name, fn in loss_fns.items():
        print(f"Optimising [{name}] ...")
        (A_opt, mu_opt, sigma_opt), res = optimize(
            fn, snapshots, DT, init_theta.copy(),
            n_steps=OPT_STEPS, epsilon=0.05,
        )
        results[name] = {
            "a0": A_opt[0, 0], "a1": A_opt[1, 1],
            "mu0": mu_opt[0],  "mu1": mu_opt[1],
            "sigma": sigma_opt,
            "converged": res.success, "fun": res.fun,
        }
        print(f"  converged={res.success}  fun={res.fun:.5f}")
        print(f"  A={A_opt.diagonal().round(3)}  mu={mu_opt.round(3)}  sigma={sigma_opt:.3f}")

    print(f"\nTrue: A={TRUE_A.diagonal()}  mu={TRUE_MU}  sigma={TRUE_SIGMA}")

    # --- phase portrait (after optimisation so recovered means can be overlaid)
    plot_phase_portrait(snapshots, TRUE_A, TRUE_MU, DT,
                        out_path=OUT / "phase_portrait.png",
                        results=results)

    # --- recovery plot --------------------------------------------------------
    plot_recovery(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                  out_path=OUT / "recovery.png")

    print(f"\nAll outputs in {OUT.resolve()}")
