"""
Loss-function diagnostic: parameter recovery under each loss combination.

Uses NetworkSimulatorNonStationaryMu (2-gene, diagonal A, sigmoid μ(t)) to
generate two consecutive cross-sectional snapshots near the sigmoid transition.
The same cells are available at both timepoints (non-destructive protocol), but
the underlying μ(t) is changing (non-stationary dynamics).

Five loss combinations are compared:
  ou          – L_OU only
  fp          – L_FP only
  consistency – L_cons only
  ou_fp       – L_OU + L_FP
  total       – L_OU + L_FP + L_cons

For each combination, scipy.optimize.minimize (Nelder-Mead) recovers five
parameters of a diagonal constant-μ OU model:

  θ = [log a₀, log a₁, μ₀, μ₁, log σ]

Outputs
-------
  output/loss_diagnostic/snapshots/snapshot_t0.csv
  output/loss_diagnostic/snapshots/snapshot_t1.csv
  output/loss_diagnostic/snapshots/metadata.csv
  output/loss_diagnostic/true_params.csv
  output/loss_diagnostic/optimization_summary.csv
  output/loss_diagnostic/parameter_recovery.png

Usage
-----
    uv run python -m multsc_grn_inference.loss_diagnostic
    # or from project root:
    uv run python src/multsc_grn_inference/loss_diagnostic.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from multsc_grn_inference.datagen.non_stationary_sim import NetworkSimulatorNonStationaryMu
from multsc_grn_inference.loss import (
    consistency_loss,
    fp_loss,
    ou_fp_loss,
    ou_loss,
    total_loss,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIM_SEED       = 42
NUM_GENES      = 2
N_CELLS        = 400
SIM_T          = 4.0        # total simulation horizon
SIM_DT         = 0.005

# 4 cross-sectional snapshots at uniform Δt = 0.5 → 3 intervals.
# Cells start far from equilibrium (IC = 0.3 × μ₀) so drift A(μ−c) is large.
# Multiple intervals give enough moment constraints to identify off-diagonal A.
T_SNAPS = [0.0, 0.5, 1.0, 1.5]   # snapshot times; Δt = 0.5 between each pair

N_PROJ    = 50    # sliced-W2 random projections during optimisation (speed/accuracy trade-off)
OPT_SEED  = 0     # fixed seed → deterministic loss each call → Nelder-Mead stable

OUT_DIR = Path("output/loss_diagnostic")

LOSS_FNS = {
    "ou":          ou_loss,
    "fp":          fp_loss,
    "consistency": consistency_loss,
    "ou_fp":       ou_fp_loss,
    "total":       total_loss,
}
LOSS_ORDER = list(LOSS_FNS.keys())   # canonical display order

PARAM_META = {   # internal key → y-axis label
    "a00":   "A[0,0]  (self-rate gene 0)",
    "a01":   "A[0,1]  (gene 1 → gene 0)",
    "a10":   "A[1,0]  (gene 0 → gene 1)",
    "a11":   "A[1,1]  (self-rate gene 1)",
    "mu0":   "μ₀  (target gene 0)",
    "mu1":   "μ₁  (target gene 1)",
    "sigma": "σ  (diffusion)",
}


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def generate_and_save(out_dir: Path):
    """
    Build 2-gene non-stationary OU snapshots with a far-from-equilibrium start.

    The simulator provides A, D, and μ(t).  We generate a custom initial
    condition at 0.3 × μ₀ (well below equilibrium) and run the SDE manually
    for T_K1 seconds.  This makes the drift A(μ − c) large and clearly
    differentiable, so the rate parameters a0, a1 are identifiable from W2.

    Returns
    -------
    snaps       : list[ndarray]   [X_tk, X_tk1], each (N_CELLS, NUM_GENES)
    dt          : float           T_K1 - T_K
    true_params : dict            ground-truth {a0, a1, mu0, mu1, sigma}
    sim         : NetworkSimulatorNonStationaryMu
    """
    print(f"  Simulator: {NUM_GENES} genes, network_density=1.0 (fully connected A), "
          f"mu_mode='sigmoid', seed={SIM_SEED}")
    sim = NetworkSimulatorNonStationaryMu(
        num_genes=NUM_GENES,
        network_density=1.0,   # fully connected → off-diagonal GRN entries non-zero
        seed=SIM_SEED,
        mu_mode="sigmoid",
    )

    dt = float(T_SNAPS[1] - T_SNAPS[0])   # uniform interval between snapshots

    # --- Far-from-equilibrium initial condition: cells start at 0.3 × μ₀
    rng_ic = np.random.default_rng(SIM_SEED + 100)
    X = rng_ic.multivariate_normal(sim.mu0 * 0.3, 0.1 * np.eye(NUM_GENES), size=N_CELLS)
    X = np.maximum(X, 0.05)

    # --- Euler–Maruyama: advance through each snapshot interval
    snaps = [X.copy()]   # snapshot at T_SNAPS[0]
    for k in range(len(T_SNAPS) - 1):
        t_start = T_SNAPS[k]
        t_end   = T_SNAPS[k + 1]
        n_steps = int(round((t_end - t_start) / SIM_DT))
        for step in range(n_steps):
            t     = t_start + step * SIM_DT
            mu    = sim.mu_t(t, SIM_T)
            drift = (mu - X) @ sim.A.T
            noise = np.sqrt(np.diag(sim.D) * SIM_DT) * rng_ic.standard_normal(X.shape)
            X     = np.maximum(X + drift * SIM_DT + noise, 0.05)
        snaps.append(X.copy())

    # --- Save snapshots
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    gene_cols = [f"gene_{g}" for g in range(NUM_GENES)]
    for i, (t_val, snap) in enumerate(zip(T_SNAPS, snaps)):
        pd.DataFrame(snap, columns=gene_cols).to_csv(
            snap_dir / f"snapshot_t{i}.csv", index=False
        )
    pd.DataFrame([{
        "t_snaps": str(T_SNAPS), "dt": dt,
        "n_cells": N_CELLS, "num_genes": NUM_GENES,
        "mu_mode": "sigmoid", "sim_seed": SIM_SEED,
        "ic": "0.3 * mu0  (far from equilibrium)",
    }]).to_csv(snap_dir / "metadata.csv", index=False)

    # Ground-truth parameters at T_SNAPS[0] (sigmoid barely active at t=0)
    mu_t0   = sim.mu_t(T_SNAPS[0], SIM_T)
    sigma_t = float(np.mean(np.sqrt(np.diag(sim.D))))
    true_p  = {
        "a00":   float(sim.A[0, 0]),
        "a01":   float(sim.A[0, 1]),
        "a10":   float(sim.A[1, 0]),
        "a11":   float(sim.A[1, 1]),
        "mu0":   float(mu_t0[0]),
        "mu1":   float(mu_t0[1]),
        "sigma": sigma_t,
    }
    pd.DataFrame([true_p]).to_csv(out_dir / "true_params.csv", index=False)

    means = [s.mean(axis=0).round(2) for s in snaps]
    print(f"  IC: cells start at 0.3 × μ₀ ≈ {(sim.mu0 * 0.3).round(2)},  "
          f"μ₀ ≈ {sim.mu0.round(2)}")
    print(f"  Snapshot means: {means}")
    print(f"  dt={dt},  n_intervals={len(T_SNAPS)-1},  T_SNAPS={T_SNAPS}")
    print(f"  True params: {true_p}")
    return snaps, dt, true_p, sim


# ---------------------------------------------------------------------------
# 2. Optimisation helpers
# ---------------------------------------------------------------------------

def _decode(theta: np.ndarray):
    """
    θ = [log_a00, r01, r10, log_a11, μ0, μ1, log_σ] → (A, mu, sigma).

    Off-diagonals: a_ij = tanh(r_ij) × 0.9 × a_ii guarantees
    diagonal dominance (|a_ij| < a_ii) throughout the search.
    """
    log_a00, r01, r10, log_a11, mu0, mu1, log_sigma = theta
    a00 = np.exp(log_a00)
    a11 = np.exp(log_a11)
    a01 = np.tanh(r01) * 0.9 * a00
    a10 = np.tanh(r10) * 0.9 * a11
    A   = np.array([[a00, a01], [a10, a11]])
    mu  = np.array([mu0, mu1])
    sigma = float(np.exp(log_sigma))
    return A, mu, sigma


def _theta_to_named(theta: np.ndarray) -> dict:
    log_a00, r01, r10, log_a11, mu0, mu1, log_sigma = theta
    a00 = float(np.exp(log_a00))
    a11 = float(np.exp(log_a11))
    return {
        "a00":   a00,
        "a01":   float(np.tanh(r01) * 0.9 * a00),
        "a10":   float(np.tanh(r10) * 0.9 * a11),
        "a11":   a11,
        "mu0":   float(mu0),
        "mu1":   float(mu1),
        "sigma": float(np.exp(log_sigma)),
    }


def optimise_loss(loss_fn, snaps, dt, init_theta):
    """Run Nelder-Mead minimisation for one loss combination."""
    n_calls = [0]

    def objective(theta):
        n_calls[0] += 1
        A, mu, sigma = _decode(theta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return loss_fn(snaps, A, mu, sigma, dt, n_proj=N_PROJ, seed=OPT_SEED)

    t0 = time.perf_counter()
    res = scipy.optimize.minimize(
        objective, init_theta,
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    elapsed = time.perf_counter() - t0
    print(f"    converged={res.success}  fun={res.fun:.5f}"
          f"  calls={n_calls[0]}  time={elapsed:.1f}s")
    return res.x, res.fun


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_recovery(results: dict, true_params: dict, out_path: Path):
    """
    One subplot per parameter, x-axis = loss combination.
    Blue line+markers = recovered value.
    Dashed black = ground truth.
    """
    params     = list(PARAM_META.keys())
    loss_names = LOSS_ORDER
    xs         = np.arange(len(loss_names))

    fig, axes = plt.subplots(len(params), 1, figsize=(9, 2.8 * len(params)), sharex=True)
    fig.suptitle(
        "Parameter recovery by optimised loss combination\n"
        "(2-gene sigmoid-OU | non-destructive snapshots | "
        f"N={N_CELLS} cells | dashed = ground truth)",
        fontsize=11, y=1.01,
    )

    for ax, param in zip(axes, params):
        ys = [results[name][param] for name in loss_names]
        ax.plot(xs, ys, marker="o", color="#2166ac", lw=1.8, markersize=7, zorder=3)
        ax.axhline(true_params[param], ls="--", color="black", lw=1.5, label="ground truth")
        ax.set_ylabel(PARAM_META[param], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3, ls=":")
        # annotate recovered values
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="#2166ac")

    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(loss_names, fontsize=11, rotation=15, ha="right")
    axes[-1].tick_params(axis="x", labelsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_loss_values(results: dict, out_path: Path):
    """Bar chart of final objective values, one bar per loss combination."""
    names  = LOSS_ORDER
    values = [results[n]["objective"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Final objective (sliced-W₂)")
    ax.set_title("Minimised loss value per combination\n(lower ≠ better recovery for consistency-only)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_snapshot_scatter(snaps: list[np.ndarray], sim, dt: float, out_path: Path):
    """
    Scatter of all snapshots coloured by timepoint, with the observed mean
    trajectory and the OU mean trajectory under true A overlaid.
    """
    from matplotlib.lines import Line2D

    cmap   = plt.cm.Blues
    colors = [cmap(0.35 + 0.55 * i / max(len(snaps) - 1, 1)) for i in range(len(snaps))]

    fig, ax = plt.subplots(figsize=(6, 5))
    means = []
    for i, (snap, col) in enumerate(zip(snaps, colors)):
        ax.scatter(snap[:, 0], snap[:, 1], s=3, alpha=0.25, color=col)
        means.append(snap.mean(axis=0))

    # Observed mean trajectory
    mx = [m[0] for m in means]
    my = [m[1] for m in means]
    ax.plot(mx, my, color="black", lw=1.5, marker="o", ms=5, zorder=4,
            label="observed mean")

    # Predicted mean trajectory under true A (first-order Euler approximation)
    true_A = sim.A
    m_pred = [means[0].copy()]
    for k in range(len(T_SNAPS) - 1):
        mu_k = sim.mu_t(T_SNAPS[k], SIM_T)
        m_next = m_pred[-1] + (mu_k - m_pred[-1]) @ true_A.T * dt
        m_pred.append(m_next)
    px = [m[0] for m in m_pred]
    py = [m[1] for m in m_pred]
    ax.plot(px, py, color="green", lw=1.5, ls="--", marker="s", ms=5, zorder=3,
            label="OU mean (true A)")

    handles = (
        [Line2D([0], [0], color=colors[i], marker="o", ms=5, ls="",
                label=f"t={T_SNAPS[i]:.1f}") for i in range(len(snaps))]
        + [
            Line2D([0], [0], color="black", lw=1.5, marker="o", ms=4, label="observed mean"),
            Line2D([0], [0], color="green", lw=1.5, ls="--", marker="s", ms=4, label="OU (true A)"),
        ]
    )
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_xlabel("gene 0")
    ax.set_ylabel("gene 1")
    ax.set_title(
        f"Snapshot scatter  ({len(snaps)} timepoints, Δt={dt:.2f})\n"
        f"n_cells={N_CELLS},  network_density=1.0"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR.resolve()}\n")

    # 1. Generate data --------------------------------------------------
    print("[1/4] Generating 2-gene non-stationary OU snapshots ...")
    snaps, dt, true_params, sim = generate_and_save(OUT_DIR)

    # Initial θ: plausible but wrong starting point
    # Off-diagonals initialised at 0 (r=0 → tanh(0)=0 → a_ij=0)
    init_theta = np.array([
        np.log(1.0),    # log_a00   (true diagonal ≈ 1.4–1.8)
        0.0,            # r01       → a01 = 0  (true off-diagonal unknown sign)
        0.0,            # r10       → a10 = 0
        np.log(1.0),    # log_a11
        2.0,            # mu0       (true ≈ 2.5–3.5 at t_k)
        2.0,            # mu1
        np.log(0.3),    # log_sigma (true ≈ 0.32–0.45)
    ])

    # 2. Optimise -------------------------------------------------------
    print("\n[2/4] Optimising each loss combination ...")
    results = {}
    for name, loss_fn in LOSS_FNS.items():
        print(f"  [{name}]", flush=True)
        theta_opt, fun = optimise_loss(loss_fn, snaps, dt, init_theta.copy())
        results[name] = {**_theta_to_named(theta_opt), "objective": fun}

    # 3. Save summary ---------------------------------------------------
    summary_df = pd.DataFrame([{**{"loss": k}, **v} for k, v in results.items()])
    summary_df.to_csv(OUT_DIR / "optimization_summary.csv", index=False)
    print("\nOptimisation summary:")
    print(summary_df.to_string(index=False))
    print(f"\nTrue params: {true_params}")

    # 4. Plot -----------------------------------------------------------
    print("\n[4/4] Plotting ...")
    plot_snapshot_scatter(snaps, sim, dt, OUT_DIR / "snapshot_scatter.png")
    plot_recovery(results, true_params, OUT_DIR / "parameter_recovery.png")
    plot_loss_values(results, OUT_DIR / "loss_values.png")

    print(f"\nDone. All outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
