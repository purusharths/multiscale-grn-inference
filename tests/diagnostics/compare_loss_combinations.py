"""
Amortized comparison of parameter-recovery quality across all seven
nonempty subsets of {L_OU, L_FP, L_cons}: which loss combination best
recovers the true (A, mu, sigma) under an equal optimization budget --
swept across multiple (n_cells, n_genes) configurations, over several
snapshots.

"Amortized" here means every combination gets the exact same budget (same
optimizer, same maxiter, same n_proj, same seed) starting from the exact
same wrong initial theta, on the exact same dataset per config -- so the
comparison isn't skewed by one combination getting more compute or an
easier start.

Same dataset generator as tests/algorithm/: imports make_stationary_sim/
make_snapshots straight from tests/algorithm/_ground_truth.py (same
seeds), so each config's data is the identical kind of dataset those
pytest specs check properties of, not a separately-generated lookalike.

Optimizer: scipy.optimize.minimize(method=OPTIMIZER_METHOD), default
Nelder-Mead -- a derivative-free simplex method, chosen because loss_ou/
loss_fp/loss_cons are (a) stochastic (sliced-W2's random projections, KDE
resampling) and (b) not differentiable in this numpy implementation (no
autodiff pipeline), so gradient-based optimizers aren't directly available.
Alternatives, roughly by how much they'd change the pipeline:
  - scipy "Powell" / "COBYLA": also derivative-free, zero new dependency,
    swap via OPTIMIZER_METHOD below.
  - CMA-ES (the `cma` package): generally more robust than Nelder-Mead on
    noisy, moderate-dimensional objectives -- Nelder-Mead's simplex is
    known to degenerate as dimensionality grows, which matters directly
    here since theta has 2*n_genes+1 parameters. Not wired up (new dep).
  - Gradient-based (Adam/L-BFGS) over a JAX-differentiable rewrite of the
    loss pipeline (as the deprecated grn_inference_fp.py did via OTT-JAX
    Sinkhorn) -- fastest per-iteration once available, but KDE resampling
    isn't naturally differentiable, so this needs a real rewrite.

Not a test -- a standalone report script; nothing here asserts pass/fail.
Located at tests/diagnostics/ (not tests/algorithm/, so pytest doesn't try
to collect it as a spec; not src/, since it's a one-shot comparison
script, not reusable library code).

Re-run anytime; always overwrites tests/diagnostics/loss_combination_comparison.{csv,png}
and tests/diagnostics/drift_vector_field.png.

Usage:
    uv run python tests/diagnostics/compare_loss_combinations.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "algorithm"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from multsc_grn_inference.compute_loss import loss_cons, loss_fp, loss_ou
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _ground_truth import make_snapshots, make_stationary_sim

# ---------------------------------------------------------------------------
# Config -- change these to sweep different dataset sizes / dimensionality
# ---------------------------------------------------------------------------

CELL_GENE_CONFIGS = [
    (300, 2),
    (500, 3),
    (800, 4),
]  # (n_cells, n_genes)

N_SNAPS = 7
DT = 0.3
N_PROJ = 50
SEED = 0
MAXITER = 300  # same budget for every combination
OPTIMIZER_METHOD = "Nelder-Mead"  # alternatives: "Powell", "COBYLA" (see module docstring)

COMBINATIONS = {
    "OU":            {"ou"},
    "FP":            {"fp"},
    "Cons":          {"cons"},
    "OU+FP":         {"ou", "fp"},
    "OU+Cons":       {"ou", "cons"},
    "FP+Cons":       {"fp", "cons"},
    "OU+FP+Cons":    {"ou", "fp", "cons"},
}


# ---------------------------------------------------------------------------
# theta <-> unconstrained vector, diagonal A only (matches the ground truth's
# diagonal-dominant, zero-off-diagonal structure)
# ---------------------------------------------------------------------------

def encode(A_diag: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    return np.concatenate([np.log(A_diag), mu, [np.log(sigma)]])


def decode(x: np.ndarray, G: int, n_intervals: int) -> Theta:
    A_diag, mu, log_sigma = x[:G], x[G:2 * G], x[-1]
    return Theta(A=np.diag(np.exp(A_diag)), mu=[mu] * n_intervals, sigma=float(np.exp(log_sigma)))


def make_objective(terms: set[str], snapshots, chi, dt, G, n_intervals):
    def objective(x):
        theta = decode(x, G, n_intervals)
        total = 0.0
        if "ou" in terms:
            total += loss_ou(theta, snapshots, chi, dt, n_proj=N_PROJ, seed=SEED)
        if "fp" in terms:
            total += loss_fp(theta, snapshots, chi, dt, n_proj=N_PROJ, seed=SEED)
        if "cons" in terms:
            total += loss_cons(theta, snapshots, chi, dt, n_proj=N_PROJ, seed=SEED)
        return total
    return objective


def drift_field(A: np.ndarray, mu: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drift A(mu - c) on a (gene_0, gene_1) grid, other genes held at mu."""
    GX, GY = np.meshgrid(gx, gy)
    DX, DY = np.zeros_like(GX), np.zeros_like(GY)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            c = mu.copy()
            c[0], c[1] = GX[i, j], GY[i, j]
            d = A @ (mu - c)
            DX[i, j], DY[i, j] = d[0], d[1]
    return DX, DY


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

rows = []
field_panels = []  # (config_label, snapshots, true_A, true_mu, best_theta_hat)

for n_cells, n_genes in CELL_GENE_CONFIGS:
    config_label = f"N={n_cells}, G={n_genes}"
    print(f"\n=== {config_label} ===")

    sim = make_stationary_sim(num_genes=n_genes, seed=42)
    TRUE_A, TRUE_MU, TRUE_SIGMA = sim.A, sim.mu0, 0.15
    G = n_genes
    n_intervals = N_SNAPS - 1

    snapshots = make_snapshots(sim, n_cells, N_SNAPS, DT, seed=7, shift=-0.6 * TRUE_MU)
    chi = [preprocessing(X) for X in snapshots]

    x0 = encode(0.4 * np.diag(TRUE_A), TRUE_MU * 1.4, 0.4)
    theta_hats = {}

    for name, terms in COMBINATIONS.items():
        print(f"[{name}] optimising ...", flush=True)
        t0 = time.perf_counter()
        res = scipy.optimize.minimize(
            make_objective(terms, snapshots, chi, DT, G, n_intervals), x0,
            method=OPTIMIZER_METHOD,
            options={"maxiter": MAXITER, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
        )
        elapsed = time.perf_counter() - t0
        theta_hat = decode(res.x, G, n_intervals)
        theta_hats[name] = theta_hat

        A_err = float(np.linalg.norm(np.diag(theta_hat.A) - np.diag(TRUE_A)))
        mu_err = float(np.linalg.norm(theta_hat.mu[0] - TRUE_MU))
        sigma_err = abs(theta_hat.sigma - TRUE_SIGMA)

        rows.append({
            "n_cells": n_cells,
            "n_genes": n_genes,
            "combination": name,
            "sigma_hat": round(theta_hat.sigma, 4),
            "A_err": A_err,
            "mu_err": mu_err,
            "sigma_err": sigma_err,
            "final_objective": res.fun,
            "n_evals": res.nfev,
            "converged": res.success,
            "seconds": round(elapsed, 2),
        })
        print(f"  A_err={A_err:.3f}  mu_err={mu_err:.3f}  sigma_err={sigma_err:.3f}  "
              f"obj={res.fun:.4f}  evals={res.nfev}  {elapsed:.1f}s")

    field_panels.append((config_label, snapshots, TRUE_A, TRUE_MU, theta_hats["OU+FP+Cons"]))

df = pd.DataFrame(rows)
csv_path = Path(__file__).parent / "loss_combination_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print(df[["n_cells", "n_genes", "combination", "A_err", "mu_err", "sigma_err", "final_objective", "seconds"]]
      .to_string(index=False))

# ---------------------------------------------------------------------------
# Bar chart: recovery error per combination, one row per config
# ---------------------------------------------------------------------------

n_configs = len(CELL_GENE_CONFIGS)
fig, axes = plt.subplots(n_configs, 3, figsize=(14, 3.6 * n_configs), squeeze=False)
metrics = [("A_err", "||A_diag_hat - A_diag_true||"), ("mu_err", "||mu_hat - mu_true||"), ("sigma_err", "|sigma_hat - sigma_true|")]

for row, (n_cells, n_genes) in enumerate(CELL_GENE_CONFIGS):
    sub = df[(df["n_cells"] == n_cells) & (df["n_genes"] == n_genes)]
    for col, (metric, title) in enumerate(metrics):
        ax = axes[row, col]
        ax.bar(sub["combination"], sub[metric], color="#4393c3")
        ax.set_title(title if row == 0 else "", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        if col == 0:
            ax.set_ylabel(f"N={n_cells}, G={n_genes}", fontsize=9)

fig.suptitle(f"Parameter-recovery error by loss combination  (equal budget: {OPTIMIZER_METHOD}, maxiter={MAXITER})")
fig.tight_layout()
png_path = Path(__file__).parent / "loss_combination_comparison.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_path}")

# ---------------------------------------------------------------------------
# Drift vector field: true vs. OU+FP+Cons-recovered, per config
# (gene_0 vs gene_1 slice; other genes held at mu)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(n_configs, 2, figsize=(11, 4.5 * n_configs), squeeze=False)

for row, (config_label, snapshots, true_A, true_mu, theta_hat) in enumerate(field_panels):
    all_X = np.vstack(snapshots)
    gx = np.linspace(all_X[:, 0].min() - 0.3, all_X[:, 0].max() + 0.3, 12)
    gy = np.linspace(all_X[:, 1].min() - 0.3, all_X[:, 1].max() + 0.3, 12)
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

    for col, (A, mu, title) in enumerate([
        (true_A, true_mu, "true drift"),
        (theta_hat.A, theta_hat.mu[0], "OU+FP+Cons recovered drift"),
    ]):
        ax = axes[row, col]
        DX, DY = drift_field(A, mu, gx, gy)
        spd = np.sqrt(DX**2 + DY**2) + 1e-9
        ax.quiver(gx, gy, DX / spd, DY / spd, color="#999999", alpha=0.8, scale=20, width=0.005)
        for snap, c in zip(snapshots, colors):
            ax.scatter(snap[:, 0], snap[:, 1], s=5, alpha=0.25, color=c)
        ax.scatter([mu[0]], [mu[1]], marker="*", s=150, color="red", edgecolors="black", zorder=5)
        ax.set_title(f"{config_label} -- {title}", fontsize=9)
        ax.set_xlabel("gene_0")
        ax.set_ylabel("gene_1")

fig.suptitle("Drift vector field (gene_0 vs gene_1): true dynamics vs. OU+FP+Cons recovery")
fig.tight_layout()
field_path = Path(__file__).parent / "drift_vector_field.png"
fig.savefig(field_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {field_path}")
