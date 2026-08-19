"""
Which loss combination best recovers the GRN from single-gene knockout data?

Dataset: the coupled 5-gene single-gene-knockout scenario from
../plot_single_gene_knockout.py (gene_0 knocked out at t_star=2.0, T=7.0,
network_density=0.3 so A has real off-diagonal edges).

What is fitted: the FULL interaction matrix A (diagonal via exp() to keep
it positive, all 20 off-diagonals free) plus sigma -- 26 parameters. The
intervention means {mu_k} are held FIXED at their true values, because
Algorithm 1 takes them as given input ("Data: Known Intervention means",
paper line 2). So this measures exactly the thing GRN inference is for:
recovering who regulates whom, given known perturbations.

Combinations compared (as requested):
    OU, FP, OU+FP, OU+Cons, FP+Cons, OU+FP+Cons
Cons-alone is deliberately excluded -- earlier sweeps
(../../../compare_loss_combinations.py) established it is degenerate on its
own: it only checks that the OU and FP forward models agree with EACH
OTHER, never with the data, so it is trivially minimised by wrong dynamics.

Amortized: every combination gets the same optimizer, the same budget, and
the same (deliberately wrong, edge-free) starting point A = 1.2*I.

Metrics reported:
  A_err          -- ||A_hat - A_true||_F over the whole matrix
  offdiag_err    -- ||.||_F restricted to off-diagonals (the GRN edges)
  edge_corr      -- Pearson r between true and recovered off-diagonals;
                    the "did we get the network topology right" number
  sigma_err      -- |sigma_hat - sigma_true|

Not a test -- a standalone report script. Re-run anytime; always overwrites
this folder's loss_comparison_knockout.{csv,png}.

Usage:
    uv run python "tests/diagnostics/interventions/single-gene-knockout/loss-combinations/compare_losses_single_gene_knockout.py"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from multsc_grn_inference.compute_loss import loss_cons, loss_fp, loss_ou
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _intervention_ground_truth import (
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_GENES = 8
KO_GENE = 3
N_CELLS = 3000
N_SNAPS = 15
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.5
SEED = 42

N_PROJ = 30
MAXITER = 2000            # same budget for every combination
OPTIMIZER_METHOD = "Nelder-Mead"   # alternatives: "Powell", "COBYLA"

COMBINATIONS = {
    "OU":          {"ou"},
    "FP":          {"fp"},
    "OU+FP":       {"ou", "fp"},
    "OU+Cons":     {"ou", "cons"},
    "FP+Cons":     {"fp", "cons"},
    "OU+FP+Cons":  {"ou", "fp", "cons"},
}

sim = make_single_gene_knockout_sim(
    num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
    t_star=T_STAR, network_density=NETWORK_DENSITY,
)
snapshots, times, MU_KNOWN, DT = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)
chi = [preprocessing(X) for X in snapshots]

A_TRUE = sim.A
SIGMA_TRUE = effective_sigma(sim)
OFF = [(i, j) for i in range(N_GENES) for j in range(N_GENES) if i != j]
true_off = np.array([A_TRUE[i, j] for i, j in OFF])


# ---------------------------------------------------------------------------
# theta <-> vector:  [log diag(A) (G), off-diagonals (G^2-G), log sigma (1)]
# ---------------------------------------------------------------------------

def encode(A: np.ndarray, sigma: float) -> np.ndarray:
    return np.concatenate([np.log(np.diag(A)), [A[i, j] for i, j in OFF], [np.log(sigma)]])


def decode(x: np.ndarray) -> Theta:
    A = np.diag(np.exp(x[:N_GENES]))
    for n, (i, j) in enumerate(OFF):
        A[i, j] = x[N_GENES + n]
    return Theta(A=A, mu=MU_KNOWN, sigma=float(np.exp(x[-1])))


def make_objective(terms: set[str]):
    def objective(x):
        theta = decode(x)
        total = 0.0
        if "ou" in terms:
            total += loss_ou(theta, snapshots, chi, DT, n_proj=N_PROJ, seed=0)
        if "fp" in terms:
            total += loss_fp(theta, snapshots, chi, DT, n_proj=N_PROJ, seed=0)
        if "cons" in terms:
            total += loss_cons(theta, snapshots, chi, DT, n_proj=N_PROJ, seed=0)
        return total
    return objective


# Deliberately wrong, edge-free start: no knowledge of the network
x0 = encode(np.eye(N_GENES) * 1.2, 0.5)

print(f"fitting {len(x0)} params (full A + sigma), mu fixed at known {{mu_k}}")
print(f"true sigma={SIGMA_TRUE:.3f} | budget={MAXITER} iters/combination\n")

rows, fitted = [], {}
for name, terms in COMBINATIONS.items():
    print(f"[{name}] optimising ...", flush=True)
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(
        make_objective(terms), x0, method=OPTIMIZER_METHOD,
        options={"maxiter": MAXITER, "xatol": 1e-3, "fatol": 1e-6, "adaptive": True},
    )
    elapsed = time.perf_counter() - t0
    theta_hat = decode(res.x)
    fitted[name] = theta_hat

    hat_off = np.array([theta_hat.A[i, j] for i, j in OFF])
    A_err = float(np.linalg.norm(theta_hat.A - A_TRUE))
    off_err = float(np.linalg.norm(hat_off - true_off))
    edge_corr = float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0
    sigma_err = abs(theta_hat.sigma - SIGMA_TRUE)

    rows.append({
        "combination": name, "A_err": A_err, "offdiag_err": off_err,
        "edge_corr": edge_corr, "sigma_err": sigma_err,
        "final_objective": res.fun, "n_evals": res.nfev, "seconds": round(elapsed, 1),
    })
    print(f"  A_err={A_err:.3f}  offdiag_err={off_err:.3f}  edge_corr={edge_corr:+.3f}  "
          f"sigma_err={sigma_err:.3f}  evals={res.nfev}  {elapsed:.0f}s")

df = pd.DataFrame(rows)
csv_path = Path(__file__).parent / "loss_comparison_knockout.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot: metric bars (top) + recovered A heatmaps (below)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 4, figsize=(18, 12))

metrics = [
    ("A_err", "||A_hat - A_true||_F   (lower better)"),
    ("offdiag_err", "off-diagonal error   (lower better)"),
    ("edge_corr", "edge correlation r   (HIGHER better)"),
    ("sigma_err", "|sigma_hat - sigma_true|   (lower better)"),
]
for ax, (col, title) in zip(axes[0], metrics):
    best = df[col].idxmax() if col == "edge_corr" else df[col].idxmin()
    colors = ["#1a9641" if i == best else "#4393c3" for i in df.index]
    ax.bar(df["combination"], df[col], color=colors)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.axhline(0, color="black", lw=0.8)

vmax = np.abs(A_TRUE).max() * 1.2
panels = [("TRUE A", A_TRUE)] + [(n, fitted[n].A) for n in COMBINATIONS]
for ax, (name, mat) in zip(axes[1:].ravel(), panels):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(N_GENES):
        for j in range(N_GENES):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(mat[i, j]) > 0.6 * vmax else "black")
    ax.set_xticks(range(N_GENES)); ax.set_xticklabels([f"g{j}" for j in range(N_GENES)], fontsize=6)
    ax.set_yticks(range(N_GENES)); ax.set_yticklabels([f"g{i}" for i in range(N_GENES)], fontsize=6)
    ax.set_title(name, fontsize=9, fontweight="bold" if name == "TRUE A" else "normal")
for ax in axes[1:].ravel()[len(panels):]:
    ax.set_visible(False)

fig.suptitle(f"Loss-combination comparison on single-gene knockout data  "
             f"(gene_{KO_GENE} KO, {N_GENES} genes, density={NETWORK_DENSITY}, "
             f"fitting full A + sigma, mu known;  {OPTIMIZER_METHOD}, maxiter={MAXITER})")
fig.tight_layout()

png_path = Path(__file__).parent / "loss_comparison_knockout.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_path}")
