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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _intervention_ground_truth import (
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)
from _ours_combinations import (
    COMBINATIONS,
    fit_combination,
    off_diag_indices,
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

# COMBINATIONS is imported from _ours_combinations rather than redefined, so
# this script and every other comparison in the branch sweep the same six.

sim = make_single_gene_knockout_sim(
    num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
    t_star=T_STAR, network_density=NETWORK_DENSITY,
)
snapshots, times, MU_KNOWN, DT = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)

A_TRUE = sim.A
SIGMA_TRUE = effective_sigma(sim)
OFF = off_diag_indices(N_GENES)
true_off = np.array([A_TRUE[i, j] for i, j in OFF])


# ---------------------------------------------------------------------------
# Fitting is delegated to _ours_combinations.fit_combination, which owns the
# theta <-> vector encoding ([log diag(A) (G), off-diagonals (G^2-G), log
# sigma (1)]), the deliberately wrong edge-free start x0 = 1.2*I, and -- the
# reason this script no longer calls scipy.optimize.minimize itself -- the
# custom initial simplex.
#
# scipy's own Nelder-Mead simplex steps each coordinate by x0[k]*1.05, EXCEPT
# where x0[k] == 0, where it substitutes a hardcoded absolute 0.00025
# (_minimize_neldermead's zdelt). Every off-diagonal entry of x0 is exactly 0
# here, so all G^2-G edge directions were born with a simplex edge length of
# 0.00025 -- below this script's own xatol=1e-3, i.e. already "converged" at
# iteration zero. That is why every earlier run of this file returned an A
# whose off-diagonals were still exactly their starting value: the reported
# offdiag_err (1.6451-1.6453 across all six) equalled ||true off-diagonals||
# = 1.6452 to five significant figures.
#
# fit_combination's _initial_simplex gives zero-valued coordinates the same
# absolute step scipy gives the diagonal (0.06 = 1.2 * 0.05). It also carries
# the LinAlgError guard for candidate A's that collapse the propagated cloud
# onto a lower-dimensional subspace and make gaussian_kde singular.
# ---------------------------------------------------------------------------

print(f"fitting {N_GENES**2 + 1} params (full A + sigma), mu fixed at known {{mu_k}}")
print(f"true sigma={SIGMA_TRUE:.3f} | budget={MAXITER} iters/combination")
print(f"optimizer={OPTIMIZER_METHOD} via _ours_combinations.fit_combination "
      f"(custom initial simplex)\n")

rows, fitted = [], {}
for name, terms in COMBINATIONS.items():
    print(f"[{name}] optimising ...", flush=True)
    theta_hat, info = fit_combination(
        terms, snapshots, MU_KNOWN, DT, N_GENES,
        n_proj=N_PROJ, maxiter=MAXITER, method=OPTIMIZER_METHOD, seed=0,
    )
    fitted[name] = theta_hat

    hat_off = np.array([theta_hat.A[i, j] for i, j in OFF])
    A_err = float(np.linalg.norm(theta_hat.A - A_TRUE))
    off_err = float(np.linalg.norm(hat_off - true_off))
    edge_corr = float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0
    sigma_err = abs(theta_hat.sigma - SIGMA_TRUE)

    rows.append({
        "combination": name, "A_err": A_err, "offdiag_err": off_err,
        "edge_corr": edge_corr, "sigma_err": sigma_err,
        "final_objective": info["final_objective"],
        "n_evals": info["n_evals"], "seconds": info["seconds"],
    })
    print(f"  A_err={A_err:.3f}  offdiag_err={off_err:.3f}  edge_corr={edge_corr:+.3f}  "
          f"sigma_err={sigma_err:.3f}  evals={info['n_evals']}  {info['seconds']:.0f}s")

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
