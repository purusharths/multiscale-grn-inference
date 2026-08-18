"""
Single-gene knockout on a COUPLED network: does the perturbation propagate
to the genes that were never targeted?

Setup: 5 genes, network_density=0.3 (so A has off-diagonal entries),
knockout of gene_0 only at t_star=2.0, total horizon T=7.0. Only gene_0's
target is driven to zero:

    mu(t) = mu0 - delta * sigmoid(k*(t - t_star)),   delta = mu0[0] * e_0

The control is an "uncoupled twin" (../_intervention_ground_truth.py:
make_uncoupled_twin) -- a deep copy with A's off-diagonals zeroed, sharing
mu0, D, diag(A), mu_kwargs AND rng state. So the two runs differ in exactly
one respect, gene-gene coupling, and the difference between them isolates
propagation from every other effect (including the noise realisation).

A structural note this plot makes visible: for this OU model the stationary
point is c* = mu (drift A(mu-c)=0 with A invertible => c=mu). Knocking out
mu_0 therefore does NOT move the other genes' final resting state -- their
response is necessarily TRANSIENT, an excursion while gene_0 is far from
its target, decaying back to baseline afterwards. That is the propagation
signal; a permanent shift in the untargeted genes would indicate something
other than this model.

Not a test -- a standalone report script. Re-run anytime; always overwrites
tests/diagnostics/interventions/single-gene-knockout/single_gene_knockout.png.

Usage:
    uv run python tests/diagnostics/interventions/single-gene-knockout/plot_single_gene_knockout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from _intervention_ground_truth import (
    extract_snapshots,
    make_single_gene_knockout_sim,
    make_uncoupled_twin,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_GENES = 5
KO_GENE = 0
N_CELLS = 400
N_SNAPS = 15
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.3
SEED = 42

coupled = make_single_gene_knockout_sim(
    num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
    t_star=T_STAR, network_density=NETWORK_DENSITY,
)
uncoupled = make_uncoupled_twin(coupled)

co_snaps, times, _, dt = extract_snapshots(coupled, N_CELLS, N_SNAPS, T=T)
un_snaps, _, _, _ = extract_snapshots(uncoupled, N_CELLS, N_SNAPS, T=T)

co_means = np.array([s.mean(axis=0) for s in co_snaps])   # (N_SNAPS, G)
un_means = np.array([s.mean(axis=0) for s in un_snaps])
A = coupled.A
mu0 = coupled.mu0

others = [g for g in range(N_GENES) if g != KO_GENE]
direct = [g for g in others if abs(A[g, KO_GENE]) > 1e-9]
indirect = [g for g in others if g not in direct]

print(f"knockout gene      : gene_{KO_GENE}   (t_star={T_STAR}, T={T}, dt={dt:.3f})")
print(f"A[:, {KO_GENE}]          : {A[:, KO_GENE].round(2)}   <- gene_{KO_GENE}'s outgoing edges")
print(f"directly coupled   : {[f'gene_{g}' for g in direct]}")
print(f"indirect / none    : {[f'gene_{g}' for g in indirect]}")
print(f"\nmu0                : {mu0.round(2)}")
print(f"coupled   final    : {co_means[-1].round(2)}")
print(f"uncoupled final    : {un_means[-1].round(2)}")
print("\npeak |coupled - uncoupled| per gene (the propagation signal):")
peak = np.abs(co_means - un_means).max(axis=0)
for g in range(N_GENES):
    tag = "KNOCKED OUT" if g == KO_GENE else ("direct" if g in direct else "indirect/none")
    print(f"  gene_{g}: {peak[g]:.3f}   ({tag})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, N_GENES))

# (0,0) A heatmap
ax = axes[0, 0]
vmax = np.abs(A).max()
im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
fig.colorbar(im, ax=ax, fraction=0.046)
for i in range(N_GENES):
    for j in range(N_GENES):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(A[i, j]) > 0.6 * vmax else "black")
ax.add_patch(plt.Rectangle((KO_GENE - 0.5, -0.5), 1, N_GENES, fill=False,
                           edgecolor="lime", lw=3))
ax.set_xticks(range(N_GENES)); ax.set_xticklabels([f"g{j}" for j in range(N_GENES)])
ax.set_yticks(range(N_GENES)); ax.set_yticklabels([f"g{i}" for i in range(N_GENES)])
ax.set_xlabel("source gene j")
ax.set_ylabel("target gene i")
ax.set_title(f"A  (drift_i = sum_j A[i,j](mu_j - c_j))\ngreen = column {KO_GENE}: "
             f"gene_{KO_GENE}'s outgoing influence", fontsize=10)

# (0,1) coupled trajectories
ax = axes[0, 1]
for g in range(N_GENES):
    lw, ls = (2.6, "-") if g == KO_GENE else (1.5, "--")
    lbl = f"gene_{g}" + (" (KO)" if g == KO_GENE else (" *" if g in direct else ""))
    ax.plot(times, co_means[:, g], color=colors[g], lw=lw, ls=ls, marker="o", ms=3.5, label=lbl)
    ax.axhline(mu0[g], color=colors[g], lw=0.6, alpha=0.35, ls=":")
ax.axvline(T_STAR, color="black", ls=":", lw=1.2)
ax.set_xlabel("time"); ax.set_ylabel("mean expression")
ax.set_title(f"COUPLED network (density={NETWORK_DENSITY})\n"
             f"dotted horizontals = each gene's baseline mu0;  * = directly coupled to gene_{KO_GENE}",
             fontsize=10)
ax.legend(fontsize=7, ncol=2)

# (1,0) uncoupled control
ax = axes[1, 0]
for g in range(N_GENES):
    lw, ls = (2.6, "-") if g == KO_GENE else (1.5, "--")
    ax.plot(times, un_means[:, g], color=colors[g], lw=lw, ls=ls, marker="o", ms=3.5,
            label=f"gene_{g}" + (" (KO)" if g == KO_GENE else ""))
    ax.axhline(mu0[g], color=colors[g], lw=0.6, alpha=0.35, ls=":")
ax.axvline(T_STAR, color="black", ls=":", lw=1.2)
ax.set_xlabel("time"); ax.set_ylabel("mean expression")
ax.set_title("UNCOUPLED control (A off-diagonals zeroed)\n"
             "non-targeted genes should sit flat on their baselines", fontsize=10)
ax.legend(fontsize=7, ncol=2)

# (1,1) propagation signal: coupled - uncoupled, non-targeted genes only
ax = axes[1, 1]
for g in others:
    style = "-" if g in direct else "--"
    lbl = f"gene_{g}" + (f"  (A[{g},{KO_GENE}]={A[g, KO_GENE]:+.2f})" if g in direct else "  (indirect)")
    ax.plot(times, co_means[:, g] - un_means[:, g], color=colors[g], ls=style,
            lw=1.8, marker="o", ms=3.5, label=lbl)
ax.axhline(0, color="black", lw=1.0)
ax.axvline(T_STAR, color="black", ls=":", lw=1.2)
ax.set_xlabel("time")
ax.set_ylabel("coupled - uncoupled  (mean expression)")
ax.set_title(f"Propagation signal: response of genes that were NEVER knocked out\n"
             f"(identical noise; only difference is coupling)", fontsize=10)
ax.legend(fontsize=7)

fig.suptitle(f"Single-gene knockout (gene_{KO_GENE}) on a {N_GENES}-gene coupled network  "
             f"--  N={N_CELLS} cells, {N_SNAPS} snapshots, t_star={T_STAR}, T={T}")
fig.tight_layout()

out_path = Path(__file__).parent / "single_gene_knockout.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
