"""
Do the OU and FP forward models RECOVER the single-gene knockout dynamics
-- including the propagation into genes that were never targeted?

plot_single_gene_knockout.py establishes that knocking out gene_0 on a
coupled network produces transient excursions in the untargeted genes.
This script asks whether the forward models reproduce that: OU and FP each
start once from the real t0 snapshot and roll forward through every
interval unassisted (open-loop, no re-anchoring at real data), so error
compounds and any failure to capture propagation shows up.

Three arms, as in ../plot_intervention_comparison.py:
  - known {mu_k}   : the per-interval intervention means Algorithm 1 takes
                     as input (paper line 2) -- here genuinely per-gene,
                     since only gene_0's target moves.
  - const mu       : mu wrongly pinned at pre-knockout baseline (the
                     "intervention ignored" arm).
  - real           : the data itself.

The per-gene panels are the interesting part: genes 1-4 are never
intervened on, so any structure the rollout reproduces there comes purely
from A's off-diagonal coupling, not from the mu sequence.

Uses true (A, sigma) -- this tests the forward models, not an optimizer.

Not a test -- a standalone report script. Re-run anytime; always overwrites
tests/diagnostics/interventions/single-gene-knockout/single_gene_recovery.png.

Usage:
    uv run python tests/diagnostics/interventions/single-gene-knockout/plot_single_gene_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _intervention_ground_truth import (
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)

# ---------------------------------------------------------------------------
# Config (matches plot_single_gene_knockout.py)
# ---------------------------------------------------------------------------

N_GENES = 5
KO_GENE = 0
N_CELLS = 400
N_SNAPS = 15
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.3
SEED = 42
N_PROJ = 100

sim = make_single_gene_knockout_sim(
    num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
    t_star=T_STAR, network_density=NETWORK_DENSITY,
)
snaps, times, mu_known, DT = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)
SIGMA = effective_sigma(sim)
A = sim.A
others = [g for g in range(N_GENES) if g != KO_GENE]
direct = [g for g in others if abs(A[g, KO_GENE]) > 1e-9]


def rollout(mu_seq, seed_ou=11, seed_fp=12):
    theta = Theta(A=A, mu=mu_seq, sigma=SIGMA)
    X0 = snaps[0]

    ou, rng = [X0], np.random.default_rng(seed_ou)
    X = X0
    for k in range(N_SNAPS - 1):
        X = ou_gene_expression(X, theta, k, DT, rng=rng)
        ou.append(X)

    fp_rng = np.random.default_rng(seed_fp)
    chi, fp = preprocessing(X0), [X0]
    for k in range(N_SNAPS - 1):
        chi = fp_cell_population(chi, theta, k, DT, rng=fp_rng)
        fp.append(chi.resample(N_CELLS, seed=k).T)
    return ou, fp


ou_known, fp_known = rollout(mu_known)
ou_const, fp_const = rollout([sim.mu0] * (N_SNAPS - 1))

rng_d = np.random.default_rng(99)
w2 = {
    "OU (known mu_k)": [_sliced_w2(ou_known[k], snaps[k], N_PROJ, rng_d) for k in range(1, N_SNAPS)],
    "FP (known mu_k)": [_sliced_w2(fp_known[k], snaps[k], N_PROJ, rng_d) for k in range(1, N_SNAPS)],
    "OU (const mu)":   [_sliced_w2(ou_const[k], snaps[k], N_PROJ, rng_d) for k in range(1, N_SNAPS)],
    "FP (const mu)":   [_sliced_w2(fp_const[k], snaps[k], N_PROJ, rng_d) for k in range(1, N_SNAPS)],
}

print(f"knockout gene_{KO_GENE} | directly coupled: {[f'gene_{g}' for g in direct]}")
print("\nmean sliced-W2 to real over all snapshots:")
for name, vals in w2.items():
    print(f"  {name:<18} {np.mean(vals):.4f}")

print("\nper-gene rollout error at the propagation peak (|sim - real| mean expression):")
real_m = np.array([s.mean(axis=0) for s in snaps])
ouk_m = np.array([s.mean(axis=0) for s in ou_known])
ouc_m = np.array([s.mean(axis=0) for s in ou_const])
peak_k = int(np.argmax(np.abs(real_m[:, direct[0]] - real_m[0, direct[0]]))) if direct else 1
print(f"  (at t={times[peak_k]:.2f}, peak displacement of gene_{direct[0] if direct else '-'})")
for g in range(N_GENES):
    tag = "KO" if g == KO_GENE else ("direct" if g in direct else "indirect")
    print(f"  gene_{g} ({tag:<8}): OU known={abs(ouk_m[peak_k, g]-real_m[peak_k, g]):.3f}   "
          f"OU const={abs(ouc_m[peak_k, g]-real_m[peak_k, g]):.3f}")

# ---------------------------------------------------------------------------
# Plot: per-gene recovery + W2 + PCA
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(17, 13))
axes = axes.ravel()
t_dense = np.linspace(0, T, 250)

for g in range(N_GENES):
    ax = axes[g]
    real_mean = np.array([s[:, g].mean() for s in snaps])
    real_std = np.array([s[:, g].std() for s in snaps])

    ax.plot(t_dense, [sim.mu_t(t, T)[g] for t in t_dense], color="gray", ls=":", lw=1.4, label="mu(t)")
    ax.fill_between(times, real_mean - real_std, real_mean + real_std, color="black", alpha=0.10)
    ax.plot(times, real_mean, color="black", lw=2.2, marker="o", ms=3.5, label="real")
    ax.plot(times, [s[:, g].mean() for s in ou_known], color="#d6604d", ls="--", marker="^", ms=3.5,
            label="OU (known mu_k)")
    ax.plot(times, [s[:, g].mean() for s in fp_known], color="#4393c3", ls=":", marker="s", ms=3.5,
            label="FP (known mu_k)")
    ax.plot(times, [s[:, g].mean() for s in ou_const], color="#7b3294", ls="-.", marker="x", ms=3.5,
            alpha=0.8, label="OU (const mu)")
    ax.axvline(T_STAR, color="black", ls=":", lw=1.0, alpha=0.5)

    if g == KO_GENE:
        role = "KNOCKED OUT"
    elif g in direct:
        role = f"direct (A[{g},{KO_GENE}]={A[g, KO_GENE]:+.2f})"
    else:
        role = "indirect (via network)"
    ax.set_title(f"gene_{g} -- {role}", fontsize=9)
    ax.set_xlabel("time"); ax.set_ylabel("mean expression")
    if g == 0:
        ax.legend(fontsize=6.5)

# W2 panel
ax = axes[N_GENES]
ks = np.arange(1, N_SNAPS)
for name, (c, ls, m) in {
    "OU (known mu_k)": ("#d6604d", "-", "o"),
    "FP (known mu_k)": ("#4393c3", "-", "s"),
    "OU (const mu)":   ("#d6604d", "--", "^"),
    "FP (const mu)":   ("#4393c3", "--", "v"),
}.items():
    ax.plot(times[1:], w2[name], color=c, ls=ls, marker=m, ms=4, label=name)
ax.axvline(T_STAR, color="black", ls=":", lw=1.2)
ax.set_yscale("log")
ax.set_xlabel("time"); ax.set_ylabel("sliced W2 to real")
ax.set_title("Distributional error (open-loop rollout)", fontsize=9)
ax.legend(fontsize=7)

# PCA panel
ax = axes[N_GENES + 1]
pca = PCA(n_components=2, random_state=42).fit(np.vstack(snaps))
sc = pca.transform(np.vstack(snaps))
ax.scatter(sc[:, 0], sc[:, 1], c=np.repeat(np.arange(N_SNAPS), N_CELLS), cmap="Greys", s=3, alpha=0.12)


def pmean(seq):
    return np.array([pca.transform(s).mean(axis=0) for s in seq])


for seq, c, ls, m, lbl in [
    (snaps,    "black",   "-",  "o", "real"),
    (ou_known, "#d6604d", "--", "^", "OU (known mu_k)"),
    (fp_known, "#4393c3", ":",  "s", "FP (known mu_k)"),
    (ou_const, "#7b3294", "-.", "x", "OU (const mu)"),
]:
    p = pmean(seq)
    ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, marker=m, ms=4, lw=1.8, label=lbl)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax.set_title(f"Mean trajectory in PCA space (all {N_GENES} genes)", fontsize=9)
ax.legend(fontsize=7)

for ax in axes[N_GENES + 2:]:
    ax.set_visible(False)

fig.suptitle(f"OU / FP recovery of single-gene knockout dynamics  "
             f"(gene_{KO_GENE} KO, {N_GENES} genes, density={NETWORK_DENSITY}, "
             f"N={N_CELLS}, {N_SNAPS} snapshots, t_star={T_STAR})")
fig.tight_layout()

out_path = Path(__file__).parent / "single_gene_recovery.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
