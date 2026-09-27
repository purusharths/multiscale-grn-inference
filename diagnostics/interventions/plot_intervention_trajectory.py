"""
OU and FP rollout trajectories on knockout data, per gene -- the
intervention counterpart of ../plot_ou_fp_trajectory.py.

Open-loop rollout: OU and FP both start once from the real t0 snapshot and
run forward through every interval unassisted (no re-anchoring at the real
data), so error compounds. Run twice: with the known intervention means
{mu_k} that Algorithm 1 takes as input, and with mu wrongly held at the
pre-intervention baseline.

Panels:
  - One per gene: real mean +/- 1 std over time, overlaid with the OU and
    FP rollout means (known {mu_k}) and the misspecified constant-mu OU
    rollout. mu(t) itself is drawn as a dotted line, so you can see the
    population chasing a moving target.
  - Final panel: PCA mean trajectory across all genes jointly (PCA is
    linear, so the projected mean equals the mean's projection -- a
    faithful whole-dynamics summary, unlike a nonlinear embedding).

Uses true (A, sigma) -- this tests the forward models and the per-interval
mu machinery, not an optimizer.

Not a test -- a standalone report script. Re-run anytime; always
overwrites tests/diagnostics/interventions/intervention_trajectory.png.

Usage:
    uv run python tests/diagnostics/interventions/plot_intervention_trajectory.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _intervention_ground_truth import (
    SIM_T,
    effective_sigma,
    extract_snapshots,
    make_knockout_sim,
)

N_CELLS = 500
N_GENES = 4
N_SNAPS = 7

sim = make_knockout_sim(num_genes=N_GENES, seed=42)
snaps, times, mu_known, DT = extract_snapshots(sim, N_CELLS, N_SNAPS)
SIGMA = effective_sigma(sim)


def rollout(mu_seq, seed_ou=11, seed_fp=12):
    theta = Theta(A=sim.A, mu=mu_seq, sigma=SIGMA)
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
ou_const, _ = rollout([sim.mu0] * (N_SNAPS - 1))

# ---------------------------------------------------------------------------
# Plot: per-gene panels + PCA summary
# ---------------------------------------------------------------------------

n_panels = N_GENES + 1
ncols = min(3, n_panels)
nrows = int(np.ceil(n_panels / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows))
axes = np.atleast_1d(axes).ravel()

t_dense = np.linspace(0, SIM_T, 200)

for g in range(N_GENES):
    ax = axes[g]
    real_mean = np.array([s[:, g].mean() for s in snaps])
    real_std = np.array([s[:, g].std() for s in snaps])

    ax.plot(t_dense, [sim.mu_t(t, SIM_T)[g] for t in t_dense],
            color="gray", ls=":", lw=1.5, label="mu(t)")
    ax.fill_between(times, real_mean - real_std, real_mean + real_std,
                    color="black", alpha=0.12)
    ax.plot(times, real_mean, color="black", lw=2, marker="o", ms=5, label="real")
    ax.plot(times, [s[:, g].mean() for s in ou_known],
            color="#d6604d", ls="--", marker="^", ms=5, label="OU (known mu_k)")
    ax.plot(times, [s[:, g].mean() for s in fp_known],
            color="#4393c3", ls=":", marker="s", ms=5, label="FP (known mu_k)")
    ax.plot(times, [s[:, g].mean() for s in ou_const],
            color="#7b3294", ls="-.", marker="x", ms=5, alpha=0.8, label="OU (const mu)")
    ax.axvline(SIM_T / 2, color="black", ls=":", lw=1.0, alpha=0.5)
    ax.set_title(f"gene_{g}", fontsize=10)
    ax.set_xlabel("time")
    ax.set_ylabel("expression")
    if g == 0:
        ax.legend(fontsize=7)

# PCA summary panel
ax = axes[N_GENES]
pca = PCA(n_components=2, random_state=42).fit(np.vstack(snaps))
scatter = pca.transform(np.vstack(snaps))
ax.scatter(scatter[:, 0], scatter[:, 1],
           c=np.repeat(np.arange(N_SNAPS), N_CELLS), cmap="Greys", s=4, alpha=0.15)


def pmean(seq):
    return np.array([pca.transform(s).mean(axis=0) for s in seq])


for seq, c, ls, m, lbl in [
    (snaps,    "black",   "-",  "o", "real"),
    (ou_known, "#d6604d", "--", "^", "OU (known mu_k)"),
    (fp_known, "#4393c3", ":",  "s", "FP (known mu_k)"),
    (ou_const, "#7b3294", "-.", "x", "OU (const mu)"),
]:
    p = pmean(seq)
    ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, marker=m, lw=1.8, label=lbl)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax.set_title(f"Mean trajectory in PCA space (all {N_GENES} genes)", fontsize=10)
ax.legend(fontsize=7)

for ax in axes[n_panels:]:
    ax.set_visible(False)

fig.suptitle(f"Knockout rollout trajectories: OU + FP vs real  "
             f"(N={N_CELLS}, G={N_GENES}, {N_SNAPS} snapshots, t_star={SIM_T/2:.1f})")
fig.tight_layout()

out_path = Path(__file__).parent / "intervention_trajectory.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
