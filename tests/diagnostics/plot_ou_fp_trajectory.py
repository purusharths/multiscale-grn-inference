"""
OU and FP trajectories vs. the real dataset: starting only from the real
t_0 snapshot, roll OU (tracked particles) and FP (density evolution) both
forward through the whole snapshot sequence -- unlike loss_ou/loss_fp,
which always re-anchor at the true previous snapshot each interval, this
is an open-loop rollout, so it shows *compounding* drift, not one-step
error.

Two comparisons against the real empirical distribution at each timepoint:
  1. Sliced-W2 distance (OU-rollout vs real, FP-rollout vs real) over time
     -- a scalar summary that works regardless of gene count.
  2. PCA phase-plot: project everything into a 2D PCA basis fit on the
     pooled real data, and compare mean trajectories there. PCA is linear,
     so "mean of the projection" == "projection of the mean" -- this
     summarizes the trajectory across ALL genes jointly, not just two of
     them (see plot_dataset_overview.py's gene_0-vs-gene_1 phase plot /
     compare_loss_combinations.py's drift field for the two-gene view).

Uses true theta (A, mu, sigma from the ground-truth simulator), not a
fitted one -- this checks whether the OU/FP *mechanisms* reproduce the
real dynamics given correct parameters, independent of any optimizer.

Same dataset generator as tests/algorithm/ (_ground_truth.py, same seeds).

Not a test -- a standalone report script. Located at tests/diagnostics/
alongside the other report scripts. Re-run anytime; always overwrites
tests/diagnostics/ou_fp_trajectory.png.

Usage:
    uv run python tests/diagnostics/plot_ou_fp_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "algorithm"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _ground_truth import make_snapshots, make_stationary_sim

# ---------------------------------------------------------------------------
# Config -- change these to explore other dataset sizes / dimensionality
# ---------------------------------------------------------------------------

N_CELLS = 500
N_GENES = 4
N_SNAPS = 7
DT = 0.3
N_PROJ = 100
SEED = 0

sim = make_stationary_sim(num_genes=N_GENES, seed=42)
theta = Theta(A=sim.A, mu=[sim.mu0] * (N_SNAPS - 1), sigma=0.15)

snapshots = make_snapshots(sim, N_CELLS, N_SNAPS, DT, seed=7, shift=-0.6 * sim.mu0)
X0 = snapshots[0]

# ---------------------------------------------------------------------------
# Rollouts, both starting only from X0 (no re-anchoring at real data)
# ---------------------------------------------------------------------------

ou_rollout = [X0]
rng_ou = np.random.default_rng(11)
X = X0
for k in range(N_SNAPS - 1):
    X = ou_gene_expression(X, theta, k, DT, rng=rng_ou)
    ou_rollout.append(X)

fp_rollout = [preprocessing(X0)]
rng_fp = np.random.default_rng(12)
chi_k = fp_rollout[0]
for k in range(N_SNAPS - 1):
    chi_k = fp_cell_population(chi_k, theta, k, DT, rng=rng_fp)
    fp_rollout.append(chi_k)

fp_rollout_samples = [X0] + [chi.resample(N_CELLS, seed=k).T for k, chi in enumerate(fp_rollout[1:])]

# ---------------------------------------------------------------------------
# 1. Sliced-W2 distance to the real distribution, over time
# ---------------------------------------------------------------------------

rng_dist = np.random.default_rng(99)
w2_ou, w2_fp = [], []
for k in range(1, N_SNAPS):
    w2_ou.append(_sliced_w2(ou_rollout[k], snapshots[k], N_PROJ, rng_dist))
    w2_fp.append(_sliced_w2(fp_rollout_samples[k], snapshots[k], N_PROJ, rng_dist))

# ---------------------------------------------------------------------------
# 2. PCA phase-plot: mean trajectories across ALL genes
# ---------------------------------------------------------------------------

pca = PCA(n_components=2, random_state=42).fit(np.vstack(snapshots))


def proj_mean(X: np.ndarray) -> np.ndarray:
    return pca.transform(X).mean(axis=0)


real_means = np.array([proj_mean(s) for s in snapshots])
ou_means = np.array([proj_mean(s) for s in ou_rollout])
fp_means = np.array([proj_mean(s) for s in fp_rollout_samples])
mu_proj = pca.transform(sim.mu0[None, :])[0]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax_w2, ax_pca) = plt.subplots(1, 2, figsize=(13, 5))

times = np.arange(1, N_SNAPS)
ax_w2.plot(times, w2_ou, marker="o", color="#d6604d", label="OU rollout vs real")
ax_w2.plot(times, w2_fp, marker="s", color="#4393c3", label="FP rollout vs real")
ax_w2.set_xlabel("snapshot k")
ax_w2.set_ylabel("sliced W2 distance to real")
ax_w2.set_title("Distributional distance over time (open-loop rollout)")
ax_w2.legend(fontsize=9)

real_scatter = pca.transform(np.vstack(snapshots))
snap_idx = np.repeat(np.arange(N_SNAPS), N_CELLS)
ax_pca.scatter(real_scatter[:, 0], real_scatter[:, 1], c=snap_idx, cmap="Greys", s=4, alpha=0.15)

ax_pca.plot(real_means[:, 0], real_means[:, 1], color="black", lw=2, marker="o", label="real (mean)")
ax_pca.plot(ou_means[:, 0], ou_means[:, 1], color="#d6604d", lw=1.8, ls="--", marker="^", label="OU rollout (mean)")
ax_pca.plot(fp_means[:, 0], fp_means[:, 1], color="#4393c3", lw=1.8, ls=":", marker="s", label="FP rollout (mean)")
ax_pca.scatter([mu_proj[0]], [mu_proj[1]], marker="*", s=200, color="gold", edgecolors="black", zorder=6, label="mu")
ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax_pca.set_title(f"Mean trajectory in PCA space (all {N_GENES} genes)")
ax_pca.legend(fontsize=8)

fig.suptitle(f"OU + FP open-loop rollout vs. real dataset  (N={N_CELLS} cells, G={N_GENES} genes, {N_SNAPS} snapshots)")
fig.tight_layout()

out_path = Path(__file__).parent / "ou_fp_trajectory.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
print(f"W2(OU, real) per snapshot: {[round(v, 4) for v in w2_ou]}")
print(f"W2(FP, real) per snapshot: {[round(v, 4) for v in w2_fp]}")
