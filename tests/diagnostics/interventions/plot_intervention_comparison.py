"""
Knockout (intervention) dataset vs. stationary control, and what happens
when the intervention is ignored.

Dataset: datagen's NetworkSimulatorNonStationaryMu under mu_mode="knockout",
i.e. mu(t) = mu0 - mu0*sigmoid(k*(t - t_star)) (mu_options.mu_inverse_sigmoid)
-- cells sit at baseline mu0, then an intervention at t_star drives the
expression target to ~0. The stationary control uses the same simulator,
same seed, same A/mu0/D, with mu_mode="constant".

Three things compared:
  1. The intervention itself: mu(t) profile and the real population means
     tracking it, knockout vs control.
  2. OU vs FP open-loop rollout fidelity on knockout data (W2 to the real
     empirical distribution at each timepoint), same rollout protocol as
     plot_ou_fp_trajectory.py -- start once at real t0, run forward
     unassisted, so error compounds.
  3. Known-{mu_k} vs misspecified-constant-mu. Algorithm 1 takes the
     intervention means as GIVEN ("Data: Known Intervention means {mu_k}",
     paper line 2) and Theta carries one mu per interval. This panel shows
     what that buys: the same rollout run with theta.mu wrongly held at the
     pre-intervention baseline mu0 for every interval.

Uses true (A, sigma) throughout -- this tests the forward models and the
per-interval mu machinery, not an optimizer.

Not a test -- a standalone report script. Re-run anytime; always overwrites
tests/diagnostics/interventions/intervention_comparison.png.

Usage:
    uv run python tests/diagnostics/interventions/plot_intervention_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _intervention_ground_truth import (
    SIM_T,
    effective_sigma,
    extract_snapshots,
    make_constant_sim,
    make_knockout_sim,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_CELLS = 500
N_GENES = 4
N_SNAPS = 7
N_PROJ = 100

ko_sim = make_knockout_sim(num_genes=N_GENES, seed=42)
const_sim = make_constant_sim(num_genes=N_GENES, seed=42)

ko_snaps, times, ko_mu, DT = extract_snapshots(ko_sim, N_CELLS, N_SNAPS)
const_snaps, _, const_mu, _ = extract_snapshots(const_sim, N_CELLS, N_SNAPS)
SIGMA = effective_sigma(ko_sim)

print(f"snapshot times: {np.round(times, 3)}  dt={DT:.3f}  sigma={SIGMA:.3f}")
print(f"mu0            : {ko_sim.mu0.round(3)}")
print(f"mu(t) knockout : {[m.round(2).tolist() for m in ko_mu]}")


# ---------------------------------------------------------------------------
# Rollouts: start once at real t0, run forward unassisted
# ---------------------------------------------------------------------------

def rollout(snaps, mu_seq, seed_ou=11, seed_fp=12):
    theta = Theta(A=ko_sim.A, mu=mu_seq, sigma=SIGMA)
    X0 = snaps[0]

    ou, rng = [X0], np.random.default_rng(seed_ou)
    X = X0
    for k in range(len(snaps) - 1):
        X = ou_gene_expression(X, theta, k, DT, rng=rng)
        ou.append(X)

    fp_rng = np.random.default_rng(seed_fp)
    chi = preprocessing(X0)
    fp = [X0]
    for k in range(len(snaps) - 1):
        chi = fp_cell_population(chi, theta, k, DT, rng=fp_rng)
        fp.append(chi.resample(len(X0), seed=k).T)

    return ou, fp


def w2_to_real(rolled, snaps, seed=99):
    rng = np.random.default_rng(seed)
    return [_sliced_w2(rolled[k], snaps[k], N_PROJ, rng) for k in range(1, len(snaps))]


ko_ou, ko_fp = rollout(ko_snaps, ko_mu)                                  # known {mu_k}
mis_ou, mis_fp = rollout(ko_snaps, [ko_sim.mu0] * (N_SNAPS - 1))         # misspecified constant mu
c_ou, c_fp = rollout(const_snaps, const_mu)                              # stationary control

w2 = {
    "knockout OU (known mu_k)":  w2_to_real(ko_ou, ko_snaps),
    "knockout FP (known mu_k)":  w2_to_real(ko_fp, ko_snaps),
    "knockout OU (const mu)":    w2_to_real(mis_ou, ko_snaps),
    "knockout FP (const mu)":    w2_to_real(mis_fp, ko_snaps),
    "control OU":                w2_to_real(c_ou, const_snaps),
    "control FP":                w2_to_real(c_fp, const_snaps),
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
gene_colors = plt.cm.tab10(np.linspace(0, 1, N_GENES))

# (0,0) mu(t) profile + real means, knockout vs control
ax = axes[0, 0]
t_dense = np.linspace(0, SIM_T, 200)
for g in range(N_GENES):
    ax.plot(t_dense, [ko_sim.mu_t(t, SIM_T)[g] for t in t_dense],
            color=gene_colors[g], lw=1.8, label=f"mu(t) gene_{g}")
    ax.plot(times, [s[:, g].mean() for s in ko_snaps],
            color=gene_colors[g], ls="", marker="o", ms=6)
    ax.plot(times, [s[:, g].mean() for s in const_snaps],
            color=gene_colors[g], ls="--", lw=0.9, alpha=0.5)
ax.axvline(SIM_T / 2, color="black", ls=":", lw=1.2)
ax.text(SIM_T / 2, ax.get_ylim()[1] * 0.97, " t_star", fontsize=8, va="top")
ax.set_xlabel("time")
ax.set_ylabel("expression / mu")
ax.set_title("Knockout mu(t) (lines), real knockout means (dots),\ncontrol means (dashed)", fontsize=10)
ax.legend(fontsize=7, ncol=2)

# (0,1) W2 rollout error -- knockout, known vs misspecified
ax = axes[0, 1]
styles = {
    "knockout OU (known mu_k)": ("#d6604d", "-", "o"),
    "knockout FP (known mu_k)": ("#4393c3", "-", "s"),
    "knockout OU (const mu)":   ("#d6604d", "--", "^"),
    "knockout FP (const mu)":   ("#4393c3", "--", "v"),
}
ks = np.arange(1, N_SNAPS)
for name, (c, ls, m) in styles.items():
    ax.plot(ks, w2[name], color=c, ls=ls, marker=m, label=name)
ax.axvline((N_SNAPS - 1) / 2, color="black", ls=":", lw=1.2)
ax.set_xlabel("snapshot k")
ax.set_ylabel("sliced W2 to real")
ax.set_yscale("log")
ax.set_title("Rollout error on knockout data:\nknown {mu_k} vs. ignoring the intervention", fontsize=10)
ax.legend(fontsize=8)

# (1,0) W2 rollout error -- knockout vs stationary control (both correct mu)
ax = axes[1, 0]
for name, (c, m) in {"knockout OU (known mu_k)": ("#d6604d", "o"),
                     "knockout FP (known mu_k)": ("#4393c3", "s")}.items():
    ax.plot(ks, w2[name], color=c, marker=m, label=name)
for name, (c, m) in {"control OU": ("#d6604d", "^"), "control FP": ("#4393c3", "v")}.items():
    ax.plot(ks, w2[name], color=c, ls="--", alpha=0.6, marker=m, label=name)
ax.set_xlabel("snapshot k")
ax.set_ylabel("sliced W2 to real")
ax.set_title("Knockout vs. stationary control\n(both with correct mu)", fontsize=10)
ax.legend(fontsize=8)

# (1,1) PCA mean trajectories on knockout data
ax = axes[1, 1]
pca = PCA(n_components=2, random_state=42).fit(np.vstack(ko_snaps))
scatter = pca.transform(np.vstack(ko_snaps))
ax.scatter(scatter[:, 0], scatter[:, 1], c=np.repeat(np.arange(N_SNAPS), N_CELLS),
           cmap="Greys", s=4, alpha=0.15)


def pmean(seq):
    return np.array([pca.transform(s).mean(axis=0) for s in seq])


for seq, c, ls, m, lbl in [
    (ko_snaps, "black",   "-",  "o", "real"),
    (ko_ou,    "#d6604d", "--", "^", "OU (known mu_k)"),
    (ko_fp,    "#4393c3", ":",  "s", "FP (known mu_k)"),
    (mis_ou,   "#7b3294", "-.", "x", "OU (const mu)"),
]:
    p = pmean(seq)
    ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, marker=m, lw=1.8, label=lbl)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax.set_title(f"Knockout mean trajectory in PCA space (all {N_GENES} genes)", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle(f"Intervention (knockout, mu(t)=inverse sigmoid) vs stationary control  "
             f"-- N={N_CELLS}, G={N_GENES}, {N_SNAPS} snapshots")
fig.tight_layout()

out_path = Path(__file__).parent / "intervention_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\nMean sliced-W2 to real over all snapshots:")
for name, vals in w2.items():
    print(f"  {name:<28} {np.mean(vals):.4f}")
