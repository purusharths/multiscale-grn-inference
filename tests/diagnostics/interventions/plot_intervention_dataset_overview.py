"""
Visual overview of the knockout (intervention) dataset -- the intervention
counterpart of ../plot_dataset_overview.py.

Three panels:
  1. UMAP of all snapshots pooled, coloured by snapshot index. Shows how
     far the intervention displaces the population relative to its own
     spread (pre-intervention snapshots should pile together, post-
     intervention ones separate).
  2. gene_0 vs gene_1 phase plot with the drift field BEFORE the
     intervention (mu = mu0).
  3. Same phase plot with the drift field AFTER the intervention
     (mu = mu(T) ~ 0).

Panels 2 and 3 are the intervention-specific part: under a knockout the
drift field's attractor itself MOVES, which a stationary dataset never
shows. Same cell cloud is drawn under both fields so the mismatch between
"where the cells are" and "where the field pulls them" is visible.

Not a test -- a standalone report script. Re-run anytime; always
overwrites tests/diagnostics/interventions/intervention_dataset_overview.png.

Usage:
    uv run python tests/diagnostics/interventions/plot_intervention_dataset_overview.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

from _intervention_ground_truth import SIM_T, extract_snapshots, make_knockout_sim

N_CELLS = 400
N_GENES = 4
N_SNAPS = 7

sim = make_knockout_sim(num_genes=N_GENES, seed=42)
snaps, times, mu_per_interval, dt = extract_snapshots(sim, N_CELLS, N_SNAPS)

all_X = np.vstack(snaps)
snap_idx = np.repeat(np.arange(N_SNAPS), N_CELLS)

fig, (ax_umap, ax_pre, ax_post) = plt.subplots(1, 3, figsize=(17, 5))

# ---- 1. UMAP ----
embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(all_X)
sc = ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=snap_idx, cmap="viridis", s=8, alpha=0.6)
cb = fig.colorbar(sc, ax=ax_umap, label="snapshot k")
ax_umap.set_title("UMAP (knockout, all snapshots pooled)", fontsize=10)
ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")

# ---- 2 & 3. phase plot + drift field, pre vs post intervention ----
gx = np.linspace(all_X[:, 0].min() - 0.3, all_X[:, 0].max() + 0.3, 14)
gy = np.linspace(all_X[:, 1].min() - 0.3, all_X[:, 1].max() + 0.3, 14)
colors = plt.cm.viridis(np.linspace(0, 1, N_SNAPS))


def drift_field(A: np.ndarray, mu: np.ndarray):
    GX, GY = np.meshgrid(gx, gy)
    DX, DY = np.zeros_like(GX), np.zeros_like(GY)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            c = mu.copy().astype(float)
            c[0], c[1] = GX[i, j], GY[i, j]
            d = A @ (mu - c)
            DX[i, j], DY[i, j] = d[0], d[1]
    return DX, DY


for ax, t_eval, label in [
    (ax_pre, 0.0, "BEFORE intervention  (mu = mu0)"),
    (ax_post, SIM_T, "AFTER intervention  (mu -> 0)"),
]:
    mu_t = sim.mu_t(t_eval, SIM_T)
    DX, DY = drift_field(sim.A, mu_t)
    spd = np.sqrt(DX**2 + DY**2) + 1e-9
    ax.quiver(gx, gy, DX / spd, DY / spd, color="#999999", alpha=0.8, scale=22, width=0.005)
    for snap, c in zip(snaps, colors):
        ax.scatter(snap[:, 0], snap[:, 1], s=5, alpha=0.25, color=c)
    means = np.array([s.mean(axis=0) for s in snaps])
    ax.plot(means[:, 0], means[:, 1], color="black", lw=2, marker="o", ms=5, zorder=5, label="real mean path")
    ax.scatter([mu_t[0]], [mu_t[1]], marker="*", s=220, color="red", edgecolors="black", zorder=6, label="attractor mu")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("gene_0")
    ax.set_ylabel("gene_1")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle(f"Knockout dataset overview  (N={N_CELLS} cells, G={N_GENES} genes, "
             f"{N_SNAPS} snapshots, t_star={SIM_T/2:.1f})")
fig.tight_layout()

out_path = Path(__file__).parent / "intervention_dataset_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
