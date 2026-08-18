"""
Quick visual sanity-check of the dataset used across tests/algorithm/:
the same stationary simulator + perturbed-start snapshots built by
_ground_truth.py (see test_compute_loss.py's make_snapshots(..., shift=...)
usage). One figure, two subplots: UMAP embedding + SDE phase plot
(gene_0 vs gene_1, since the test dataset is 2-gene).

Not a test -- a standalone script. Re-run anytime; always overwrites
tests/diagnostics/dataset_overview.png.

Usage:
    uv run python tests/diagnostics/plot_dataset_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "algorithm"))

import matplotlib.pyplot as plt
import numpy as np
import umap

from _ground_truth import make_snapshots, make_stationary_sim

N_CELLS = 500
DT = 0.3
N_SNAPS = 4

sim = make_stationary_sim(seed=42)
snapshots = make_snapshots(sim, N_CELLS, N_SNAPS, DT, seed=7, shift=-0.6 * sim.mu0)

all_X = np.vstack(snapshots)
snap_idx = np.repeat(np.arange(N_SNAPS), N_CELLS)

fig, (ax_umap, ax_phase) = plt.subplots(1, 2, figsize=(11, 5))

# ---- UMAP ----
embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(all_X)
sc = ax_umap.scatter(embedding[:, 0], embedding[:, 1], c=snap_idx, cmap="viridis", s=8, alpha=0.6)
fig.colorbar(sc, ax=ax_umap, label="snapshot k")
ax_umap.set_title("UMAP")
ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")

# ---- SDE phase plot (gene_0 vs gene_1) ----
colors = plt.cm.viridis(np.linspace(0, 1, N_SNAPS))
for k, (snap, col) in enumerate(zip(snapshots, colors)):
    ax_phase.scatter(snap[:, 0], snap[:, 1], s=6, alpha=0.3, color=col, label=f"t{k}")
means = np.array([s.mean(axis=0) for s in snapshots])
ax_phase.plot(means[:, 0], means[:, 1], color="black", lw=2, marker="o", zorder=5, label="mean")
ax_phase.scatter(*sim.mu0, marker="*", s=200, color="red", edgecolors="black", zorder=6, label="mu0")
ax_phase.set_title("Phase plot (gene_0 vs gene_1)")
ax_phase.set_xlabel("gene_0")
ax_phase.set_ylabel("gene_1")
ax_phase.legend(fontsize=7)

fig.suptitle(f"Test dataset overview  (N={N_CELLS} cells x {N_SNAPS} snapshots, DT={DT})")
fig.tight_layout()

out_path = Path(__file__).parent / "dataset_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
