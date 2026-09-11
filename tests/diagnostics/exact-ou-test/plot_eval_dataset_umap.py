"""
UMAP overview of the exact dataset used across every comparison in this
investigation (jko-testing, sch-bridge-test, cma-es-test, exact-ou-test):
single-gene knockout, 8 genes, network_density=0.5, knockout gene 3,
t_star=2.0, T=7.0, 10 snapshots, 1500 cells/snapshot, seed=42.

One panel, all snapshots pooled and embedded together, coloured by
snapshot index -- shows the population drifting away from its pre-
knockout cloud after the intervention at t_star, then re-settling.

Not a test -- a standalone report script; re-run anytime, overwrites
this folder's eval_dataset_umap.png.

Usage:
    uv run python tests/diagnostics/exact-ou-test/plot_eval_dataset_umap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from _intervention_ground_truth import (  # noqa: E402
    extract_snapshots,
    make_single_gene_knockout_sim,
)

N_GENES = 8
KO_GENE = 3
N_CELLS = 1500
N_SNAPS = 10
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.5
SEED = 42

HERE = Path(__file__).parent


def main() -> None:
    sim = make_single_gene_knockout_sim(
        num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
        t_star=T_STAR, network_density=NETWORK_DENSITY,
    )
    snapshots, times, _, _ = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)

    all_X = np.vstack(snapshots)
    snap_idx = np.repeat(np.arange(N_SNAPS), N_CELLS)

    embedding = umap.UMAP(n_components=2, random_state=SEED).fit_transform(all_X)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=snap_idx, cmap="viridis", s=6, alpha=0.55)
    cb = fig.colorbar(sc, ax=ax, label="snapshot k  (t_star at k="
                       f"{int(np.searchsorted(times, T_STAR))})")
    ax.set_title(
        f"UMAP: 8-gene single-knockout dataset, all {N_SNAPS} snapshots pooled\n"
        f"(gene {KO_GENE} knocked out at t*={T_STAR}, density={NETWORK_DENSITY}, "
        f"N={N_CELLS} cells/snapshot)",
        fontsize=10,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()

    png_path = HERE / "eval_dataset_umap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
