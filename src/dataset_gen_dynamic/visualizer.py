"""UMAP, PHATE, and dynamics plots for mRNA simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATE_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]


# ------------------------------------------------------------------ #
# Embeddings                                                           #
# ------------------------------------------------------------------ #

def _umap_coords(expression: pd.DataFrame) -> np.ndarray:
    import umap
    return umap.UMAP(n_components=2, random_state=42).fit_transform(expression.values)


def _phate_coords(expression: pd.DataFrame) -> np.ndarray:
    import phate
    return phate.PHATE(n_components=2, random_state=42, verbose=False).fit_transform(expression.values)


# ------------------------------------------------------------------ #
# Public plot functions                                                #
# ------------------------------------------------------------------ #

def plot_dynamics(
    trajectory: pd.DataFrame,
    n_genes: int = 10,
    output: str | Path = "dynamics.png",
) -> None:
    """Plot OU trajectory for a subset of genes, with state-transition markers."""
    gene_cols = [c for c in trajectory.columns if c.startswith("gene_")][:n_genes]
    times = trajectory["time"].values
    states = trajectory["state"].values

    fig, ax = plt.subplots(figsize=(13, 4))

    for col in gene_cols:
        ax.plot(times, trajectory[col].values, linewidth=1.0, alpha=0.7)

    # Mark state transitions
    prev = states[0]
    for i, s in enumerate(states[1:], 1):
        if s != prev:
            ax.axvline(times[i], color="black", linestyle="--", linewidth=1.2, alpha=0.6)
            ax.text(times[i] + 0.02, ax.get_ylim()[1] * 0.95, s, fontsize=8, va="top")
            prev = s

    ax.set_xlabel("Time")
    ax.set_ylabel("mRNA level")
    ax.set_title(f"OU Dynamics — {n_genes} genes  (dashed lines = state transitions)")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def plot_embeddings(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    output: str | Path = "embeddings.png",
) -> None:
    """2×2 grid: (UMAP | PHATE) × (colored by state | pseudotime)."""
    print("  Computing UMAP …")
    umap_xy = _umap_coords(expression)
    print("  Computing PHATE …")
    phate_xy = _phate_coords(expression)

    unique_states = sorted(metadata["state"].unique())
    state_color = {s: STATE_PALETTE[i % len(STATE_PALETTE)] for i, s in enumerate(unique_states)}
    cell_colors = [state_color[s] for s in metadata["state"]]
    pseudotime = metadata["pseudotime"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("mRNA Expression Dynamics — OU Process", fontsize=15, fontweight="bold")

    _scatter_state(axes[0, 0], umap_xy, cell_colors, unique_states, state_color, "UMAP", "state")
    _scatter_pt(axes[0, 1], umap_xy, pseudotime, "UMAP", fig)
    _scatter_state(axes[1, 0], phate_xy, cell_colors, unique_states, state_color, "PHATE", "state")
    _scatter_pt(axes[1, 1], phate_xy, pseudotime, "PHATE", fig)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _scatter_state(ax, xy, colors, unique_states, state_color, method, label):
    import matplotlib.patches as mpatches
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=4, alpha=0.6, rasterized=True)
    patches = [mpatches.Patch(color=state_color[s], label=s) for s in unique_states]
    ax.legend(handles=patches, markerscale=2, framealpha=0.8, fontsize=9)
    ax.set_title(f"{method} — {label}")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")


def _scatter_pt(ax, xy, pseudotime, method, fig):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pseudotime, cmap="viridis", s=4, alpha=0.6, rasterized=True)
    fig.colorbar(sc, ax=ax, label="pseudotime", shrink=0.85)
    ax.set_title(f"{method} — pseudotime")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
