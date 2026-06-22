
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POP_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]


def _umap_coords(X: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(n_components=2, random_state=42).fit_transform(X)


def _phate_coords(X: np.ndarray) -> np.ndarray:
    import phate
    return phate.PHATE(n_components=2, random_state=42, verbose=False).fit_transform(X)


def _pca_coords(X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(X)






def plot_dynamics(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    output: str | Path = "dynamics.png",
) -> None:
    """
    One subplot per gene showing every cell's OU trajectory over time,
    coloured by population.
    """
    gene_cols = [c for c in expression.columns if c.startswith("gene_")]
    n_genes = len(gene_cols)
    unique_pops = sorted(metadata["population"].unique())
    pop_color = {p: POP_PALETTE[i % len(POP_PALETTE)] for i, p in enumerate(unique_pops)}

    fig, axes = plt.subplots(n_genes, 1, figsize=(12, 2.8 * n_genes), sharex=True)
    if n_genes == 1:
        axes = [axes]

    
    df = metadata.copy()
    for g in gene_cols:
        df[g] = expression[g].values

    for ax, gene in zip(axes, gene_cols):
        for pop in unique_pops:
            pop_df = df[df["population"] == pop]
            for cid, cell_df in pop_df.groupby("cell_id"):
                cell_df = cell_df.sort_values("time")
                ax.plot(
                    cell_df["time"].values,
                    cell_df[gene].values,
                    color=pop_color[pop],
                    linewidth=0.6,
                    alpha=0.25,
                )
            # mean pop bold
            mean_traj = pop_df.groupby("time")[gene].mean()
            ax.plot(
                mean_traj.index,
                mean_traj.values,
                color=pop_color[pop],
                linewidth=2.0,
                label=pop,
            )
        ax.set_ylabel(gene)

    axes[0].legend(fontsize=8, framealpha=0.8)
    axes[-1].set_xlabel("Time")
    fig.suptitle("OU Dynamics (per cell trajectories)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def plot_embeddings(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    output: str | Path = "embeddings.png",
    include_pca: bool = False,
) -> None:
    """Grid: (UMAP | PHATE [| PCA]) × (coloured by population | pseudotime)."""
    X = expression.values
    print("  Computing UMAP ...")
    umap_xy = _umap_coords(X)
    print("  Computing PHATE ...")
    phate_xy = _phate_coords(X)
    if include_pca:
        print("  Computing PCA ...")
        pca_xy = _pca_coords(X)

    unique_pops = sorted(metadata["population"].unique())
    pop_color = {p: POP_PALETTE[i % len(POP_PALETTE)] for i, p in enumerate(unique_pops)}
    cell_colors = [pop_color[p] for p in metadata["population"]]
    pseudotime = metadata["pseudotime"].values

    n_rows = 3 if include_pca else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    fig.suptitle("mRNA Expression Dynamics", fontsize=14, fontweight="bold")

    _scatter_pop(axes[0, 0], umap_xy,  cell_colors, unique_pops, pop_color, "UMAP")
    _scatter_pt (axes[0, 1], umap_xy,  pseudotime,  "UMAP",  fig)
    _scatter_pop(axes[1, 0], phate_xy, cell_colors, unique_pops, pop_color, "PHATE")
    _scatter_pt (axes[1, 1], phate_xy, pseudotime,  "PHATE", fig)
    if include_pca:
        _scatter_pop(axes[2, 0], pca_xy, cell_colors, unique_pops, pop_color, "PCA")
        _scatter_pt (axes[2, 1], pca_xy, pseudotime,  "PCA",  fig)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def plot_cell_trajectories(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    n_cells: int = 6,
    output: str | Path = "cell_trajectories.png",
    seed: int = 0,
) -> None:
    """
    One subplot per cell; each subplot shows all genes as separate lines over time.
    Y-axis: mRNA level. X-axis: time.
    """
    gene_cols = [c for c in expression.columns if c.startswith("gene_")]
    n_genes = len(gene_cols)

    df = metadata.copy()
    for g in gene_cols:
        df[g] = expression[g].values

    rng = np.random.default_rng(seed)
    all_cells = df["cell_id"].unique()
    chosen = sorted(rng.choice(all_cells, size=min(n_cells, len(all_cells)), replace=False))

    gene_cmap = plt.cm.get_cmap("tab10", n_genes)
    gene_color = {g: gene_cmap(i) for i, g in enumerate(gene_cols)}

    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, cid in zip(axes_flat, chosen):
        cell_df = df[df["cell_id"] == cid].sort_values("time")
        for gene in gene_cols:
            ax.plot(
                cell_df["time"].values,
                cell_df[gene].values,
                color=gene_color[gene],
                linewidth=1.4,
                label=gene,
            )
        ax.set_title(f"cell {cid}", fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("mRNA level")
        ax.grid(True, linewidth=0.4, alpha=0.4)

    # hide unused axes
    for ax in axes_flat[len(chosen):]:
        ax.set_visible(False)

    # shared legend from first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Gene", fontsize=9, loc="lower right", framealpha=0.85)

    fig.suptitle("mRNA expression (in cells)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def plot_expression_matrix(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    n_cells: int = 50,
    output: str | Path = "expression_matrix.png",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Sample one timepoint per cell (mimicking a cross-sectional scRNA-seq snapshot),
    then plot the resulting (cells × genes) heatmap.

    Sampling: for each cell, draw one timepoint uniformly at random — the mRNA
    level at that snapshot is the observed expression.

    Returns the sampled expression DataFrame (cells × genes).
    """
    gene_cols = [c for c in expression.columns if c.startswith("gene_")]

    df = metadata.copy()
    for g in gene_cols:
        df[g] = expression[g].values

    rng = np.random.default_rng(seed)
    all_cells = df["cell_id"].unique()
    chosen_cells = rng.choice(all_cells, size=min(n_cells, len(all_cells)), replace=False)

    rows = []
    for cid in chosen_cells:
        cell_df = df[df["cell_id"] == cid]
        snapshot = cell_df.sample(n=1, random_state=int(rng.integers(1e6)))
        row = {"cell_id": cid, "time": snapshot["time"].values[0]}
        for g in gene_cols:
            row[g] = snapshot[g].values[0]
        rows.append(row)

    sampled = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    matrix = sampled[gene_cols].values          # (n_cells, n_genes)
    cell_labels = [f"cell_{int(r['cell_id'])}" for _, r in sampled.iterrows()]
    time_labels  = [f"t={r['time']:.2f}" for _, r in sampled.iterrows()]
    ytick_labels = [f"{c}  {t}" for c, t in zip(cell_labels, time_labels)]

    fig, ax = plt.subplots(figsize=(max(6, len(gene_cols) * 0.9), max(6, n_cells * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="mRNA level", shrink=0.6)

    ax.set_xticks(range(len(gene_cols)))
    ax.set_xticklabels(gene_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cell_labels)))
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cell  (sorted by sampled timepoint)")
    ax.set_title("Gene expression matrix", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)

    return sampled[["cell_id", "time"] + gene_cols]





def plot_clustermap( # from scanpy docs
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    n_cells: int = 100,
    output: str | Path = "clustermap.png",
    seed: int = 0,
    explicit_cell_pop: bool = False,
) -> None:
    """
    Hierarchically clustered heatmap (cells × genes).
    Each row = one cell, expression averaged across all timepoints so clustering
    reflects cell identity, not which timepoint was sampled.
    If explicit_cell_pop=True, rows are colour-coded and labelled by population.
    """
    import seaborn as sns

    gene_cols = [c for c in expression.columns if c.startswith("gene_")]

    df = metadata.copy()
    for g in gene_cols:
        df[g] = expression[g].values

    rng = np.random.default_rng(seed)
    chosen = rng.choice(df["cell_id"].unique(), size=min(n_cells, df["cell_id"].nunique()), replace=False)
    df = df[df["cell_id"].isin(chosen)]

    # average over all timepoints → one row per cell
    cell_mean = df.groupby("cell_id")[gene_cols].mean()

    row_colors = None
    if explicit_cell_pop:
        pop_map = df.groupby("cell_id")["population"].first()
        unique_pops = sorted(pop_map.unique())
        pop_color = {p: POP_PALETTE[i % len(POP_PALETTE)] for i, p in enumerate(unique_pops)}
        row_colors = pop_map.loc[cell_mean.index].map(pop_color)

    g = sns.clustermap(
        cell_mean,
        row_colors=row_colors,
        cmap="YlOrRd",
        standard_scale=1,       # z-score per gene (column)
        figsize=(max(6, len(gene_cols) * 0.9), max(7, n_cells * 0.12)),
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "z-score"},
    )

    if explicit_cell_pop:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=pop_color[p], label=p) for p in unique_pops]
        g.fig.legend(handles=patches, title="population", loc="lower left", fontsize=9, framealpha=0.8)

    g.fig.suptitle("Clustermap (clustering by mean expression profile)", fontsize=12, fontweight="bold", y=1.01)
    g.fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(g.fig)


def plot_expression_snapshots(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    n_timepoints: int = 4,
    output: str | Path = "expression_snapshots.png",
) -> None:
    """
    4 subplotst.
    Each subplot is a (cells × genes) heatmap — all cells at that exact time.
    Timepoints are chosen evenly across the time grid.
    """
    gene_cols = [c for c in expression.columns if c.startswith("gene_")]

    df = metadata.copy()
    for g in gene_cols:
        df[g] = expression[g].values

    all_times = np.sort(df["time"].unique())
    indices = np.linspace(0, len(all_times) - 1, n_timepoints, dtype=int)
    chosen_times = all_times[indices]

    # shared colour scale across all subplots
    vmin = expression[gene_cols].values.min()
    vmax = expression[gene_cols].values.max()

    fig, axes = plt.subplots(1, n_timepoints, figsize=(4.5 * n_timepoints, 6), sharey=False)

    for ax, t in zip(axes, chosen_times):
        snap = df[np.isclose(df["time"], t)].sort_values("cell_id")
        matrix = snap[gene_cols].values          # (n_cells, n_genes)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {t:.2f}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(gene_cols)))
        ax.set_xticklabels(gene_cols, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Gene")
        ax.set_ylabel("Cell index" if ax is axes[0] else "")
        if ax is not axes[0]:
            ax.set_yticks([])

    fig.colorbar(im, ax=axes[-1], label="mRNA level", shrink=0.8)
    fig.suptitle("Gene expression snapshots (cells)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def plot_dynamics_snapshot(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    output: str | Path = "dynamics_snapshot.png",
    n_scatter: int = 200,
    seed: int = 0,
) -> None:
    """
    Dynamics plot for cross-sectional / destructive-measurement data where each
    cell has exactly one timepoint.

    Per gene, per population:
      - Scatter a random subset of individual cells (shows stochastic spread).
      - Overlay mean ± 1 std ribbon connecting collection timepoints.
    """
    gene_cols = [c for c in expression.columns if c.startswith("gene_")]
    n_genes = len(gene_cols)
    unique_pops = sorted(metadata["population"].unique())
    n_pops = len(unique_pops)

    # extend palette if needed
    palette = POP_PALETTE * (n_pops // len(POP_PALETTE) + 1)
    pop_color = {p: palette[i] for i, p in enumerate(unique_pops)}

    df = metadata.copy()
    for g in gene_cols:
        df[g] = expression[g].values

    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(n_genes, 1, figsize=(12, 2.8 * n_genes), sharex=True)
    if n_genes == 1:
        axes = [axes]

    for ax, gene in zip(axes, gene_cols):
        for pop in unique_pops:
            pop_df = df[df["population"] == pop]
            color = pop_color[pop]

            # --- scatter a random subset of individual cells ---
            n_plot = min(n_scatter, len(pop_df))
            sample_idx = rng.choice(len(pop_df), size=n_plot, replace=False)
            sub = pop_df.iloc[sample_idx]
            ax.scatter(
                sub["time"].values,
                sub[gene].values,
                color=color,
                s=6,
                alpha=0.25,
                linewidths=0,
            )

            # --- mean ± std ribbon across collection timepoints ---
            stats = pop_df.groupby("time")[gene].agg(["mean", "std"]).reset_index()
            ax.plot(stats["time"], stats["mean"], color=color, linewidth=2.0, label=pop)
            ax.fill_between(
                stats["time"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=color,
                alpha=0.15,
            )

        ax.set_ylabel(gene)

    axes[0].legend(fontsize=8, framealpha=0.8)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Population dynamics — cross-sectional snapshots\n(scatter = individual cells, ribbon = mean ± 1 std)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def _scatter_pop(ax, xy, colors, unique_pops, pop_color, method):
    import matplotlib.patches as mpatches
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=3, alpha=0.4, rasterized=True)
    patches = [mpatches.Patch(color=pop_color[p], label=p) for p in unique_pops]
    ax.legend(handles=patches, fontsize=9, framealpha=0.8)
    ax.set_title(f"{method} — population")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")


def _scatter_pt(ax, xy, pseudotime, method, fig):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pseudotime, cmap="viridis", s=3, alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=ax, label="pseudotime", shrink=0.85)
    ax.set_title(f"{method} — pseudotime")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
