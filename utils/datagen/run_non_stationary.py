
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_gen_dynamic.non_stationary_sim import NetworkSimulatorNonStationaryMu
from dataset_gen_dynamic.visualizer import plot_cell_trajectories, plot_clustermap, plot_dynamics, plot_embeddings, plot_expression_matrix, plot_expression_snapshots


def _sim_to_dataframes(
    data: np.ndarray,       #(num_samples, n_timepoints, num_genes)
    time_grid: np.ndarray,
    population_label: str = "pop_0",
    cell_id_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # flatten simulation output to (expression, metadata) DataFrames."""
    num_samples, _, num_genes = data.shape
    gene_cols = [f"gene_{i}" for i in range(num_genes)]

    rows_expr = []
    rows_meta = []

    for s in range(num_samples):
        for t_idx, t in enumerate(time_grid):
            rows_expr.append(data[s, t_idx])
            rows_meta.append({
                "cell_id":    cell_id_offset + s,
                "time":       t,
                "population": population_label,
                "pseudotime": t / time_grid[-1],
            })

    expression = pd.DataFrame(rows_expr, columns=gene_cols)
    metadata   = pd.DataFrame(rows_meta)
    return expression, metadata



def plot_phase(
    data: np.ndarray,       # (num_samples, T, num_genes)  — one population
    time_grid: np.ndarray,
    gene_x: int = 0,
    gene_y: int = 1,
    output: str | Path = "phase_non_stationary.png",
    n_show: int = 80,
) -> None:
    """
    Phase-space plot: gene_x vs gene_y trajectories for n_show cells,
    coloured by time (viridis).  Mean trajectory overlaid in white.
    """
    num_samples = data.shape[0]
    idx = np.random.default_rng(0).choice(num_samples, size=min(n_show, num_samples), replace=False)

    norm = plt.Normalize(time_grid[0], time_grid[-1])
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in idx:
        traj_x = data[i, :, gene_x]
        traj_y = data[i, :, gene_y]
        # colour each segment by time
        for t in range(len(time_grid) - 1):
            c = cmap(norm(time_grid[t]))
            ax.plot(traj_x[t:t+2], traj_y[t:t+2], color=c, linewidth=0.7, alpha=0.4)

    # mean trajectory
    mean_x = data[:, :, gene_x].mean(axis=0)
    mean_y = data[:, :, gene_y].mean(axis=0)
    ax.plot(mean_x, mean_y, color="white", linewidth=2.5, label="mean", zorder=5)
    ax.plot(mean_x, mean_y, color="black", linewidth=1.0, zorder=4)

    # start / end markers on the mean
    ax.scatter([mean_x[0]],  [mean_y[0]],  color="lime",   s=60, zorder=6, label="t=0")
    ax.scatter([mean_x[-1]], [mean_y[-1]], color="red",    s=60, zorder=6, label=f"t={time_grid[-1]:.1f}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="time")

    ax.set_xlabel(f"gene_{gene_x}")
    ax.set_ylabel(f"gene_{gene_y}")
    ax.set_title("Phase plot SDE", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_facecolor("#1a1a2e")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)



def _save_timepoints(
    combined: pd.DataFrame,     # cell_id, time, population, gene_0, gene_1, …
    out_dir: Path,
) -> None:
    for t_val, group in combined.groupby("time"):
        t_label = f"{t_val:.4f}".replace(".", "_")
        (
            group
            .drop(columns=["time"])
            .reset_index(drop=True)
            .to_csv(out_dir / f"expression_t{t_label}.csv", index=False)
        )


POPULATIONS = [
    {"label": "pop_0", "seed": 0, "mu_mode": "sigmoid",   "mu_kwargs": {}},
    {"label": "pop_1", "seed": 1, "mu_mode": "linear",    "mu_kwargs": {}},
    {"label": "pop_2", "seed": 2, "mu_mode": "heaviside", "mu_kwargs": {}},
    {"label": "pop_3", "seed": 3, "mu_mode": "constant",  "mu_kwargs": {}},
]


def main() -> None:
    out_dir = Path("output/")
    all_expr, all_meta = [], []
    cell_offset = 0 #?

    for pop in POPULATIONS:
        print(f"Simulating {pop['label']} (mu_mode={pop['mu_mode']}) ...")
        sim = NetworkSimulatorNonStationaryMu(
            num_genes=6, network_density=0.3,
            seed=pop["seed"],
            mu_mode=pop["mu_mode"],
            mu_kwargs=pop["mu_kwargs"],
        )
        data, time_grid = sim.simulate(T=4.0, num_samples=200, save_every=15, dt=0.005)
        print(f"  data shape: {data.shape}")

        # ground truth GRN per population
        num_genes = sim.num_genes
        gene_names = [f"gene_{i}" for i in range(num_genes)]
        pd.DataFrame(sim.A, index=gene_names, columns=gene_names).to_csv(
            out_dir / f"grn_weighted_{pop['label']}.csv"
        )
        pd.DataFrame((sim.A != 0).astype(int), index=gene_names, columns=gene_names).to_csv(
            out_dir / f"grn_adjacency_{pop['label']}.csv"
        )

        expr, meta = _sim_to_dataframes(data, time_grid, population_label=pop["label"], cell_id_offset=cell_offset)
        all_expr.append(expr)
        all_meta.append(meta)
        cell_offset += data.shape[0]

    expression = pd.concat(all_expr, ignore_index=True)
    metadata   = pd.concat(all_meta, ignore_index=True)
    print(f"Combined: {metadata['cell_id'].nunique()} cells × {expression.shape[1]} genes "
          f"× {metadata['population'].nunique()} populations  ({len(expression)} total observations)")

    # phase plot — pop_0 only (re-simulate; loop left data/sim at pop_3)
    print("Plotting phase plot ...")
    sim0 = NetworkSimulatorNonStationaryMu(num_genes=6, network_density=0.3, seed=0, mu_mode="sigmoid")
    pop0_data, _ = sim0.simulate(T=4.0, num_samples=200, save_every=15, dt=0.005)
    plot_phase(pop0_data, time_grid, gene_x=0, gene_y=1, output=out_dir / "phase_non_stationary.png")

    print("Building DataFrames …")

    print("Saving per-timepoint expression matrices …")
    timepoints_dir = out_dir / "expression_by_timepoint"
    timepoints_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([metadata[["cell_id", "time", "population"]], expression], axis=1)
    _save_timepoints(combined, timepoints_dir)
    print(f"  Saved {combined['time'].nunique()} timepoint files to {timepoints_dir}/")

    print("Plotting individual cell trajectories …")
    plot_cell_trajectories(
        expression, metadata,
        n_cells=6,
        output=out_dir / "cell_trajectories_non_stationary.png",
    )

    print("Plotting dynamics …")
    plot_dynamics(expression, metadata, output=out_dir / "dynamics_non_stationary.png")

    print("Plotting UMAP + PHATE embeddings …")
    plot_embeddings(expression, metadata, output=out_dir / "embeddings_non_stationary.png", include_pca=True)

    print("Plotting gene expression matrix …")
    plot_expression_matrix(
        expression, metadata,
        n_cells=50,
        output=out_dir / "expression_matrix_non_stationary.png",
    )

    print("Plotting expression snapshots …")
    plot_expression_snapshots(
        expression, metadata,
        n_timepoints=4,
        output=out_dir / "expression_snapshots_non_stationary.png",
    )

    # --- Clustermap
    print("Plotting clustermap …")
    plot_clustermap(
        expression, metadata,
        n_cells=100,
        output=out_dir / "clustermap_non_stationary.png",
        explicit_cell_pop=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
