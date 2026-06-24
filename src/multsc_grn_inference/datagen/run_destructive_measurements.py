"""
Destructive-measurement version of the gene-level perturbation simulation.

Each cell is observed at exactly one timepoint (mimicking scRNA-seq, where
sequencing lyses the cell). The output contains one row per cell, not one row
per (cell, timepoint) pair.

Population layout and simulator are identical to run_gene_perturb_major.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multsc_grn_inference.housekeeping.enforce_diagonal_dominance import enforce_diagonal_dominance
from multsc_grn_inference.housekeeping.mu_options import (
    mu_constant,
    mu_heaviside,
    mu_linear,
    mu_sigmoid,
)
from multsc_grn_inference.visualizer import (
    plot_clustermap,
    plot_dynamics,
    plot_embeddings,
    plot_expression_matrix,
    plot_expression_snapshots,
)


# ---------------------------------------------------------------------------
# tanh mu function
# ---------------------------------------------------------------------------

def mu_tanh(
    mu0: np.ndarray,
    t: float,
    T: float,
    delta: np.ndarray | None = None,
    t_star: float | None = None,
    k: float = 5.0,
    **kwargs,
) -> np.ndarray:
    if delta is None:
        delta = np.ones_like(mu0)
    if t_star is None:
        t_star = T / 2.0
    return mu0 + delta * np.tanh(k * (t - t_star))


#  from run_gene_perturb_major.py
class NetworkSimulatorPerGeneMu:
    def __init__(
        self,
        num_genes: int = 4,
        network_density: float = 0.3,
        seed: int = 0,
        gene_mu_modes: list[str] | None = None,
        gene_mu_kwargs: list[dict] | None = None,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.num_genes = num_genes
        self.gene_mu_modes = gene_mu_modes or ["constant"] * num_genes
        self.gene_mu_kwargs = gene_mu_kwargs or [{} for _ in range(num_genes)]

        assert len(self.gene_mu_modes) == num_genes
        assert len(self.gene_mu_kwargs) == num_genes

        self.A = np.zeros((num_genes, num_genes))
        for i in range(num_genes):
            self.A[i, i] = rng.uniform(1.0, 1.8)

        n_edges = int(network_density * num_genes * (num_genes - 1))
        edges_added = 0
        while edges_added < n_edges:
            i, j = rng.choice(num_genes, 2, replace=False)
            if self.A[i, j] == 0:
                sign = 1 if rng.random() > 0.3 else -1
                self.A[i, j] = sign * rng.uniform(0.3, 1.2)
                edges_added += 1

        enforce_diagonal_dominance(self.A)

        self.mu0 = rng.uniform(2.5, 3.5, num_genes)
        self.D = np.diag(rng.uniform(0.1, 0.2, num_genes))

        self.mu_drift_dir = rng.normal(size=num_genes)
        self.mu_drift_amp = 4.0

    def mu_t(self, t: float, T: float) -> np.ndarray:
        mu = np.empty(self.num_genes)
        for i, (mode, kw) in enumerate(zip(self.gene_mu_modes, self.gene_mu_kwargs)):
            mu0i = self.mu0[i : i + 1]
            diri = self.mu_drift_dir[i : i + 1]

            if mode == "constant":
                mu[i] = mu_constant(mu0i)[0]
            elif mode == "linear":
                mu[i] = mu_linear(mu0i, t, T, diri, self.mu_drift_amp)[0]
            elif mode == "sigmoid":
                mu[i] = mu_sigmoid(mu0i, t, T, **kw)[0]
            elif mode in ("heaviside", "piecewise"):
                mu[i] = mu_heaviside(mu0i, t, T, **kw)[0]
            elif mode == "tanh":
                mu[i] = mu_tanh(mu0i, t, T, **kw)[0]
            else:
                raise ValueError(f"Unknown mu_mode '{mode}' for gene {i}.")
        return mu

    def simulate(
        self,
        T: float = 4.0,
        num_samples: int = 100,
        save_every: int = 15,
        dt: float = 0.005,
    ) -> tuple[np.ndarray, np.ndarray]:
        steps = int(T / dt)
        save_steps = steps // save_every

        data = np.zeros((num_samples, save_steps + 1, self.num_genes))
        time_grid = np.linspace(0.0, T, save_steps + 1)

        for s in range(num_samples):
            mu_init = self.mu_t(0.0, T)
            X = mu_init + 0.5 * self.rng.normal(size=self.num_genes)
            X = np.maximum(X, 0.1)
            data[s, 0] = X.copy()

            save_idx = 1
            for step in range(1, steps):
                t = step * dt
                mu = self.mu_t(t, T)
                drift = self.A @ (mu - X)
                noise = np.sqrt(np.diag(self.D) * dt) * self.rng.normal(size=self.num_genes)
                X += drift * dt + noise
                X = np.maximum(X, 0.05)

                if step % save_every == 0 and save_idx < save_steps + 1:
                    data[s, save_idx] = X.copy()
                    save_idx += 1

        return data, time_grid


# Population definitions 


NUM_GENES = 4

POPULATIONS = [
    {
        "label":          "pop_0",
        "seed":           10,
        "num_samples":    40000,
        "gene_mu_modes":  ["constant",  "constant",  "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_1",
        "seed":           11,
        "num_samples":    10000,
        "gene_mu_modes":  ["sigmoid",   "constant",  "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_2",
        "seed":           12,
        "num_samples":    10000,
        "gene_mu_modes":  ["constant",  "heaviside", "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_3",
        "seed":           13,
        "num_samples":    10000,
        "gene_mu_modes":  ["constant",  "constant",  "linear",   "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_4",
        "seed":           14,
        "num_samples":    10000,
        "gene_mu_modes":  ["constant",  "constant",  "constant", "tanh"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
]



def _sim_to_dataframes_destructive(
    data: np.ndarray,
    time_grid: np.ndarray,
    population_label: str = "pop_0",
    cell_id_offset: int = 0,
    n_per_timepoint: int = 5,
    collection_times: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Destructive measurement model: at each collection timepoint, n_per_timepoint
    cells are sampled from the population, their mRNA is recorded, and they are
    destroyed (never observed again). Each simulated trajectory is used at most once.

    collection_times: explicit experiment collection timepoints (e.g. [0, 1, 3, 7]).
        Each value is snapped to the nearest point in time_grid.
        Defaults to all of time_grid (original behaviour).

    Output rows = n_per_timepoint × len(collection_times).
    Requires num_samples >= n_per_timepoint × len(collection_times).
    """
    num_samples, _, num_genes = data.shape
    gene_cols = [f"gene_{i}" for i in range(num_genes)]
    rng = rng or np.random.default_rng(42)

    # Resolve collection times → nearest indices in the dense time_grid
    if collection_times is None:
        collect_indices = list(range(len(time_grid)))
    else:
        collect_indices = [int(np.argmin(np.abs(time_grid - t))) for t in collection_times]

    n_collect = len(collect_indices)
    n_needed = n_per_timepoint * n_collect
    assert num_samples >= n_needed, (
        f"num_samples ({num_samples}) must be >= "
        f"n_per_timepoint × n_collection_times ({n_per_timepoint} × {n_collect} = {n_needed})"
    )

    # Draw distinct cell indices upfront — no trajectory is reused.
    cell_pool = rng.choice(num_samples, size=n_needed, replace=False)

    rows_expr, rows_meta = [], []
    obs_id = 0
    for pool_pos, grid_idx in enumerate(collect_indices):
        t = time_grid[grid_idx]
        for k in range(n_per_timepoint):
            cell_idx = cell_pool[pool_pos * n_per_timepoint + k]
            rows_expr.append(data[cell_idx, grid_idx])
            rows_meta.append({
                "cell_id":    cell_id_offset + obs_id,
                "time":       t,
                "population": population_label,
                "pseudotime": t / time_grid[-1],
            })
            obs_id += 1

    return pd.DataFrame(rows_expr, columns=gene_cols), pd.DataFrame(rows_meta)


def _save_timepoints(combined: pd.DataFrame, out_dir: Path) -> None:
    for t_val, group in combined.groupby("time"):
        t_label = f"{t_val:.4f}".replace(".", "_")
        (
            group
            .drop(columns=["time"])
            .reset_index(drop=True)
            .to_csv(out_dir / f"expression_t{t_label}.csv", index=False)
        )


def plot_snapshot_counts(metadata: pd.DataFrame, output: str | Path) -> None:
    """Bar chart of how many cells were captured at each timepoint."""
    counts = metadata.groupby("time").size().reset_index(name="n_cells")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(counts["time"], counts["n_cells"], width=0.05, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Cells captured")
    ax.set_title("Cells per timepoint (destructive measurements)")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Sparse collection timepoints — mimics real experimental design (e.g. harvest
# cells at day 0, 1, 2, 3, 4 instead of at every simulation save step).
COLLECTION_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

N_PER_TIMEPOINT = 1000  # cells harvested (and destroyed) per collection timepoint per population


def main() -> None:
    out_dir = Path("output-destructive-measurements/")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_expr, all_meta = [], []
    cell_offset = 0
    time_grid = None

    for pop in POPULATIONS:
        print(f"Simulating {pop['label']} (gene_mu_modes={pop['gene_mu_modes']}) ...")
        sim = NetworkSimulatorPerGeneMu(
            num_genes=NUM_GENES,
            network_density=0.3,
            seed=pop["seed"],
            gene_mu_modes=pop["gene_mu_modes"],
            gene_mu_kwargs=pop["gene_mu_kwargs"],
        )
        data, time_grid = sim.simulate(
            T=4.0,
            num_samples=pop["num_samples"],
            save_every=15,
            dt=0.005,
        )
        print(f"  data shape: {data.shape}")

        gene_names = [f"gene_{i}" for i in range(NUM_GENES)]
        pd.DataFrame(sim.A, index=gene_names, columns=gene_names).to_csv(
            out_dir / f"grn_weighted_{pop['label']}.csv"
        )
        pd.DataFrame(
            (sim.A != 0).astype(int), index=gene_names, columns=gene_names
        ).to_csv(out_dir / f"grn_adjacency_{pop['label']}.csv")

        expr, meta = _sim_to_dataframes_destructive(
            data,
            time_grid,
            population_label=pop["label"],
            cell_id_offset=cell_offset,
            n_per_timepoint=N_PER_TIMEPOINT,
            collection_times=COLLECTION_TIMES,
            rng=np.random.default_rng(pop["seed"] + 100),
        )
        all_expr.append(expr)
        all_meta.append(meta)
        cell_offset += pop["num_samples"]

    expression = pd.concat(all_expr, ignore_index=True)
    metadata   = pd.concat(all_meta, ignore_index=True)
    print(
        f"Combined: {len(expression)} observations × "
        f"{expression.shape[1]} genes × "
        f"{metadata['population'].nunique()} populations "
        f"({N_PER_TIMEPOINT} cells sampled per timepoint per population)"
    )

    # Snapshot size diagnostic
    print("Plotting snapshot counts ...")
    plot_snapshot_counts(metadata, out_dir / "snapshot_counts.png")

    # Per-timepoint CSVs (now each cell appears in exactly one file)
    print("Saving per-timepoint expression matrices ...")
    timepoints_dir = out_dir / "expression_by_timepoint"
    timepoints_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([metadata[["cell_id", "time", "population"]], expression], axis=1)
    _save_timepoints(combined, timepoints_dir)
    print(f"  Saved {combined['time'].nunique()} timepoint files to {timepoints_dir}/")

    # cell_trajectories is omitted: each cell has only one timepoint,
    # so there are no trajectories to plot.

    print("Plotting dynamics ...")
    plot_dynamics(expression, metadata, output=out_dir / "dynamics_destructive.png")

    print("Plotting UMAP + PHATE embeddings ...")
    plot_embeddings(
        expression, metadata,
        output=out_dir / "embeddings_destructive.png",
        include_pca=True,
    )

    print("Plotting gene expression matrix ...")
    plot_expression_matrix(
        expression, metadata,
        n_cells=50,
        output=out_dir / "expression_matrix_destructive.png",
    )

    print("Plotting expression snapshots ...")
    plot_expression_snapshots(
        expression, metadata,
        n_timepoints=4,
        output=out_dir / "expression_snapshots_destructive.png",
    )

    print("Plotting clustermap ...")
    plot_clustermap(
        expression, metadata,
        n_cells=100,
        output=out_dir / "clustermap_destructive.png",
        explicit_cell_pop=True,
    )

    print("Done. All outputs in", out_dir)


if __name__ == "__main__":
    main()
