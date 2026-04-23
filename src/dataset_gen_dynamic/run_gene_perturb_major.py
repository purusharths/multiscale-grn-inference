"""
Gene-level perturbation simulation: 4 genes, 800 cells across 5 populations.

Population layout (per-gene mu_mode):
  pop_0  cells   0-399  all 4 genes constant
  pop_1  cells 400-499  gene_0 = sigmoid,   genes 1-3 constant
  pop_2  cells 500-599  gene_1 = heaviside, genes 0,2-3 constant
  pop_3  cells 600-699  gene_2 = linear,    genes 0-1,3 constant
  pop_4  cells 700-799  gene_3 = tanh,      genes 0-2 constant
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_gen_dynamic.housekeeping.enforce_diagonal_dominance import enforce_diagonal_dominance
from dataset_gen_dynamic.housekeeping.mu_options import (
    mu_constant,
    mu_heaviside,
    mu_linear,
    mu_sigmoid,
)
from dataset_gen_dynamic.visualizer import (
    plot_cell_trajectories,
    plot_clustermap,
    plot_dynamics,
    plot_embeddings,
    plot_expression_matrix,
    plot_expression_snapshots,
)

# ---------------------------------------------------------------------------
# tanh mu function (not in mu_options yet)
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
    """mu(t) = mu0 + delta * tanh(k * (t - t_star))"""
    if delta is None:
        delta = np.ones_like(mu0)
    if t_star is None:
        t_star = T / 2.0
    return mu0 + delta * np.tanh(k * (t - t_star))


# ---------------------------------------------------------------------------
# Per-gene simulator
# ---------------------------------------------------------------------------

class NetworkSimulatorPerGeneMu:
    """
    Identical to NetworkSimulatorNonStationaryMu but accepts a list of
    mu_mode strings — one per gene — instead of a single global mu_mode.
    """

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

        # Interaction matrix (same construction as original)
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

        # linear drift parameters (per-gene: each gene has its own scalar direction)
        self.mu_drift_dir = rng.normal(size=num_genes)
        self.mu_drift_amp = 4.0

    # ------------------------------------------------------------------

    def mu_t(self, t: float, T: float) -> np.ndarray:
        mu = np.empty(self.num_genes)
        for i, (mode, kw) in enumerate(zip(self.gene_mu_modes, self.gene_mu_kwargs)):
            mu0i = self.mu0[i : i + 1]          # shape (1,) — functions expect array
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

    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Population definitions
# ---------------------------------------------------------------------------

NUM_GENES = 4

POPULATIONS = [
    {
        "label":          "pop_0",
        "seed":           10,
        "num_samples":    400,
        "gene_mu_modes":  ["constant",  "constant",  "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_1",
        "seed":           11,
        "num_samples":    100,
        "gene_mu_modes":  ["sigmoid",   "constant",  "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_2",
        "seed":           12,
        "num_samples":    100,
        "gene_mu_modes":  ["constant",  "heaviside", "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_3",
        "seed":           13,
        "num_samples":    100,
        "gene_mu_modes":  ["constant",  "constant",  "linear",   "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_4",
        "seed":           14,
        "num_samples":    100,
        "gene_mu_modes":  ["constant",  "constant",  "constant", "tanh"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
]


# ---------------------------------------------------------------------------
# Helpers (copied / adapted from run_non_stationary.py)
# ---------------------------------------------------------------------------

def _sim_to_dataframes(
    data: np.ndarray,
    time_grid: np.ndarray,
    population_label: str = "pop_0",
    cell_id_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_samples, _, num_genes = data.shape
    gene_cols = [f"gene_{i}" for i in range(num_genes)]
    rows_expr, rows_meta = [], []

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


def _save_timepoints(combined: pd.DataFrame, out_dir: Path) -> None:
    for t_val, group in combined.groupby("time"):
        t_label = f"{t_val:.4f}".replace(".", "_")
        (
            group
            .drop(columns=["time"])
            .reset_index(drop=True)
            .to_csv(out_dir / f"expression_t{t_label}.csv", index=False)
        )


def plot_phase(
    data: np.ndarray,
    time_grid: np.ndarray,
    gene_x: int = 0,
    gene_y: int = 1,
    output: str | Path = "phase.png",
    n_show: int = 80,
) -> None:
    num_samples = data.shape[0]
    idx = np.random.default_rng(0).choice(num_samples, size=min(n_show, num_samples), replace=False)
    norm = plt.Normalize(time_grid[0], time_grid[-1])
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in idx:
        tx, ty = data[i, :, gene_x], data[i, :, gene_y]
        for t in range(len(time_grid) - 1):
            ax.plot(tx[t:t+2], ty[t:t+2], color=cmap(norm(time_grid[t])), linewidth=0.7, alpha=0.4)

    mean_x = data[:, :, gene_x].mean(axis=0)
    mean_y = data[:, :, gene_y].mean(axis=0)
    ax.plot(mean_x, mean_y, color="white", linewidth=2.5, label="mean", zorder=5)
    ax.plot(mean_x, mean_y, color="black", linewidth=1.0, zorder=4)
    ax.scatter([mean_x[0]],  [mean_y[0]],  color="lime", s=60, zorder=6, label="t=0")
    ax.scatter([mean_x[-1]], [mean_y[-1]], color="red",  s=60, zorder=6, label=f"t={time_grid[-1]:.1f}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="time")
    ax.set_xlabel(f"gene_{gene_x}")
    ax.set_ylabel(f"gene_{gene_y}")
    ax.set_title("Phase plot — gene-level perturbation", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_facecolor("#1a1a2e")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("output-gene-perturb-major/")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_expr, all_meta = [], []
    cell_offset = 0

    for pop in POPULATIONS:
        print(f"Simulating {pop['label']} "
              f"(gene_mu_modes={pop['gene_mu_modes']}) ...")
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

        # ground-truth GRN
        gene_names = [f"gene_{i}" for i in range(NUM_GENES)]
        pd.DataFrame(sim.A, index=gene_names, columns=gene_names).to_csv(
            out_dir / f"grn_weighted_{pop['label']}.csv"
        )
        pd.DataFrame(
            (sim.A != 0).astype(int), index=gene_names, columns=gene_names
        ).to_csv(out_dir / f"grn_adjacency_{pop['label']}.csv")

        expr, meta = _sim_to_dataframes(
            data, time_grid,
            population_label=pop["label"],
            cell_id_offset=cell_offset,
        )
        all_expr.append(expr)
        all_meta.append(meta)
        cell_offset += data.shape[0]

    expression = pd.concat(all_expr, ignore_index=True)
    metadata   = pd.concat(all_meta, ignore_index=True)
    print(
        f"Combined: {metadata['cell_id'].nunique()} cells × "
        f"{expression.shape[1]} genes × "
        f"{metadata['population'].nunique()} populations  "
        f"({len(expression)} total observations)"
    )

    # Phase plot (pop_1 = sigmoid on gene_0, gene_0 vs gene_1)
    print("Plotting phase plot ...")
    sim_phase = NetworkSimulatorPerGeneMu(
        num_genes=NUM_GENES, network_density=0.3, seed=11,
        gene_mu_modes=["sigmoid", "constant", "constant", "constant"],
    )
    pop1_data, _ = sim_phase.simulate(T=4.0, num_samples=100, save_every=15, dt=0.005)
    plot_phase(pop1_data, time_grid, gene_x=0, gene_y=1,
               output=out_dir / "phase_gene_perturb.png")

    # Per-timepoint CSVs
    print("Saving per-timepoint expression matrices ...")
    timepoints_dir = out_dir / "expression_by_timepoint"
    timepoints_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([metadata[["cell_id", "time", "population"]], expression], axis=1)
    _save_timepoints(combined, timepoints_dir)
    print(f"  Saved {combined['time'].nunique()} timepoint files to {timepoints_dir}/")

    print("Plotting cell trajectories ...")
    plot_cell_trajectories(
        expression, metadata,
        n_cells=6,
        output=out_dir / "cell_trajectories_gene_perturb.png",
    )

    print("Plotting dynamics ...")
    plot_dynamics(expression, metadata, output=out_dir / "dynamics_gene_perturb.png")

    print("Plotting UMAP + PHATE embeddings ...")
    plot_embeddings(
        expression, metadata,
        output=out_dir / "embeddings_gene_perturb.png",
        include_pca=True,
    )

    print("Plotting gene expression matrix ...")
    plot_expression_matrix(
        expression, metadata,
        n_cells=50,
        output=out_dir / "expression_matrix_gene_perturb.png",
    )

    print("Plotting expression snapshots ...")
    plot_expression_snapshots(
        expression, metadata,
        n_timepoints=4,
        output=out_dir / "expression_snapshots_gene_perturb.png",
    )

    print("Plotting clustermap ...")
    plot_clustermap(
        expression, metadata,
        n_cells=100,
        output=out_dir / "clustermap_gene_perturb.png",
        explicit_cell_pop=True,
    )

    print("Done. All outputs in", out_dir)


if __name__ == "__main__":
    main()
