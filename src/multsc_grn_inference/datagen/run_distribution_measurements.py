"""
Distribution-based destructive measurement simulation.

Workflow per population per collection timepoint
-------------------------------------------------
1. Simulate a large ensemble of SDE trajectories (NUM_ENSEMBLE cells).
2. At each COLLECTION_TIME, snapshot all ensemble cells → population cloud
   in gene-expression space.
3. Fit a multivariate Gaussian to that cloud (captures mean expression +
   gene-gene covariance at that moment in time).
4. Draw N_CELLS_PER_TIMEPOINT samples from the fitted distribution →
   these are the stored "sequenced" observations.

Why this is more realistic than picking individual trajectories
--------------------------------------------------------------
- The number of stored cells is independent of ensemble size.
- Gene-gene correlations at each timepoint are preserved via the covariance.
- Smooths over simulation noise; the distribution is the thing that matters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from multsc_grn_inference.housekeeping.enforce_diagonal_dominance import enforce_diagonal_dominance
from multsc_grn_inference.housekeeping.mu_options import (
    mu_constant,
    mu_heaviside,
    mu_linear,
    mu_sigmoid,
)
from multsc_grn_inference.visualizer import (
    plot_clustermap,
    plot_dynamics_snapshot,
    plot_embeddings,
    plot_expression_matrix,
    plot_expression_snapshots,
)


# ---------------------------------------------------------------------------
# tanh mu (same as run_destructive_measurements.py)
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


# ---------------------------------------------------------------------------
# SDE simulator
# ---------------------------------------------------------------------------

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

        # Gene regulatory network matrix
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

    def simulate_ensemble(
        self,
        T: float = 4.0,
        num_cells: int = 5000,
        collection_times: list[float] | None = None,
        dt: float = 0.005,
    ) -> dict[float, np.ndarray]:
        """
        Simulate `num_cells` independent trajectories and return snapshots of
        the full population at each collection time.

        Returns
        -------
        snapshots : dict mapping collection_time -> array of shape (num_cells, num_genes)
        """
        collection_times = sorted(set(collection_times or [T]))
        # Pre-compute which steps to snapshot
        snap_steps = {round(t / dt): t for t in collection_times}

        steps = int(T / dt)
        snapshots: dict[float, list[np.ndarray]] = {t: [] for t in collection_times}

        for _ in range(num_cells):
            mu_init = self.mu_t(0.0, T)
            X = mu_init + 0.5 * self.rng.normal(size=self.num_genes)
            X = np.maximum(X, 0.1)

            if 0 in snap_steps:
                snapshots[snap_steps[0]].append(X.copy())

            for step in range(1, steps + 1):
                t = step * dt
                mu = self.mu_t(t, T)
                drift = self.A @ (mu - X)
                noise = np.sqrt(np.diag(self.D) * dt) * self.rng.normal(size=self.num_genes)
                X += drift * dt + noise
                X = np.maximum(X, 0.05)

                rounded = round(step)
                if rounded in snap_steps:
                    snapshots[snap_steps[rounded]].append(X.copy())

        return {t: np.array(cells) for t, cells in snapshots.items()}


# ---------------------------------------------------------------------------
# Distribution fitting + sampling
# ---------------------------------------------------------------------------

def fit_and_sample(
    population_snapshot: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    regularisation: float = 1e-4,
) -> np.ndarray:
    """
    Fit a multivariate Gaussian to `population_snapshot` (shape: n_cells × n_genes)
    and draw `n_samples` from it.

    `regularisation` adds a small diagonal to the covariance to prevent
    singular matrices when genes are nearly collinear.
    """
    mean = population_snapshot.mean(axis=0)
    cov  = np.cov(population_snapshot, rowvar=False)
    cov += regularisation * np.eye(cov.shape[0])

    samples = rng.multivariate_normal(mean, cov, size=n_samples)
    return np.maximum(samples, 0.0)   # expression is non-negative


# ---------------------------------------------------------------------------
# Population definitions
# ---------------------------------------------------------------------------

NUM_GENES = 4

POPULATIONS = [
    {
        "label":          "pop_0",
        "seed":           10,
        "gene_mu_modes":  ["constant",  "constant",  "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_1",
        "seed":           11,
        "gene_mu_modes":  ["sigmoid",   "constant",  "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_2",
        "seed":           12,
        "gene_mu_modes":  ["constant",  "heaviside", "constant", "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_3",
        "seed":           13,
        "gene_mu_modes":  ["constant",  "constant",  "linear",   "constant"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
    {
        "label":          "pop_4",
        "seed":           14,
        "gene_mu_modes":  ["constant",  "constant",  "constant", "tanh"],
        "gene_mu_kwargs": [{},          {},          {},         {}],
    },
]

# Sparse experimental collection timepoints
COLLECTION_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Cells drawn from the fitted distribution per timepoint per population
N_CELLS_PER_TIMEPOINT = 1000

# Size of the SDE ensemble used to build the distribution (not stored directly)
NUM_ENSEMBLE = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    counts = metadata.groupby("time").size().reset_index(name="n_cells")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(
        counts["time"], counts["n_cells"],
        width=0.08, color="steelblue", edgecolor="white", linewidth=0.3,
    )
    ax.set_xlabel("Collection timepoint")
    ax.set_ylabel("Cells stored")
    ax.set_title("Cells per timepoint (distribution-sampled destructive measurements)")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("output-distribution-measurements/")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_expr, all_meta = [], []
    cell_offset = 0
    gene_cols = [f"gene_{i}" for i in range(NUM_GENES)]

    for pop in POPULATIONS:
        print(f"Simulating ensemble for {pop['label']} ...")
        sim = NetworkSimulatorPerGeneMu(
            num_genes=NUM_GENES,
            network_density=0.3,
            seed=pop["seed"],
            gene_mu_modes=pop["gene_mu_modes"],
            gene_mu_kwargs=pop["gene_mu_kwargs"],
        )

        snapshots = sim.simulate_ensemble(
            T=max(COLLECTION_TIMES),
            num_cells=NUM_ENSEMBLE,
            collection_times=COLLECTION_TIMES,
            dt=0.005,
        )
        print(f"  Ensemble snapshots at {len(snapshots)} timepoints × {NUM_ENSEMBLE} cells each")

        # Save GRN for this population
        pd.DataFrame(sim.A, index=gene_cols, columns=gene_cols).to_csv(
            out_dir / f"grn_weighted_{pop['label']}.csv"
        )
        pd.DataFrame(
            (sim.A != 0).astype(int), index=gene_cols, columns=gene_cols
        ).to_csv(out_dir / f"grn_adjacency_{pop['label']}.csv")

        # Fit distribution at each collection time; sample stored observations
        rng = np.random.default_rng(pop["seed"] + 200)
        T_max = max(COLLECTION_TIMES)

        rows_expr, rows_meta = [], []
        obs_id = 0
        for t in sorted(snapshots):
            population_cloud = snapshots[t]           # shape: (NUM_ENSEMBLE, NUM_GENES)
            sampled = fit_and_sample(population_cloud, N_CELLS_PER_TIMEPOINT, rng)

            for cell_expr in sampled:
                rows_expr.append(cell_expr)
                rows_meta.append({
                    "cell_id":    cell_offset + obs_id,
                    "time":       t,
                    "population": pop["label"],
                    "pseudotime": t / T_max,
                })
                obs_id += 1

        expr = pd.DataFrame(rows_expr, columns=gene_cols)
        meta = pd.DataFrame(rows_meta)

        print(f"  Stored {len(expr)} observations for {pop['label']}")
        all_expr.append(expr)
        all_meta.append(meta)
        cell_offset += obs_id

    expression = pd.concat(all_expr, ignore_index=True)
    metadata   = pd.concat(all_meta, ignore_index=True)
    print(
        f"\nCombined dataset: {len(expression)} cells × "
        f"{expression.shape[1]} genes × "
        f"{metadata['population'].nunique()} populations"
    )

    print("Plotting snapshot counts ...")
    plot_snapshot_counts(metadata, out_dir / "snapshot_counts.png")

    print("Saving per-timepoint expression matrices ...")
    timepoints_dir = out_dir / "expression_by_timepoint"
    timepoints_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([metadata[["cell_id", "time", "population"]], expression], axis=1)
    _save_timepoints(combined, timepoints_dir)
    print(f"  Saved {combined['time'].nunique()} timepoint files to {timepoints_dir}/")

    print("Plotting dynamics ...")
    plot_dynamics_snapshot(expression, metadata, output=out_dir / "dynamics.png")

    print("Plotting UMAP + PHATE embeddings ...")
    plot_embeddings(
        expression, metadata,
        output=out_dir / "embeddings.png",
        include_pca=True,
    )

    print("Plotting gene expression matrix ...")
    plot_expression_matrix(
        expression, metadata,
        n_cells=50,
        output=out_dir / "expression_matrix.png",
    )

    print("Plotting expression snapshots ...")
    plot_expression_snapshots(
        expression, metadata,
        n_timepoints=4,
        output=out_dir / "expression_snapshots.png",
    )

    print("Plotting clustermap ...")
    plot_clustermap(
        expression, metadata,
        n_cells=100,
        output=out_dir / "clustermap.png",
        explicit_cell_pop=True,
    )

    print("Done. All outputs in", out_dir)


if __name__ == "__main__":
    main()
