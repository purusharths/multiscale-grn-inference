"""
Bayesian OU-process parameter recovery from destructive measurement data.

Reads output-destructive-measurements/ (cross-sectional snapshots — each cell
observed at exactly one timepoint) and recovers A, mu, D via MCMC (PyMC).\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Challenge: no cell has a (X_t, X_{t+1}) pair because destructive sequencing
           lyses the cell at measurement time.

Approach: nearest-neighbour coupling
    For each consecutive pair of collection timepoints (t_k, t_{k+1}):
    subsample N_PAIRS cells from t_k, find their closest neighbour in t_{k+1}
    (by Euclidean distance in gene space), and treat those pairs as
    pseudo-transitions.

Note: all cells across all populations are pooled — population labels are
      simulation scaffolding only.  We know cells are perturbed but not how,
      and recover one set of OU parameters (A, mu, D) from the aggregate
      cross-sectional data.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA_DIR = Path("output-destructive-measurements")

NUM_GENES = 4

N_PAIRS     = 300
MCMC_DRAWS  = 2000
MCMC_TUNE   = 1000
MCMC_CHAINS = 4

# Cell counts per population from run_destructive_measurements.py
# Used to weight the pooled ground-truth A
POP_CELL_COUNTS = {
    "pop_0": 40000,
    "pop_1": 10000,
    "pop_2": 10000,
    "pop_3": 10000,
    "pop_4": 10000,
}

# Reference A used for comparison plots.
#   "pop0"   — pop_0's ground-truth A (unperturbed baseline, 50% of cells; default)
#   "pooled" — cell-count-weighted average across all populations
TRUE_A_MODE: str = "pop0"

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_expression_by_time(data_dir: Path) -> dict[float, np.ndarray]:
    """
    Load all per-timepoint expression CSVs, pooling all cells regardless of
    population label.  Returns {time: (n_cells, n_genes) array}.
    """
    gene_cols = [f"gene_{g}" for g in range(NUM_GENES)]
    result: dict[float, np.ndarray] = {}

    tp_dir = data_dir / "expression_by_timepoint"
    for csv_path in sorted(tp_dir.glob("expression_t*.csv")):
        df = pd.read_csv(csv_path)
        t_str = csv_path.stem.replace("expression_t", "").replace("_", ".", 1)
        t = float(t_str)
        result[t] = df[gene_cols].values.astype(np.float32)

    return dict(sorted(result.items()))


def load_true_grns(data_dir: Path) -> dict[str, np.ndarray]:
    """
    Load the ground-truth weighted A matrices saved by run_destructive_measurements.py.
    Returns {pop_label: (G, G) array}.
    """
    grns: dict[str, np.ndarray] = {}
    for path in sorted(data_dir.glob("grn_weighted_pop_*.csv")):
        label = path.stem.replace("grn_weighted_", "")
        grns[label] = pd.read_csv(path, index_col=0).values.astype(np.float64)
    return grns


def pooled_true_A(true_grns: dict[str, np.ndarray]) -> np.ndarray:
    """Cell-count-weighted average of all per-population true A matrices."""
    total = sum(POP_CELL_COUNTS.get(k, 1) for k in true_grns)
    return sum(
        POP_CELL_COUNTS.get(k, 1) / total * A
        for k, A in true_grns.items()
    )


def get_reference_A(true_grns: dict[str, np.ndarray], mode: str = TRUE_A_MODE) -> tuple[np.ndarray, str]:
    """
    Return (A_ref, label) for the chosen comparison mode.

    Parameters
    ----------
    mode : "pop0"   — pop_0's ground-truth A (unperturbed baseline, 50% of pooled cells)
           "pooled" — cell-count-weighted average across all populations
    """
    if mode == "pop0":
        return true_grns["pop_0"], "True A  (pop_0, unperturbed)"
    if mode == "pooled":
        return pooled_true_A(true_grns), "True A  (pooled, weighted)"
    raise ValueError(f"Unknown TRUE_A_MODE '{mode}'. Choose 'pop0' or 'pooled'.")


# --------------------------------------------------------------------------
# Pseudo-transition construction via nearest-neighbour coupling
# --------------------------------------------------------------------------

def build_nn_transitions(
    expr_by_time: dict[float, np.ndarray],
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    For each consecutive (t_k, t_{k+1}) pair: subsample n_pairs cells from
    t_k, find their nearest neighbour in t_{k+1}, collect (before, after).

    Returns
    -------
    X_t  : (M, G)   before states
    X_t1 : (M, G)   after  states
    dt   : mean gap between consecutive collection times
    """
    times = sorted(expr_by_time.keys())
    dts = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    dt = float(np.mean(dts))

    X_t_list, X_t1_list = [], []
    for t_k, t_k1 in zip(times[:-1], times[1:]):
        before_pool = expr_by_time[t_k]
        after_pool  = expr_by_time[t_k1]

        n = min(n_pairs, len(before_pool), len(after_pool))
        idx = rng.choice(len(before_pool), size=n, replace=False)
        before = before_pool[idx]

        tree = cKDTree(after_pool)
        _, nn_idx = tree.query(before, k=1)
        after = after_pool[nn_idx]

        X_t_list.append(before)
        X_t1_list.append(after)

    return np.vstack(X_t_list), np.vstack(X_t1_list), dt


# --------------------------------------------------------------------------
# Bayesian model
# --------------------------------------------------------------------------

def build_model(X_t: np.ndarray, X_t1: np.ndarray, dt: float) -> pm.Model:
    """
    Likelihood: X_{t+1} | X_t, A, μ, D ~ N(X_t + dt·A(μ − X_t), diag(D·dt))
    """
    G = X_t.shape[1]
    I = np.eye(G, dtype=np.float32)

    with pm.Model() as model:
        Xt  = pm.Data("X_t",  X_t.astype(np.float32))
        Xt1 = pm.Data("X_t1", X_t1.astype(np.float32))

        A_diag = pm.Normal("A_diag", mu=1.2, sigma=0.3, shape=G)
        A_off  = pm.Laplace("A_off", mu=0.0, b=2.0, shape=(G, G))
        A      = pm.Deterministic("A", A_off * (1.0 - I) + pt.diag(A_diag))

        mu = pm.TruncatedNormal("mu", mu=3.0, sigma=0.5,
                                lower=0.5, upper=6.0, shape=G)
        log_D = pm.TruncatedNormal("log_D", mu=np.log(0.15), sigma=0.3,
                                   lower=np.log(0.01), upper=np.log(1.0), shape=G)
        D_diag = pm.Deterministic("D_diag", pt.exp(log_D))

        pred  = pm.Deterministic("pred", Xt + dt * ((mu - Xt) @ A.T))
        sigma = pt.sqrt(D_diag * dt)

        pm.Normal("obs", mu=pred, sigma=sigma, observed=Xt1)

    return model


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def plot_data_overview(expr_by_time: dict[float, np.ndarray], out_dir: Path) -> None:
    """Mean ± std expression of each gene over collection timepoints (all cells pooled)."""
    times = sorted(expr_by_time.keys())
    means = np.array([expr_by_time[t].mean(axis=0) for t in times])
    stds  = np.array([expr_by_time[t].std(axis=0)  for t in times])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_GENES))
    for g in range(NUM_GENES):
        ax.plot(times, means[:, g], color=colors[g], label=f"gene_{g}", lw=1.8)
        ax.fill_between(times,
                        means[:, g] - stds[:, g],
                        means[:, g] + stds[:, g],
                        color=colors[g], alpha=0.15)
    ax.set_xlabel("Collection time")
    ax.set_ylabel("Expression (mean ± std)")
    ax.set_title("All cells pooled — gene expression over time")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "data_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_grn_comparison(
    true_grns: dict[str, np.ndarray],
    idata: az.InferenceData,
    out_dir: Path,
) -> None:
    """
    Three-panel heatmap: pooled true A (cell-count-weighted average),
    inferred posterior mean A, and posterior std A.
    """
    post   = idata.posterior
    A_mean = post["A"].mean(dim=("chain", "draw")).values
    A_std  = post["A"].std(dim=("chain", "draw")).values
    A_true, true_label = get_reference_A(true_grns)

    vmin = min(A_true.min(), A_mean.min())
    vmax = max(A_true.max(), A_mean.max())

    fig, (ax_true, ax_mean, ax_std) = plt.subplots(1, 3, figsize=(11, 3.8))

    im0 = ax_true.imshow(A_true, cmap="magma", vmin=vmin, vmax=vmax)
    ax_true.set_title(true_label, fontsize=9)
    ax_true.set_xticks(range(NUM_GENES))
    ax_true.set_yticks(range(NUM_GENES))
    fig.colorbar(im0, ax=ax_true, fraction=0.046, pad=0.04)

    im1 = ax_mean.imshow(A_mean, cmap="magma", vmin=vmin, vmax=vmax)
    ax_mean.set_title("Inferred A  (post. mean)", fontsize=9)
    ax_mean.set_xticks(range(NUM_GENES))
    ax_mean.set_yticks(range(NUM_GENES))
    fig.colorbar(im1, ax=ax_mean, fraction=0.046, pad=0.04)

    im2 = ax_std.imshow(A_std, cmap="viridis")
    ax_std.set_title("Inferred A  (post. std)", fontsize=9)
    ax_std.set_xticks(range(NUM_GENES))
    ax_std.set_yticks(range(NUM_GENES))
    fig.colorbar(im2, ax=ax_std, fraction=0.046, pad=0.04)

    frob = np.linalg.norm(A_mean - A_true)
    mae  = np.mean(np.abs(A_mean - A_true))
    fig.suptitle(
        f"{true_label} vs inferred  —  ||ΔA||_F = {frob:.3f},  MAE = {mae:.3f}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path = out_dir / "grn_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_empirical_vs_posterior(
    X_t: np.ndarray,
    X_t1: np.ndarray,
    dt: float,
    idata: az.InferenceData,
    out_dir: Path,
    n_pp_samples: int = 300,
) -> None:
    """
    Two-row comparison per gene:
      Row 0 — observed X_{t+1} vs posterior predictive X_{t+1}
      Row 1 — empirical increment (X_{t+1}−X_t)/dt vs posterior-implied drift A(μ−X_t)
    """
    post    = idata.posterior
    A_flat  = post["A"].values.reshape(-1, NUM_GENES, NUM_GENES).astype(np.float64)
    mu_flat = post["mu"].values.reshape(-1, NUM_GENES).astype(np.float64)

    rng = np.random.default_rng(0)
    idx = rng.choice(A_flat.shape[0], size=min(n_pp_samples, A_flat.shape[0]), replace=False)
    A_s  = A_flat[idx]   # (S, G, G)
    mu_s = mu_flat[idx]  # (S, G)

    # diff[s, n, g] = mu_s[s, g] - X_t[n, g]
    diff = mu_s[:, np.newaxis, :] - X_t[np.newaxis, :, :]  # (S, N, G)

    # pred[s, n, h] = X_t[n, h] + dt * sum_g diff[s,n,g] * A_s[s,h,g]
    # i.e. diff[s,n] @ A_s[s].T  →  einsum("sng,shg->snh", diff, A_s)
    drift_pp = np.einsum("sng,shg->snh", diff, A_s)           # (S, N, G)
    pred_pp  = X_t[np.newaxis, :, :] + dt * drift_pp          # (S, N, G)

    fig, axes = plt.subplots(2, NUM_GENES, figsize=(4 * NUM_GENES, 7))

    for g in range(NUM_GENES):
        obs_next  = X_t1[:, g]
        pp_next   = pred_pp[:, :, g].ravel()
        lo = min(obs_next.min(), pp_next.min())
        hi = max(obs_next.max(), pp_next.max())
        bins = np.linspace(lo, hi, 50)

        axes[0, g].hist(obs_next, bins=bins, density=True, alpha=0.6,
                        color="#d6604d", label="Observed")
        axes[0, g].hist(pp_next,  bins=bins, density=True, alpha=0.4,
                        color="#4393c3", label="Post. pred.")
        axes[0, g].set_title(f"gene_{g}  —  X_{{t+1}}", fontsize=9)
        axes[0, g].legend(fontsize=7)

        emp_drift = (X_t1[:, g] - X_t[:, g]) / dt
        pp_drift  = drift_pp[:, :, g].ravel()
        lo2 = min(emp_drift.min(), pp_drift.min())
        hi2 = max(emp_drift.max(), pp_drift.max())
        bins2 = np.linspace(lo2, hi2, 50)

        axes[1, g].hist(emp_drift, bins=bins2, density=True, alpha=0.6,
                        color="#d6604d", label="Empirical")
        axes[1, g].hist(pp_drift,  bins=bins2, density=True, alpha=0.4,
                        color="#4393c3", label="Post. drift")
        axes[1, g].set_title(f"gene_{g}  —  (X_{{t+1}}−X_t)/dt", fontsize=9)
        axes[1, g].legend(fontsize=7)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    fig.suptitle("Empirical data vs posterior predictive", fontsize=12)
    fig.tight_layout()
    path = out_dir / "empirical_vs_posterior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _build_nx_graph(A: np.ndarray, gene_labels: list[str], threshold: float = 0.05) -> tuple:
    """Build a DiGraph from A; return (graph, edge_list, edge_widths)."""
    G = nx.DiGraph()
    G.add_nodes_from(gene_labels)
    edge_list, edge_widths = [], []
    for src in range(len(gene_labels)):
        for tgt in range(len(gene_labels)):
            if src == tgt:
                continue
            w = A[tgt, src]   # A[tgt, src]: src influences tgt
            if abs(w) < threshold:
                continue
            edge_list.append((gene_labels[src], gene_labels[tgt]))
            edge_widths.append(max(0.8, abs(w) * 3.0))
    G.add_edges_from(edge_list)
    return G, edge_list, edge_widths


def plot_network(true_grns: dict[str, np.ndarray], idata: az.InferenceData, out_dir: Path) -> None:
    """
    Two side-by-side NetworkX directed graphs:
      Left  — pooled true A (cell-count-weighted average)
      Right — inferred A (posterior mean); edges in the top-25% posterior std
              are coloured red to flag high uncertainty
    """
    post   = idata.posterior
    A_mean = post["A"].mean(dim=("chain", "draw")).values
    A_std  = post["A"].std(dim=("chain", "draw")).values
    A_true, true_label = get_reference_A(true_grns)

    off_mask      = ~np.eye(NUM_GENES, dtype=bool)
    std_threshold = np.percentile(A_std[off_mask], 75)

    gene_labels = [f"g{i}" for i in range(NUM_GENES)]

    G_true, edges_true, widths_true = _build_nx_graph(A_true, gene_labels)
    G_inf,  edges_inf,  widths_inf  = _build_nx_graph(A_mean, gene_labels)

    edge_colors_inf = [
        "red" if A_std[gene_labels.index(tgt), gene_labels.index(src)] > std_threshold
        else "#4393c3"
        for src, tgt in edges_inf
    ]

    fig, (ax_true, ax_inf) = plt.subplots(1, 2, figsize=(12, 5))
    pos = nx.circular_layout(G_true)   # same layout for both

    node_kw = dict(node_size=900, node_color="#f5f5f5", edgecolors="black", linewidths=1.5)
    edge_kw = dict(arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.15")

    # ---- true A ----
    nx.draw_networkx_nodes(G_true, pos, ax=ax_true, **node_kw)
    nx.draw_networkx_labels(G_true, pos, font_size=11, ax=ax_true)
    nx.draw_networkx_edges(G_true, pos, edgelist=edges_true,
                           edge_color="#555555", width=widths_true, ax=ax_true, **edge_kw)
    ax_true.set_title(true_label, fontsize=10)
    ax_true.axis("off")

    # ---- inferred A ----
    nx.draw_networkx_nodes(G_inf, pos, ax=ax_inf, **node_kw)
    nx.draw_networkx_labels(G_inf, pos, font_size=11, ax=ax_inf)
    nx.draw_networkx_edges(G_inf, pos, edgelist=edges_inf,
                           edge_color=edge_colors_inf, width=widths_inf, ax=ax_inf, **edge_kw)

    legend_handles = [
        Line2D([0], [0], color="red",     lw=2.5,
               label=f"High uncertainty  (std > {std_threshold:.2f}, top 25%)"),
        Line2D([0], [0], color="#4393c3", lw=2.5,
               label="Lower uncertainty"),
    ]
    ax_inf.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax_inf.set_title("Inferred GRN  (edge width ∝ |A_mean|)", fontsize=10)
    ax_inf.axis("off")

    fig.suptitle("GRN comparison  —  true vs inferred", fontsize=11)
    fig.tight_layout()
    path = out_dir / "grn_network.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_posterior(idata: az.InferenceData, out_dir: Path) -> None:
    """Bar charts for posterior μ and D."""
    post    = idata.posterior
    mu_mean = post["mu"].mean(dim=("chain", "draw")).values
    mu_std  = post["mu"].std(dim=("chain", "draw")).values
    D_mean  = post["D_diag"].mean(dim=("chain", "draw")).values
    D_std   = post["D_diag"].std(dim=("chain", "draw")).values

    fig, (ax_mu, ax_D) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(NUM_GENES)

    ax_mu.bar(x, mu_mean, yerr=mu_std, capsize=4, color="#4393c3")
    ax_mu.set_xticks(x)
    ax_mu.set_xticklabels([f"g{g}" for g in x])
    ax_mu.set_ylabel("μ")
    ax_mu.set_title("Posterior μ")

    ax_D.bar(x, D_mean, yerr=D_std, capsize=4, color="#4393c3")
    ax_D.set_xticks(x)
    ax_D.set_xticklabels([f"g{g}" for g in x])
    ax_D.set_ylabel("D diag")
    ax_D.set_title("Posterior D")

    fig.tight_layout()
    path = out_dir / "mu_D_posterior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_traces(idata: az.InferenceData, out_dir: Path) -> None:
    """Trace plot for diagonal A entries, mu, log_D."""
    az.plot_trace(idata, var_names=["mu", "A_diag", "log_D"])
    plt.suptitle("Posterior traces", y=1.01, fontsize=10)
    plt.tight_layout()
    path = out_dir / "trace.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("output-bayes-destructive/")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found. Run run_destructive_measurements.py first."
        )

    expr_by_time = load_expression_by_time(DATA_DIR)
    n_times = len(expr_by_time)
    n_cells = next(iter(expr_by_time.values())).shape[0]
    print(f"Loaded {n_times} timepoints × ~{n_cells} cells each (all populations pooled)")

    true_grns = load_true_grns(DATA_DIR)
    print(f"Loaded ground-truth GRNs for: {list(true_grns.keys())}")

    plot_data_overview(expr_by_time, out_dir)

    rng = np.random.default_rng(42)
    X_t, X_t1, dt_obs = build_nn_transitions(expr_by_time, N_PAIRS, rng)
    print(f"Pseudo-transitions: {X_t.shape[0]} pairs,  dt_obs = {dt_obs:.4f}")

    model = build_model(X_t, X_t1, dt_obs)
    print("Sampling ...")
    with model:
        idata = pm.sample(
            draws=MCMC_DRAWS,
            tune=MCMC_TUNE,
            chains=MCMC_CHAINS,
            target_accept=0.9,
            random_seed=42,
            progressbar=True,
        )

    idata.to_netcdf(out_dir / "idata.nc")

    summary = az.summary(idata, var_names=["A", "mu", "D_diag"], hdi_prob=0.95)
    summary.to_csv(out_dir / "posterior_summary.csv")
    print(summary)

    plot_grn_comparison(true_grns, idata, out_dir)
    plot_empirical_vs_posterior(X_t, X_t1, dt_obs, idata, out_dir)
    plot_network(true_grns, idata, out_dir)
    plot_posterior(idata, out_dir)
    plot_traces(idata, out_dir)

    post = idata.posterior
    print(f"\nμ_inferred: {post['mu'].mean(dim=('chain','draw')).values.round(3)}")
    print(f"D_inferred: {post['D_diag'].mean(dim=('chain','draw')).values.round(3)}")
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
