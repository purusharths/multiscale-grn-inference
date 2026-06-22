"""
Visualisation utilities for multiscale GRN inference results.

All functions write PNG files to output_dir and return the figure for
interactive use.

Functions
---------
plot_loss_curve       — outer loss and component curves vs iteration
plot_theta_comparison — heatmap side-by-side: BIP init vs final A
plot_a_matrix         — single heatmap of any G×G matrix
plot_snapshot_pca     — PCA scatter of observed vs simulated snapshots
compare_grns          — metrics: Frobenius dist, correlation, sign accuracy
plot_grn_comparison   — heatmap grid: inferred vs each population's GRN
plot_edge_scatter     — scatter plot of inferred vs true edge weights
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    history: list[dict],
    output_dir: str | Path = "results",
    filename: str = "loss_curve.png",
) -> plt.Figure:
    """
    Plot total loss and each component (L_OU, L_FP, L_cons) vs outer iteration.
    """
    _ensure_dir(output_dir)
    steps = [r["step"] for r in history]
    total = [r["loss"] for r in history]
    l_ou  = [r["L_OU"] for r in history]
    l_fp  = [r["L_FP"] for r in history]
    l_cons = [r["L_cons"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(steps, total, color="black", linewidth=1.8, label="Total")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Total loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, l_ou,   label=r"$\mathcal{L}_\mathrm{OU}$",   linewidth=1.4)
    ax.plot(steps, l_fp,   label=r"$\mathcal{L}_\mathrm{FP}$",   linewidth=1.4)
    ax.plot(steps, l_cons, label=r"$\mathcal{L}_\mathrm{cons}$", linewidth=1.4)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Loss component")
    ax.set_title("Loss components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# A matrix comparison
# ---------------------------------------------------------------------------

def plot_theta_comparison(
    A_init: np.ndarray | torch.Tensor,
    A_hat: np.ndarray | torch.Tensor,
    output_dir: str | Path = "results",
    filename: str = "theta0_vs_thetahat.png",
    gene_names: list[str] | None = None,
) -> plt.Figure:
    """
    Side-by-side heatmap: BIP initialisation vs final inferred A.
    """
    _ensure_dir(output_dir)
    A_init = _to_numpy(A_init)
    A_hat  = _to_numpy(A_hat)
    G = A_init.shape[0]
    labels = gene_names or [f"g{i}" for i in range(G)]

    vmin = min(A_init.min(), A_hat.min())
    vmax = max(A_init.max(), A_hat.max())
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(
        axes,
        [A_init, A_hat],
        [r"$A_0$ (BIP init)", r"$\hat{A}$ (inferred)"],
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(G))
        ax.set_yticks(range(G))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("BIP warm start vs final GRN", fontsize=13)
    fig.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


def plot_a_matrix(
    A: np.ndarray | torch.Tensor,
    title: str = "Inferred drift matrix A",
    output_dir: str | Path = "results",
    filename: str = "A_hat.png",
    gene_names: list[str] | None = None,
) -> plt.Figure:
    """Single heatmap of a G×G matrix."""
    _ensure_dir(output_dir)
    A = _to_numpy(A)
    G = A.shape[0]
    labels = gene_names or [f"g{i}" for i in range(G)]

    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = max(abs(A.min()), abs(A.max()))
    im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# PCA snapshot comparison
# ---------------------------------------------------------------------------

def plot_snapshot_pca(
    observed: dict[int, np.ndarray],
    simulated: dict[int, np.ndarray] | None = None,
    output_dir: str | Path = "results",
    filename: str = "snapshot_pca.png",
    n_subsample: int = 500,
    rng: np.random.Generator | None = None,
) -> plt.Figure:
    """
    PCA scatter of observed (and optionally simulated) snapshots per timepoint.
    """
    from sklearn.decomposition import PCA

    _ensure_dir(output_dir)
    rng = rng or np.random.default_rng(0)

    timepoints = sorted(observed.keys())
    all_obs = np.vstack([observed[k] for k in timepoints])

    pca = PCA(n_components=2)
    pca.fit(all_obs)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(timepoints) - 1, 1)) for i in range(len(timepoints))]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, k in enumerate(timepoints):
        X = observed[k]
        idx = rng.choice(len(X), size=min(n_subsample, len(X)), replace=False)
        proj = pca.transform(X[idx])
        ax.scatter(proj[:, 0], proj[:, 1], c=[colors[i]], s=8, alpha=0.5, label=f"obs t={k}")

        if simulated and k in simulated:
            S = simulated[k]
            idx_s = rng.choice(len(S), size=min(n_subsample, len(S)), replace=False)
            proj_s = pca.transform(S[idx_s])
            ax.scatter(proj_s[:, 0], proj_s[:, 1], c=[colors[i]], s=8,
                       marker="x", alpha=0.5, label=f"sim t={k}")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: observed snapshots")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Ground-truth GRN comparison
# ---------------------------------------------------------------------------

def compare_grns(
    A_hat: np.ndarray | torch.Tensor,
    true_grns: dict[str, np.ndarray],
    reference: str | np.ndarray = "weighted",
    cell_counts: dict[str, int] | None = None,
) -> dict[str, float | str]:
    """
    Compute comparison metrics between A_hat and the ground-truth GRN(s).

    Parameters
    ----------
    A_hat      : (G, G) inferred drift matrix
    true_grns  : output of data.load_true_grns()
    reference  : "weighted" (default) — compare against cell-count-weighted mean;
                 a pop label like "pop_0" — compare against that population only;
                 a (G, G) array — compare against that matrix directly.
    cell_counts: population cell counts for the weighted reference.
                 Defaults to the known destructive-measurement counts when None.

    Returns
    -------
    dict with keys:
        frob_dist   — Frobenius distance  ||A_hat - A_ref||_F
        rel_frob    — relative Frobenius  ||A_hat - A_ref||_F / ||A_ref||_F
        pearson_r   — Pearson r on all G² entries (entry-wise correlation)
        mae         — mean absolute error on all entries
        sign_acc    — fraction of entries with the same sign as A_ref
        reference   — label describing what A_ref is
    """
    from data import pooled_grn

    A_hat = _to_numpy(A_hat)

    _DEFAULT_COUNTS = {"pop_0": 40000, "pop_1": 10000, "pop_2": 10000,
                       "pop_3": 10000, "pop_4": 10000}

    if isinstance(reference, str) and reference == "weighted":
        counts = cell_counts or _DEFAULT_COUNTS
        A_ref = pooled_grn(true_grns, counts)
        ref_label = "weighted mean (pop counts)"
    elif isinstance(reference, str):
        A_ref = true_grns[reference]
        ref_label = reference
    else:
        A_ref = np.asarray(reference)
        ref_label = "custom"

    diff = A_hat - A_ref
    frob_dist = float(np.linalg.norm(diff, "fro"))
    rel_frob  = frob_dist / max(float(np.linalg.norm(A_ref, "fro")), 1e-12)

    h, r = A_hat.ravel(), A_ref.ravel()
    pearson_r = float(np.corrcoef(h, r)[0, 1])
    mae = float(np.mean(np.abs(diff)))
    sign_acc = float(np.mean(np.sign(h) == np.sign(r)))

    return {
        "reference":  ref_label,
        "frob_dist":  frob_dist,
        "rel_frob":   rel_frob,
        "pearson_r":  pearson_r,
        "mae":        mae,
        "sign_acc":   sign_acc,
    }


def plot_grn_comparison(
    A_hat: np.ndarray | torch.Tensor,
    true_grns: dict[str, np.ndarray],
    cell_counts: dict[str, int] | None = None,
    output_dir: str | Path = "results",
    filename: str = "grn_comparison.png",
    gene_names: list[str] | None = None,
) -> plt.Figure:
    """
    Heatmap grid: inferred A_hat alongside each population's true GRN and the
    cell-count-weighted mean.  Metrics are printed in each panel title.

    Layout: top row = inferred + weighted reference;
            bottom rows = individual population GRNs.
    """
    from data import pooled_grn

    _ensure_dir(output_dir)
    A_hat = _to_numpy(A_hat)

    _DEFAULT_COUNTS = {"pop_0": 40000, "pop_1": 10000, "pop_2": 10000,
                       "pop_3": 10000, "pop_4": 10000}
    counts = cell_counts or _DEFAULT_COUNTS
    A_ref = pooled_grn(true_grns, counts)

    pops = sorted(true_grns.keys())
    G = A_hat.shape[0]
    labels = gene_names or [f"gene_{i}" for i in range(G)]

    # panels: inferred, weighted ref, then each pop
    panels = [("Inferred $\\hat{A}$", A_hat), ("Weighted ref", A_ref)] + \
             [(p, true_grns[p]) for p in pops]

    n_cols = 3
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    all_vals = np.concatenate([m.ravel() for _, m in panels])
    vmax = max(abs(all_vals.min()), abs(all_vals.max()))

    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        # add metrics vs inferred
        if mat is not A_hat:
            diff = A_hat - mat
            r = float(np.corrcoef(A_hat.ravel(), mat.ravel())[0, 1])
            frob = float(np.linalg.norm(diff, "fro"))
            ax.set_title(f"{title}\nr={r:.2f}  ‖Δ‖={frob:.2f}", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)
        ax.set_xticks(range(G))
        ax.set_yticks(range(G))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("GRN comparison: inferred vs ground truth", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


def plot_edge_scatter(
    A_hat: np.ndarray | torch.Tensor,
    true_grns: dict[str, np.ndarray],
    cell_counts: dict[str, int] | None = None,
    output_dir: str | Path = "results",
    filename: str = "edge_scatter.png",
) -> plt.Figure:
    """
    Scatter plot of inferred edge weights vs ground-truth (weighted mean).
    Each point is one (i, j) entry.  Diagonal = perfect recovery.
    """
    from data import pooled_grn

    _ensure_dir(output_dir)
    A_hat = _to_numpy(A_hat)

    _DEFAULT_COUNTS = {"pop_0": 40000, "pop_1": 10000, "pop_2": 10000,
                       "pop_3": 10000, "pop_4": 10000}
    counts = cell_counts or _DEFAULT_COUNTS
    A_ref = pooled_grn(true_grns, counts)

    x, y = A_ref.ravel(), A_hat.ravel()
    r = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(y - x)))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, alpha=0.7, edgecolors="k", linewidths=0.4, s=60, zorder=3)
    lim = max(abs(x).max(), abs(y).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="y = x")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True edge weight (weighted mean GRN)")
    ax.set_ylabel("Inferred edge weight $\\hat{A}_{ij}$")
    ax.set_title(f"Edge weight recovery\nr={r:.3f}  MAE={mae:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = _ensure_dir(output_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig
