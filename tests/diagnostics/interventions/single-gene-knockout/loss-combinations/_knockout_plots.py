"""
The two figures, factored out so the serial runner and the array merger
produce byte-identical output.

  metrics_and_heatmaps -> loss_comparison_knockout.png
      Metric bars + the recovered A matrices as heatmaps: are the NUMBERS right.
  grn_networks         -> grn_networks_knockout.png
      The recovered regulatory networks as graphs: is the STRUCTURE right.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from multsc_grn_inference.housekeeping.grn_graph import (
    ACTIVATE,
    INHIBIT,
    draw_grn,
    shared_layout,
    true_edge_set,
)

from _knockout_shared import (
    COMBINATIONS,
    KO_GENE,
    MAXITER,
    N_GENES,
    NETWORK_DENSITY,
    OPTIMIZER_METHOD,
)


def metrics_and_heatmaps(df, fitted: dict, A_true: np.ndarray, path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    metrics = [
        ("A_err", "||A_hat - A_true||_F   (lower better)"),
        ("auprc", "AUPRC   (HIGHER better)"),
        ("precision_at_k", "precision@k   (HIGHER better)"),
        ("sigma_err", "|sigma_hat - sigma_true|   (lower better)"),
    ]
    for ax, (col, title) in zip(axes[0], metrics):
        best = df[col].idxmin() if col in ("A_err", "sigma_err") else df[col].idxmax()
        colors = ["#1a9641" if i == best else "#4393c3" for i in df.index]
        ax.bar(df["combination"], df[col], color=colors)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.axhline(0, color="black", lw=0.8)

    vmax = np.abs(A_true).max() * 1.2
    panels = [("TRUE A", A_true)] + [(n, fitted[n]) for n in COMBINATIONS if n in fitted]
    for ax, (name, mat) in zip(axes[1:].ravel(), panels):
        ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(N_GENES):
            for j in range(N_GENES):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(mat[i, j]) > 0.6 * vmax else "black")
        ax.set_xticks(range(N_GENES)); ax.set_xticklabels([f"g{j}" for j in range(N_GENES)], fontsize=6)
        ax.set_yticks(range(N_GENES)); ax.set_yticklabels([f"g{i}" for i in range(N_GENES)], fontsize=6)
        ax.set_title(name, fontsize=9, fontweight="bold" if name == "TRUE A" else "normal")
    for ax in axes[1:].ravel()[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(f"Loss-combination comparison on single-gene knockout data  "
                 f"(gene_{KO_GENE} KO, {N_GENES} genes, density={NETWORK_DENSITY}, "
                 f"fitting full A + sigma, mu known;  {OPTIMIZER_METHOD}, maxiter={MAXITER})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grn_networks(df, fitted: dict, A_true: np.ndarray, true_off: np.ndarray,
                 k_edges: int, path):
    """
    Ground truth plus one panel per combination, all sharing the true
    network's node layout so the same gene sits in the same place everywhere.
    Each recovered net is thresholded to its top-k edges by |A_hat_ij| -- the
    same rule precision@k scores -- so picture and metric agree.
    """
    pos = shared_layout(A_true)
    scale = float(np.abs(true_off).max())
    true_edges = true_edge_set(A_true)
    names = [n for n in COMBINATIONS if n in fitted]

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    ax_list = axes.ravel()

    s = draw_grn(ax_list[0], A_true, pos, threshold=1e-9,
                 highlight=KO_GENE, scale_by=scale)
    ax_list[0].set_title(f"GROUND TRUTH\n{s['n_edges']} edges", fontsize=10, fontweight="bold")

    for ax, name in zip(ax_list[1:], names):
        row = df[df["combination"] == name].iloc[0]
        st = draw_grn(ax, fitted[name], pos, top_k=k_edges, highlight=KO_GENE,
                      true_edges=true_edges, scale_by=scale)
        ax.set_title(f"{name}\n{st['n_correct']}/{st['n_edges']} correct  "
                     f"AUPRC {row['auprc']:.2f}  P@{k_edges} {row['precision_at_k']:.2f}",
                     fontsize=9)

    for ax in ax_list[len(names) + 1:]:
        ax.set_visible(False)

    legend = [
        plt.Line2D([], [], color=ACTIVATE, lw=3, label="activation  (A_ij > 0)"),
        plt.Line2D([], [], color=INHIBIT, lw=3, label="inhibition  (A_ij < 0)"),
        plt.Line2D([], [], color="#4A544F", lw=2, ls=(0, (3, 2)), alpha=.6, label="false positive"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="#F2D7C9",
                   markeredgecolor="black", markeredgewidth=2, markersize=11,
                   label=f"knocked-out gene (g{KO_GENE})"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(
        f"Recovered GRN structure vs ground truth  (gene_{KO_GENE} KO, {N_GENES} genes, "
        f"density={NETWORK_DENSITY}; each panel thresholded to its top {k_edges} edges; "
        f"arrow j→i means gene j regulates gene i)", fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
