"""
The two recovery figures, parameterised so interventional and non-interventional
runs share one implementation.

  metrics_and_heatmaps -> metric bars + recovered A as heatmaps: are the NUMBERS right
  grn_networks         -> recovered regulatory networks as graphs: is the STRUCTURE right

Previously these lived in the knockout folder with the knocked-out gene and the
scenario title baked in as module globals, which made them unusable anywhere
else. Everything scenario-specific is now an argument; `highlight` is None for
datasets with no perturbed gene.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multsc_grn_inference.housekeeping.grn_graph import (
    ACTIVATE, INHIBIT, draw_grn, shared_layout, true_edge_set,
)

METRIC_PANELS = [
    ("A_err", "||A_hat - A_true||_F   (lower better)", "min"),
    ("auprc", "AUPRC   (HIGHER better)", "max"),
    ("precision_at_k", "precision@k   (HIGHER better)", "max"),
    ("sigma_err", "|sigma_hat - sigma_true|   (lower better)", "min"),
]


def metrics_and_heatmaps(df, fitted: dict, A_true: np.ndarray, path: Path, *,
                         title: str = "", chance: float | None = None):
    """chance, if given, draws the AUPRC/precision@k baseline as a dashed line --
    without it a bar of 0.55 looks like a result rather than a coin flip."""
    G = A_true.shape[0]
    names = [n for n in fitted]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for ax, (col, sub, better) in zip(axes[0], METRIC_PANELS):
        best = df[col].idxmin() if better == "min" else df[col].idxmax()
        colors = ["#1a9641" if i == best else "#4393c3" for i in df.index]
        ax.bar(df["combination"], df[col], color=colors)
        if chance is not None and col in ("auprc", "precision_at_k"):
            ax.axhline(chance, color="#C2601F", ls="--", lw=1.4)
            ax.text(0.98, chance, f" chance {chance:.2f}", color="#C2601F",
                    fontsize=7, va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.set_title(sub, fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.axhline(0, color="black", lw=.8)

    vmax = np.abs(A_true).max() * 1.2
    panels = [("TRUE A", A_true)] + [(n, fitted[n]) for n in names]
    for ax, (name, mat) in zip(axes[1:].ravel(), panels):
        ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(G):
            for j in range(G):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(mat[i, j]) > .6 * vmax else "black")
        ax.set_xticks(range(G)); ax.set_xticklabels([f"g{j}" for j in range(G)], fontsize=6)
        ax.set_yticks(range(G)); ax.set_yticklabels([f"g{i}" for i in range(G)], fontsize=6)
        ax.set_title(name, fontsize=9, fontweight="bold" if name == "TRUE A" else "normal")
    for ax in axes[1:].ravel()[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(title or "Loss-combination comparison")
    fig.tight_layout()
    fig.savefig(Path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def grn_networks(df, fitted: dict, A_true: np.ndarray, true_off: np.ndarray,
                 k_edges: int, path: Path, *, title: str = "",
                 highlight: int | None = None, highlight_label: str = ""):
    """
    Ground truth plus one panel per combination, all sharing the true network's
    node layout so the same gene sits in the same place everywhere. Each
    recovered net is thresholded to its top-k edges by |A_hat_ij| -- the same
    rule precision@k scores -- so picture and metric agree by construction.
    """
    pos = shared_layout(A_true)
    scale = float(np.abs(true_off).max())
    true_edges = true_edge_set(A_true)
    names = [n for n in fitted]

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    ax_list = axes.ravel()

    s = draw_grn(ax_list[0], A_true, pos, threshold=1e-9, highlight=highlight, scale_by=scale)
    ax_list[0].set_title(f"GROUND TRUTH\n{s['n_edges']} edges", fontsize=10, fontweight="bold")

    for ax, name in zip(ax_list[1:], names):
        row = df[df["combination"] == name].iloc[0]
        st = draw_grn(ax, fitted[name], pos, top_k=k_edges, highlight=highlight,
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
    ]
    if highlight is not None:
        legend.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="#F2D7C9",
                                 markeredgecolor="black", markeredgewidth=2, markersize=11,
                                 label=highlight_label or f"highlighted gene (g{highlight})"))
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=10, frameon=False)
    fig.suptitle(title or f"Recovered GRN structure vs ground truth "
                          f"(top {k_edges} edges; arrow j→i means gene j regulates gene i)",
                 fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(Path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
