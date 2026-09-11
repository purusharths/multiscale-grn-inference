"""
Draw a GRN as a directed graph -- ground truth and recovered side by side.

Companion to edge_recovery_metrics.py: that module scores edge recovery as a
number, this one shows WHICH edges were recovered. A heatmap of A answers
"are the magnitudes right"; a graph answers "did we find the right regulatory
structure", which is the question the inference exists to answer.

Edge direction convention. ou_gene_expression computes the drift as
    drift = (mu_k - c) @ A.T      i.e.  dc_i/dt = sum_j A[i,j] (mu_j - c_j)
so gene i's dynamics depend on gene j's displacement with weight A[i,j].
Gene j therefore REGULATES gene i, and the edge runs j -> i. Reading A as a
picture (row i, column j) transposes this, which is an easy way to draw every
arrow backwards -- hence to_digraph() below owning the convention in one place.

Thresholding a recovered A. A fitted A is dense: every off-diagonal is some
small nonzero number, so "the recovered network" is meaningless without a
cutoff. `top_k` keeps the k largest |A_ij|, which is the same rule
precision_at_k scores, so the picture and the metric agree by construction.
Passing k = (number of true edges) asks the fairest question: given that the
method is allowed exactly as many edges as really exist, which does it pick?
"""
from __future__ import annotations

import numpy as np

# Activation / inhibition. Chosen to stay distinguishable in greyscale print
# (the teal is darker than the orange) and for red-green colour blindness.
ACTIVATE = "#1B7C6F"
INHIBIT = "#C2601F"


def off_diag_mask(n: int) -> np.ndarray:
    """Boolean (n, n) mask, True off the diagonal."""
    return ~np.eye(n, dtype=bool)


def to_digraph(
    A: np.ndarray,
    *,
    top_k: int | None = None,
    threshold: float = 0.0,
):
    """
    Directed graph of A's off-diagonal structure, edge j -> i for A[i, j].

    top_k     : keep only the k largest |A[i, j]| (None = keep all above
                `threshold`). Use k = number of true edges for a like-for-like
                comparison against ground truth.
    threshold : minimum |A[i, j]| to draw. Ground truth is exactly sparse, so
                a tiny value (1e-9) selects its real edges.

    Self-loops (the diagonal) are never drawn -- every gene has self-decay, so
    they carry no structural information and only clutter the picture.
    """
    import networkx as nx

    n = A.shape[0]
    mask = off_diag_mask(n)
    idx = np.argwhere(mask)
    weights = np.array([A[i, j] for i, j in idx])

    keep = np.abs(weights) > threshold
    if top_k is not None:
        order = np.argsort(-np.abs(weights))
        chosen = np.zeros(len(weights), dtype=bool)
        chosen[order[:top_k]] = True
        keep &= chosen

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (i, j), w in zip(idx[keep], weights[keep]):
        G.add_edge(int(j), int(i), weight=float(w))   # j regulates i
    return G


def shared_layout(A_true: np.ndarray, *, seed: int = 42):
    """
    Node positions from the TRUE network, to be reused for every recovered
    panel. Without this each panel gets its own spring layout and the figures
    cannot be compared by eye -- the same gene would sit somewhere different
    in every picture.
    """
    import networkx as nx

    return nx.spring_layout(to_digraph(A_true, threshold=1e-9).to_undirected(),
                            seed=seed, k=1.4)


def draw_grn(
    ax,
    A: np.ndarray,
    pos: dict,
    *,
    title: str = "",
    top_k: int | None = None,
    threshold: float = 0.0,
    highlight: int | None = None,
    true_edges: set[tuple[int, int]] | None = None,
    max_width: float = 4.0,
    scale_by: float | None = None,
) -> dict:
    """
    Draw one GRN panel. Returns {"n_edges", "n_correct"}.

    highlight   : node index to ring in black (the knocked-out gene).
    true_edges  : if given, edges NOT in this set are drawn dashed and pale --
                  so false positives are visible as false positives rather
                  than blending in with correct recoveries.
    scale_by    : |weight| mapped to max_width at this value. Pass the true
                  network's max |edge| so every panel shares one width scale;
                  otherwise a panel whose edges are all tiny would draw them
                  as thick as the true network's strongest.
    """
    import networkx as nx

    G = to_digraph(A, top_k=top_k, threshold=threshold)
    edges = list(G.edges(data=True))
    denom = scale_by if scale_by else max((abs(d["weight"]) for *_, d in edges), default=1.0)

    node_colors = ["#E8EAE9"] * A.shape[0]
    edgecolors = ["#4A544F"] * A.shape[0]
    linewidths = [1.0] * A.shape[0]
    if highlight is not None:
        node_colors[highlight] = "#F2D7C9"
        edgecolors[highlight] = "#000000"
        linewidths[highlight] = 2.2

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=560, node_color=node_colors,
                           edgecolors=edgecolors, linewidths=linewidths)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            labels={i: f"g{i}" for i in G.nodes})

    n_correct = 0
    for u, v, d in edges:
        w = d["weight"]
        correct = true_edges is None or (u, v) in true_edges
        n_correct += int(correct)
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            width=0.6 + max_width * min(abs(w) / denom, 1.0),
            edge_color=ACTIVATE if w > 0 else INHIBIT,
            style="solid" if correct else (0, (3, 2)),
            alpha=1.0 if correct else 0.45,
            arrowsize=11, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.13",
            node_size=560,
        )

    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    return {"n_edges": len(edges), "n_correct": n_correct}


def true_edge_set(A_true: np.ndarray, *, tol: float = 1e-9) -> set[tuple[int, int]]:
    """{(j, i)} for every real edge, in the same j -> i orientation
    to_digraph uses -- so membership tests line up with drawn edges."""
    n = A_true.shape[0]
    return {(j, i) for i in range(n) for j in range(n)
            if i != j and abs(A_true[i, j]) > tol}
