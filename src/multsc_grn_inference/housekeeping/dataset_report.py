"""
Describe the dataset a run was performed on: one figure, plus the numbers that
belong in the run's README.

Every diagnostic run should carry a picture of what it was run against, so a
result can be read months later without re-deriving the dataset from config
constants. describe() returns both the figure path and a stats dict for the
README table.

Reusable across interventional and non-interventional runs -- pass mu_t for a
time-varying target and it is drawn per gene; omit it for stationary data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Cells per snapshot fed to UMAP. The full set (n_cells x n_snaps) is far more
# than UMAP needs to show the shape and makes the run noticeably slower.
UMAP_PER_SNAPSHOT = 300


def dataset_stats(sim, snapshots, times, *, dt: float) -> dict:
    """The numbers worth recording about a dataset."""
    A = sim.A
    G = A.shape[0]
    off = A[~np.eye(G, dtype=bool)]
    nz = np.abs(off[np.abs(off) > 1e-9])
    eig = np.linalg.eigvals(A)
    sigma = float(np.sqrt(np.diag(sim.D).mean()))
    return {
        "genes": G,
        "cells per snapshot": len(snapshots[0]),
        "snapshots": len(snapshots),
        "dt": f"{dt:.3f}",
        "total time T": f"{times[-1]:.2f}",
        "true edges": int((np.abs(off) > 1e-9).sum()),
        "possible edges": G * G - G,
        "edge density (realised)": f"{(np.abs(off) > 1e-9).mean():.3f}",
        "mean |edge|": f"{nz.mean():.3f}" if nz.size else "0",
        "max |edge|": f"{nz.max():.3f}" if nz.size else "0",
        "sigma (noise)": f"{sigma:.3f}",
        "edge / sigma": f"{nz.mean() / sigma:.2f}" if nz.size else "0",
        "eig(A) real range": f"{eig.real.min():.2f} .. {eig.real.max():.2f}",
        "stable (all Re>0)": bool(np.all(eig.real > 0)),
        "oscillatory modes": bool(np.any(np.abs(eig.imag) > 1e-9)),
        "slowest relax time": f"{1 / eig.real.min():.3f}",
        "dt / slowest tau": f"{dt * eig.real.min():.2f}",
    }


def describe(sim, snapshots, times, out_path: Path, *, dt: float,
             mu_t=None, title: str = "", umap_seed: int = 42) -> tuple[Path, dict]:
    """
    Four panels: UMAP over time, PCA mean trajectory, per-gene means, true A.
    Returns (figure path, stats dict).
    """
    G = sim.A.shape[0]
    n_snaps = len(snapshots)
    stats = dataset_stats(sim, snapshots, times, dt=dt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax_umap, ax_pca), (ax_genes, ax_A) = axes

    # ---- UMAP: does the population move as a blob, or split? ----------------
    rng = np.random.default_rng(umap_seed)
    take = min(UMAP_PER_SNAPSHOT, len(snapshots[0]))
    sub = [s[rng.choice(len(s), take, replace=False)] for s in snapshots]
    stacked = np.vstack(sub)
    idx = np.repeat(np.arange(n_snaps), take)
    try:
        import umap
        emb = umap.UMAP(n_components=2, random_state=umap_seed).fit_transform(stacked)
        label = "UMAP"
    except Exception as exc:                      # umap-learn optional/fragile
        emb = PCA(n_components=2, random_state=umap_seed).fit_transform(stacked)
        label = f"PCA (UMAP unavailable: {type(exc).__name__})"
    sc = ax_umap.scatter(emb[:, 0], emb[:, 1], c=idx, cmap="viridis", s=6, alpha=.65)
    plt.colorbar(sc, ax=ax_umap, label="snapshot index")
    ax_umap.set_title(f"{label} — {take} cells/snapshot, coloured by time", fontsize=10)
    ax_umap.set_xlabel(f"{label.split()[0]}-1"); ax_umap.set_ylabel(f"{label.split()[0]}-2")

    # ---- PCA mean trajectory: whole dynamics, all genes jointly -------------
    # PCA is linear, so mean-of-projection == projection-of-mean; the mean
    # trajectory drawn here really is the population mean's path.
    pca = PCA(n_components=2, random_state=umap_seed).fit(np.vstack(snapshots))
    full = pca.transform(np.vstack(sub))
    ax_pca.scatter(full[:, 0], full[:, 1], c=idx, cmap="Greys", s=4, alpha=.2)
    means = np.array([pca.transform(s).mean(axis=0) for s in snapshots])
    ax_pca.plot(means[:, 0], means[:, 1], "-o", color="#C2601F", lw=2, ms=5, label="population mean")
    ax_pca.scatter(*means[0], marker="s", s=110, color="#1B7C6F", zorder=6, label="t0 (perturbed start)")
    mu_p = pca.transform(sim.mu0[None, :])[0]
    ax_pca.scatter(*mu_p, marker="*", s=240, color="gold", edgecolors="black", zorder=7, label="mu")
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax_pca.set_title(f"Mean trajectory in PCA space (all {G} genes)", fontsize=10)
    ax_pca.legend(fontsize=8)

    # ---- per-gene relaxation ------------------------------------------------
    cmap = plt.get_cmap("tab10")
    for g in range(G):
        m = np.array([s[:, g].mean() for s in snapshots])
        sd = np.array([s[:, g].std() for s in snapshots])
        c = cmap(g % 10)
        ax_genes.fill_between(times, m - sd, m + sd, color=c, alpha=.12)
        ax_genes.plot(times, m, color=c, lw=1.6, marker="o", ms=3, label=f"g{g}")
        target = ([mu_t(t)[g] for t in times] if mu_t else [sim.mu0[g]] * len(times))
        ax_genes.plot(times, target, color=c, ls=":", lw=1.1, alpha=.7)
    ax_genes.set_xlabel("time"); ax_genes.set_ylabel("mean expression")
    ax_genes.set_title("Per-gene mean ± sd (dotted = mu target)", fontsize=10)
    ax_genes.legend(fontsize=7, ncol=2)

    # ---- the network being recovered ---------------------------------------
    vmax = np.abs(sim.A).max() * 1.1
    ax_A.imshow(sim.A, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(G):
        for j in range(G):
            v = sim.A[i, j]
            if abs(v) > 1e-9:
                ax_A.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                          color="white" if abs(v) > .6 * vmax else "black")
    ax_A.set_xticks(range(G)); ax_A.set_xticklabels([f"g{j}" for j in range(G)], fontsize=6)
    ax_A.set_yticks(range(G)); ax_A.set_yticklabels([f"g{i}" for i in range(G)], fontsize=6)
    ax_A.set_title(f"TRUE A — {stats['true edges']} edges, mean |edge| {stats['mean |edge|']}", fontsize=10)

    fig.suptitle(title or "Dataset overview", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, stats
