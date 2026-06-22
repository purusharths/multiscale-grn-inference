"""
Wasserstein comparison: destructive measurements vs distributional measurements.

Answers two questions:
  1. How different are the distributions produced by each method at each
     collection timepoint?  (measured against a large reference ensemble)
  2. Is KDE a better distribution-fitting strategy than multivariate Gaussian
     for the distributional method?

Setup
-----
Reference (ground truth)
    N_REF-cell SDE ensemble; its empirical snapshot at each timepoint is the
    target distribution.

Destructive  (mirrors run_destructive_measurements)
    Subsample N_OUT cells without replacement from the reference snapshot —
    each trajectory used at most once.

Gaussian  (mirrors run_distribution_measurements)
    Independent N_ENS-cell ensemble (same dynamics, different seed) →
    fit multivariate Gaussian → draw N_OUT samples.

KDE  (alternative to Gaussian)
    Same N_ENS ensemble → Gaussian KDE (Scott bandwidth) → draw N_OUT samples.

Distance
    ot.sliced_wasserstein_distance (200 random projections) for the
    multivariate comparison; scipy's wasserstein_distance per gene for
    marginal inspection.

KDE vs Gaussian
---------------
Gaussian is fast and exact for the first two moments but forces a symmetric,
unimodal shape. During sharp transitions (heaviside, sigmoid) the snapshot
cloud can be skewed or have a heavier tail toward zero (from the non-negativity
floor). KDE adapts to any shape but requires enough cells; with 4 genes and
N_ENS=5000 it should be well-estimated. Expect KDE to win during transitions
and be roughly equivalent at equilibrium.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance

from dataset_gen_dynamic.housekeeping.enforce_diagonal_dominance import (
    enforce_diagonal_dominance,
)
from dataset_gen_dynamic.housekeeping.mu_options import (
    mu_constant,
    mu_heaviside,
    mu_linear,
    mu_sigmoid,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_GENES = 4
N_REF = 10_000   # large reference ensemble — treated as ground truth
N_ENS = 5_000    # smaller ensemble available to the distributional methods
N_OUT = 1_000    # output cells per timepoint (all three methods)
N_PROJ = 200     # sliced-Wasserstein random projections

COLLECTION_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

POPULATIONS = [
    {
        "label":         "pop_0",
        "seed":          10,
        "gene_mu_modes": ["constant",  "constant",  "constant", "constant"],
        "gene_mu_kwargs":[{},          {},          {},         {}],
    },
    {
        "label":         "pop_1",
        "seed":          11,
        "gene_mu_modes": ["sigmoid",   "constant",  "constant", "constant"],
        "gene_mu_kwargs":[{},          {},          {},         {}],
    },
    {
        "label":         "pop_2",
        "seed":          12,
        "gene_mu_modes": ["constant",  "heaviside", "constant", "constant"],
        "gene_mu_kwargs":[{},          {},          {},         {}],
    },
    {
        "label":         "pop_3",
        "seed":          13,
        "gene_mu_modes": ["constant",  "constant",  "linear",   "constant"],
        "gene_mu_kwargs":[{},          {},          {},         {}],
    },
    {
        "label":         "pop_4",
        "seed":          14,
        "gene_mu_modes": ["constant",  "constant",  "constant", "tanh"],
        "gene_mu_kwargs":[{},          {},          {},         {}],
    },
]

# ---------------------------------------------------------------------------
# mu helpers (same as both run_* files)
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
# SDE simulator (simulate_ensemble only — no full trajectory storage)
# ---------------------------------------------------------------------------

class _Sim:
    def __init__(
        self,
        num_genes: int,
        network_density: float,
        seed: int,
        gene_mu_modes: list[str],
        gene_mu_kwargs: list[dict],
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.num_genes = num_genes
        self.gene_mu_modes = gene_mu_modes
        self.gene_mu_kwargs = gene_mu_kwargs

        self.A = np.zeros((num_genes, num_genes))
        for i in range(num_genes):
            self.A[i, i] = rng.uniform(1.0, 1.8)
        n_edges = int(network_density * num_genes * (num_genes - 1))
        added = 0
        while added < n_edges:
            i, j = rng.choice(num_genes, 2, replace=False)
            if self.A[i, j] == 0:
                self.A[i, j] = (1 if rng.random() > 0.3 else -1) * rng.uniform(0.3, 1.2)
                added += 1
        enforce_diagonal_dominance(self.A)

        self.mu0 = rng.uniform(2.5, 3.5, num_genes)
        self.D = np.diag(rng.uniform(0.1, 0.2, num_genes))
        self.mu_drift_dir = rng.normal(size=num_genes)
        self.mu_drift_amp = 4.0

    def _mu_t(self, t: float, T: float) -> np.ndarray:
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
                raise ValueError(mode)
        return mu

    def simulate_ensemble(
        self,
        T: float,
        num_cells: int,
        collection_times: list[float],
        dt: float = 0.005,
    ) -> dict[float, np.ndarray]:
        """Return {t: (num_cells, num_genes)} snapshots at each collection time."""
        collection_times = sorted(set(collection_times))
        snap_steps = {round(t / dt): t for t in collection_times}
        steps = int(T / dt)
        snapshots: dict[float, list] = {t: [] for t in collection_times}

        for _ in range(num_cells):
            mu_init = self._mu_t(0.0, T)
            X = np.maximum(mu_init + 0.5 * self.rng.normal(size=self.num_genes), 0.1)
            if 0 in snap_steps:
                snapshots[snap_steps[0]].append(X.copy())
            for step in range(1, steps + 1):
                mu = self._mu_t(step * dt, T)
                drift = self.A @ (mu - X)
                noise = np.sqrt(np.diag(self.D) * dt) * self.rng.normal(size=self.num_genes)
                X = np.maximum(X + drift * dt + noise, 0.05)
                if step in snap_steps:
                    snapshots[snap_steps[step]].append(X.copy())

        return {t: np.array(cells) for t, cells in snapshots.items()}


# ---------------------------------------------------------------------------
# Sampling / fitting helpers
# ---------------------------------------------------------------------------

def destructive_sample(
    snapshot: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick n cells without replacement — each trajectory observed once."""
    return snapshot[rng.choice(len(snapshot), size=n, replace=False)]


def gaussian_sample(
    snapshot: np.ndarray,
    n: int,
    rng: np.random.Generator,
    reg: float = 1e-4,
) -> np.ndarray:
    mean = snapshot.mean(axis=0)
    cov = np.cov(snapshot, rowvar=False) + reg * np.eye(snapshot.shape[1])
    return np.maximum(rng.multivariate_normal(mean, cov, size=n), 0.0)


def kde_sample(
    snapshot: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    kde = gaussian_kde(snapshot.T)  # Scott bandwidth
    seed_val = int(rng.integers(0, 2**31))
    return np.maximum(kde.resample(n, seed=seed_val).T, 0.0)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def sliced_w(A: np.ndarray, B: np.ndarray) -> float:
    return float(ot.sliced_wasserstein_distance(A, B, n_projections=N_PROJ, seed=0))


def marginal_w(A: np.ndarray, B: np.ndarray) -> list[float]:
    return [wasserstein_distance(A[:, g], B[:, g]) for g in range(A.shape[1])]


# ---------------------------------------------------------------------------
# Comparison loop
# ---------------------------------------------------------------------------

def run_comparison() -> pd.DataFrame:
    rows: list[dict] = []

    for pop in POPULATIONS:
        label, seed = pop["label"], pop["seed"]
        print(f"\n[{label}]  simulating reference ({N_REF} cells) ...", flush=True)

        sim_ref = _Sim(
            NUM_GENES, 0.3, seed,
            pop["gene_mu_modes"], pop["gene_mu_kwargs"],
        )
        ref_snaps = sim_ref.simulate_ensemble(
            T=max(COLLECTION_TIMES), num_cells=N_REF,
            collection_times=COLLECTION_TIMES,
        )

        print(f"[{label}]  simulating fitting ensemble ({N_ENS} cells) ...", flush=True)
        # Different seed → independent trajectories, same underlying dynamics
        sim_ens = _Sim(
            NUM_GENES, 0.3, seed + 500,
            pop["gene_mu_modes"], pop["gene_mu_kwargs"],
        )
        ens_snaps = sim_ens.simulate_ensemble(
            T=max(COLLECTION_TIMES), num_cells=N_ENS,
            collection_times=COLLECTION_TIMES,
        )

        rng = np.random.default_rng(seed + 999)

        for t in sorted(COLLECTION_TIMES):
            ref = ref_snaps[t]   # (N_REF, NUM_GENES) — ground truth
            ens = ens_snaps[t]   # (N_ENS,  NUM_GENES) — available to fitters

            method_samples: dict[str, np.ndarray] = {
                "destructive": destructive_sample(ref, N_OUT, rng),
                "gaussian":    gaussian_sample(ens, N_OUT, rng),
                "kde":         kde_sample(ens, N_OUT, rng),
            }

            for method, s in method_samples.items():
                sw = sliced_w(ref, s)
                gws = marginal_w(ref, s)
                row: dict = {
                    "population": label,
                    "time": t,
                    "method": method,
                    "sliced_wasserstein": sw,
                }
                for g, gw in enumerate(gws):
                    row[f"gene_{g}_w"] = gw
                rows.append(row)

            print(f"  t={t:.1f}  ok", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_STYLE = {
    "destructive": dict(color="#2166ac", ls="-",  lw=2.0, label="Destructive (subsample ref)"),
    "gaussian":    dict(color="#d6604d", ls="--", lw=1.8, label="Distributional — Gaussian"),
    "kde":         dict(color="#1a9641", ls=":",  lw=1.8, label="Distributional — KDE"),
}


def _plot_over_time(df: pd.DataFrame, out_dir: Path) -> None:
    pops = df["population"].unique()
    fig, axes = plt.subplots(1, len(pops), figsize=(4 * len(pops), 4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, pop in zip(axes, pops):
        for method, sty in _STYLE.items():
            sub = df[(df["population"] == pop) & (df["method"] == method)].sort_values("time")
            ax.plot(sub["time"], sub["sliced_wasserstein"],
                    marker="o", markersize=3, **sty)
        ax.set_title(pop, fontsize=10)
        ax.set_xlabel("Time")
        if ax is axes[0]:
            ax.set_ylabel("Sliced Wasserstein")
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Sliced Wasserstein vs reference  (lower = closer to ground truth)", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = out_dir / "wasserstein_over_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_kde_vs_gaussian(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of (W_gaussian − W_kde) / W_gaussian per timepoint per population."""
    pops = df["population"].unique()
    times = sorted(df["time"].unique())

    delta_rows = []
    for pop in pops:
        sub = df[df["population"] == pop]
        for t in times:
            gv = sub[(sub["method"] == "gaussian") & (sub["time"] == t)]["sliced_wasserstein"].values
            kv = sub[(sub["method"] == "kde")      & (sub["time"] == t)]["sliced_wasserstein"].values
            if gv.size and kv.size:
                delta_rows.append({"population": pop, "time": t,
                                   "rel_improvement": (gv[0] - kv[0]) / gv[0]})

    delta_df = pd.DataFrame(delta_rows)
    fig, axes = plt.subplots(1, len(pops), figsize=(4 * len(pops), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, pop in zip(axes, pops):
        sub = delta_df[delta_df["population"] == pop].sort_values("time")
        colors = ["#1a9641" if v > 0 else "#d6604d" for v in sub["rel_improvement"]]
        ax.bar(sub["time"], sub["rel_improvement"], width=0.35, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(pop, fontsize=10)
        ax.set_xlabel("Time")
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("(W_gauss − W_kde) / W_gauss")
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color="#1a9641", label="KDE better"),
                 Patch(color="#d6604d", label="Gaussian better")],
        loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("KDE vs Gaussian: relative improvement in Wasserstein distance", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = out_dir / "kde_vs_gaussian_improvement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: mean sliced Wasserstein across all timepoints."""
    pops = list(df["population"].unique())
    methods = list(_STYLE.keys())
    x = np.arange(len(pops))
    width = 0.25

    summary = df.groupby(["population", "method"])["sliced_wasserstein"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, method in enumerate(methods):
        vals = [summary[pop][method] for pop in pops]
        ax.bar(x + (i - 1) * width, vals, width,
               color=_STYLE[method]["color"], label=_STYLE[method]["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(pops)
    ax.set_ylabel("Mean sliced Wasserstein (all timepoints)")
    ax.set_title("Summary: mean Wasserstein distance from ground-truth distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "summary_mean_wasserstein.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_density_shapes(out_dir: Path) -> None:
    """
    Gene-by-gene marginal density for pop_2 (heaviside, sharpest transition)
    at t=2.0 (mid-point). Visually shows whether Gaussian or KDE better
    captures the snapshot shape.
    """
    pop = next(p for p in POPULATIONS if p["label"] == "pop_2")
    t_target = 2.0
    seed = pop["seed"]

    sim_ref = _Sim(NUM_GENES, 0.3, seed, pop["gene_mu_modes"], pop["gene_mu_kwargs"])
    ref_snap = sim_ref.simulate_ensemble(
        T=max(COLLECTION_TIMES), num_cells=N_REF,
        collection_times=[t_target],
    )[t_target]

    sim_ens = _Sim(NUM_GENES, 0.3, seed + 500, pop["gene_mu_modes"], pop["gene_mu_kwargs"])
    ens_snap = sim_ens.simulate_ensemble(
        T=max(COLLECTION_TIMES), num_cells=N_ENS,
        collection_times=[t_target],
    )[t_target]

    rng = np.random.default_rng(seed + 999)
    dest_s  = destructive_sample(ref_snap, N_OUT, rng)
    gauss_s = gaussian_sample(ens_snap, N_OUT, rng)
    kde_s   = kde_sample(ens_snap, N_OUT, rng)

    fig, axes = plt.subplots(1, NUM_GENES, figsize=(4 * NUM_GENES, 3.5))
    for g, ax in enumerate(axes):
        all_vals = np.concatenate([ref_snap[:, g], gauss_s[:, g], kde_s[:, g]])
        xs = np.linspace(all_vals.min(), all_vals.max(), 300)

        def _kde_curve(data, g=g):
            return gaussian_kde(data[:, g])(xs)

        ax.fill_between(xs, _kde_curve(ref_snap), alpha=0.15, color="#2166ac")
        ax.plot(xs, _kde_curve(ref_snap),  color="#2166ac", lw=1.8, label="Reference")
        ax.plot(xs, _kde_curve(dest_s),    color="#2166ac", lw=1.0, ls="--", label="Destructive")
        ax.plot(xs, _kde_curve(gauss_s),   color="#d6604d", lw=1.8, ls="--", label="Gaussian")
        ax.plot(xs, _kde_curve(kde_s),     color="#1a9641", lw=1.8, ls=":",  label="KDE")

        w_dest  = wasserstein_distance(ref_snap[:, g], dest_s[:, g])
        w_gauss = wasserstein_distance(ref_snap[:, g], gauss_s[:, g])
        w_kde   = wasserstein_distance(ref_snap[:, g], kde_s[:, g])
        ax.set_title(
            f"gene_{g}\nW: dest={w_dest:.3f}  gauss={w_gauss:.3f}  kde={w_kde:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("Expression")
        if g == 0:
            ax.set_ylabel("Density")
        ax.tick_params(labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        f"pop_2 (heaviside on gene_1) at t={t_target} — marginal density comparison",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = out_dir / "density_shapes_pop2_t2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("output-wasserstein-comparison/")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference ensemble : {N_REF} cells")
    print(f"Fitting ensemble   : {N_ENS} cells  (independent seed)")
    print(f"Output per timepoint: {N_OUT} cells")
    print(f"Sliced-W projections: {N_PROJ}")

    df = run_comparison()
    df.to_csv(out_dir / "wasserstein_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'wasserstein_results.csv'}")

    print("\nGenerating plots ...")
    _plot_over_time(df, out_dir)
    _plot_kde_vs_gaussian(df, out_dir)
    _plot_summary(df, out_dir)
    _plot_density_shapes(out_dir)

    # ---- console summary ----
    print("\n=== Mean sliced Wasserstein across all timepoints ===")
    summary = (
        df.groupby(["population", "method"])["sliced_wasserstein"]
        .mean()
        .unstack("method")
        .round(4)
    )
    print(summary.to_string())

    print("\n=== KDE vs Gaussian verdict ===")
    for pop in df["population"].unique():
        sub = df[df["population"] == pop]
        g = sub[sub["method"] == "gaussian"]["sliced_wasserstein"].mean()
        k = sub[sub["method"] == "kde"]["sliced_wasserstein"].mean()
        winner = "KDE" if k < g else "Gaussian"
        pct = abs(g - k) / g * 100
        print(f"  {pop}: {winner} wins by {pct:.1f}%  (gaussian={g:.4f}, kde={k:.4f})")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
