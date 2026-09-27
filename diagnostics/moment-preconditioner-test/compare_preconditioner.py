"""
Does a closed-form moment-matching preconditioner get CMA-ES to a better fit,
faster, than its current generic diag(1.2)/all-zero-off-diagonal/sigma=0.5
starting guess?

Follows from ../non-interventional/: switching that script from Nelder-Mead
to CMA-ES (see ../cma-es-test/README.md for why) got real off-diagonal
structure back instead of a diagonal-only fit, but 3 of 6 loss combinations
still only barely beat chance, and none get close to the true A in magnitude
(A_err ~2.5-3.6 across the board). One candidate reason: CMA-ES still starts
completely blind to the data, at the same x0 regardless of what was actually
observed.

_moment_preconditioner.py computes a closed-form (A_hat, sigma_hat) from just
the snapshot means and covariances -- no optimizer, no sampling, effectively
free next to a single CMA-ES fit -- by matching the mean and covariance
trajectories to the linear SDE's own ODEs. See that module's docstring for
why this is "population-level moment matching", not a full per-particle
Kalman filter: it only ever uses per-snapshot sample statistics, which
respects the unpaired-snapshot assumption real single-cell data (and the
OU/FP/Cons losses) already make.

This script fits all 6 loss combinations twice, matched budget
(maxfevals=2200, sigma0=0.15 -- only x0 differs):
  - "cold"        x0 = diag(1.2), off-diagonal 0, sigma=0.5 (current default)
  - "preconditioned" x0 = the moment-matched (A_hat, sigma_hat)
against the SAME dataset as ../non-interventional/compare_losses_stationary.py
(8 genes, density=0.3, constant mu, seed=42), and also reports the
preconditioner's own (zero-optimizer-cost) accuracy as a reference point.

Usage:
    uv run python tests/diagnostics/moment-preconditioner-test/compare_preconditioner.py
    uv run python .../compare_preconditioner.py --jobs 12   # all 12 fits at once
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "interventions"))
sys.path.insert(0, str(HERE.parent / "non-interventional"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multsc_grn_inference.housekeeping.edge_recovery_metrics import (
    auprc, auroc, edge_labels, precision_at_k,
)

from _moment_preconditioner import estimate_theta0
from _ours_combinations import COMBINATIONS, encode, fit_combination_cma, off_diag_indices
from _stationary_ground_truth import build, effective_sigma

N_GENES = 8
N_CELLS = 3000
N_SNAPS = 15
DT = 0.5
NETWORK_DENSITY = 0.3
DATA_SEED = 42
FIT_SEED = 0
N_PROJ = 300
MAXFEVALS = 2200
SIGMA0 = 0.15

OFF = off_diag_indices(N_GENES)


def _edge_metrics(true_off: np.ndarray, hat_off: np.ndarray) -> dict:
    k = int(edge_labels(true_off).sum())
    return {
        "edge_corr": float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0,
        "auprc": auprc(true_off, hat_off),
        "auroc": auroc(true_off, hat_off),
        "precision_at_k": precision_at_k(true_off, hat_off, k),
    }


def fit_one(args):
    """Top-level so it survives fork/spawn -- see ../non-interventional/
    compare_losses_stationary.py's fit_one for why."""
    combo_name, variant, x0, sim_A, sigma_true, snapshots, mu_known = args
    true_off = np.array([sim_A[i, j] for i, j in OFF])

    t0 = time.perf_counter()
    theta_hat, info = fit_combination_cma(
        COMBINATIONS[combo_name], snapshots, mu_known, DT, N_GENES,
        n_proj=N_PROJ, maxfevals=MAXFEVALS, sigma0=SIGMA0, seed=FIT_SEED, x0=x0,
    )
    hat_off = np.array([theta_hat.A[i, j] for i, j in OFF])
    row = {
        "combination": combo_name, "variant": variant,
        "A_err": float(np.linalg.norm(theta_hat.A - sim_A)),
        "offdiag_err": float(np.linalg.norm(hat_off - true_off)),
        **_edge_metrics(true_off, hat_off),
        "sigma_err": abs(theta_hat.sigma - sigma_true),
        "final_objective": info["final_objective"],
        "n_evals": info["n_evals"],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return combo_name, variant, theta_hat.A, row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=1, help="fit this many (combo, variant) pairs concurrently (12 = all at once)")
    args = ap.parse_args()

    sim, snapshots, times, mu_known, dt = build(
        n_genes=N_GENES, n_cells=N_CELLS, n_snaps=N_SNAPS, dt=DT,
        seed=DATA_SEED, density=NETWORK_DENSITY,
    )
    sigma_true = effective_sigma(sim)
    true_off = np.array([sim.A[i, j] for i, j in OFF])
    chance = float(edge_labels(true_off).mean())

    A_pre, sigma_pre = estimate_theta0(snapshots, mu_known, dt, N_GENES)
    pre_off = np.array([A_pre[i, j] for i, j in OFF])
    pre_metrics = _edge_metrics(true_off, pre_off)
    print("Preconditioner alone (zero CMA-ES cost):")
    print(f"  A_err={np.linalg.norm(A_pre - sim.A):.3f}  offdiag_err={np.linalg.norm(pre_off - true_off):.3f}  "
          f"sigma_err={abs(sigma_pre - sigma_true):.3f}  edge_corr={pre_metrics['edge_corr']:+.3f}  "
          f"auprc={pre_metrics['auprc']:.3f}  (chance={chance:.3f})\n", flush=True)

    x0_cold = encode(np.eye(N_GENES) * 1.2, 0.5, OFF)
    x0_pre = encode(A_pre, sigma_pre, OFF)

    payload = [
        (combo, variant, x0, sim.A, sigma_true, snapshots, mu_known)
        for combo in COMBINATIONS
        for variant, x0 in [("cold", x0_cold), ("preconditioned", x0_pre)]
    ]

    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(fit_one, payload):
                print(f"[{r[0]} / {r[1]}] done  {r[3]['seconds']}s  AUPRC={r[3]['auprc']:.3f}", flush=True)
                results.append(r)
    else:
        for p in payload:
            print(f"[{p[0]} / {p[1]}] optimising ...", flush=True)
            r = fit_one(p)
            print(f"  AUPRC={r[3]['auprc']:.3f}  edge_corr={r[3]['edge_corr']:+.3f}  "
                  f"A_err={r[3]['A_err']:.3f}  {r[3]['seconds']}s", flush=True)
            results.append(r)

    df = pd.DataFrame([r[3] for r in results])
    A_hats = {f"{r[0]}__{r[1]}": r[2] for r in results}

    csv_path = HERE / "preconditioner_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot_bars(df, chance)
    _plot_matrices(sim.A, A_hats, A_pre)


def _plot_bars(df: pd.DataFrame, chance: float) -> None:
    combos = list(COMBINATIONS.keys())
    x = np.arange(len(combos))
    width = 0.35

    fig, (ax_corr, ax_auprc, ax_evals) = plt.subplots(1, 3, figsize=(19, 5))

    for ax, col, title, hline in [
        (ax_corr, "edge_corr", "edge_corr (higher better)", 0),
        (ax_auprc, "auprc", "AUPRC (higher better)", chance),
        (ax_evals, "seconds", "wall time to converge (s)", None),
    ]:
        cold = [df[(df.combination == c) & (df.variant == "cold")][col].iloc[0] for c in combos]
        pre = [df[(df.combination == c) & (df.variant == "preconditioned")][col].iloc[0] for c in combos]
        ax.bar(x - width / 2, cold, width, label="cold start", color="#4393c3")
        ax.bar(x + width / 2, pre, width, label="preconditioned", color="#1a9641")
        if hline is not None:
            ax.axhline(hline, color="black", lw=0.8, ls="--" if col == "auprc" else "-")
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"CMA-ES cold start vs moment-matching preconditioner -- "
        f"non-interventional, {N_GENES} genes, density={NETWORK_DENSITY}, matched budget"
    )
    fig.tight_layout()
    png_path = HERE / "preconditioner_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _plot_matrices(A_true: np.ndarray, A_hats: dict[str, np.ndarray], A_pre: np.ndarray) -> None:
    combos = list(COMBINATIONS.keys())
    rows = ["TRUE", "preconditioner only", "cold start", "preconditioned"]
    vmax = np.abs(A_true).max() * 1.2

    fig, axes = plt.subplots(len(rows), len(combos), figsize=(2.0 * len(combos), 2.2 * len(rows) + 0.5))

    im = None
    for row, label in enumerate(rows):
        for col, combo in enumerate(combos):
            ax = axes[row, col]
            if label == "TRUE":
                mat = A_true
            elif label == "preconditioner only":
                mat = A_pre
            else:
                variant = "cold" if label == "cold start" else "preconditioned"
                mat = A_hats[f"{combo}__{variant}"]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(combo, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    fig.suptitle(f"Recovered A -- {N_GENES} genes, density={NETWORK_DENSITY}, non-interventional", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="A_ij", pad=0.01)
    png_path = HERE / "preconditioner_recovered_matrices.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
