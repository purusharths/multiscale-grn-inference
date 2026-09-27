"""
Same question as compare_exact_ou.py -- does the exact OU transition
recover the network better than Euler-Maruyama, when actually re-fit --
but with CMA-ES instead of Nelder-Mead.

Why a separate script: compare_exact_ou.py's Nelder-Mead run showed the
exact-OU objective is dramatically more expensive AND inconsistent for
Nelder-Mead specifically (one quick check: combo "OU" took 13x more evals
than its Euler-Maruyama counterpart and scored worse; combo "FP" took 9x
more evals and scored close to as well). A separate quick check with
CMA-ES on combo "OU" alone told a cleaner story: CMA-ES + exact-OU beat
CMA-ES + Euler-Maruyama on both edge_corr (0.356 vs 0.260) and AUPRC
(0.554 vs 0.518) at a similar cost (~2200 evals either way) and a better
final loss value. That's consistent with ../loss-identifiability/'s
prediction (exact OU should recover better) actually holding -- provided
the optimizer can search its landscape, which Nelder-Mead evidently
struggles to do here even with the simplex fix.

This runs the full 6-combination sweep with CMA-ES on both sides, so the
comparison isn't resting on one combination.

Not a test -- a standalone report script; re-run anytime, overwrites this
folder's exact_ou_cma_comparison.{csv,png}.

Usage:
    uv run python tests/diagnostics/exact-ou-test/compare_exact_ou_cma.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.housekeeping.edge_recovery_metrics import (  # noqa: E402
    auprc,
    auroc,
    precision_at_k,
)

from _exact_combinations import fit_combination_cma_exact  # noqa: E402
from _intervention_ground_truth import (  # noqa: E402
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)
from _ours_combinations import (  # noqa: E402
    COMBINATIONS,
    fit_combination_cma,
    off_diag_indices,
    one_step_ahead_w2,
)

N_GENES = 8
KO_GENE = 3
N_CELLS = 1500
N_EVAL_CELLS = 500
N_SNAPS = 10
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.5
SEED = 42

N_PROJ = 30
CMA_MAXFEVALS = 2200
CMA_SIGMA0 = 0.15

HERE = Path(__file__).parent
OFF = off_diag_indices(N_GENES)


def _edge_metrics(true_off: np.ndarray, hat_off: np.ndarray) -> dict:
    k = int((true_off != 0).sum())
    return {
        "edge_corr": float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0,
        "auprc": auprc(true_off, hat_off),
        "auroc": auroc(true_off, hat_off),
        "precision_at_k": precision_at_k(true_off, hat_off, k),
    }


def main() -> None:
    sim = make_single_gene_knockout_sim(
        num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
        t_star=T_STAR, network_density=NETWORK_DENSITY,
    )
    snapshots, times, mu_known, dt = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)
    eval_snapshots, _, _, _ = extract_snapshots(sim, N_EVAL_CELLS, N_SNAPS, T=T)
    A_true = sim.A
    true_off = np.array([A_true[i, j] for i, j in OFF])

    rows = []
    A_hats: dict[str, np.ndarray] = {}

    for combo_name, terms in COMBINATIONS.items():
        for method_label, fit_fn in [
            ("CMA-ES + Euler-Maruyama", fit_combination_cma),
            ("CMA-ES + exact OU", fit_combination_cma_exact),
        ]:
            print(f"[{combo_name} / {method_label}] fitting ...", flush=True)
            theta_hat, fit_info = fit_fn(
                terms, snapshots, mu_known, dt, N_GENES,
                n_proj=N_PROJ, maxfevals=CMA_MAXFEVALS, sigma0=CMA_SIGMA0, seed=0,
            )
            hat_off = np.array([theta_hat.A[i, j] for i, j in OFF])
            w2 = one_step_ahead_w2(theta_hat, eval_snapshots, dt, n_proj=N_PROJ, seed=0)
            rows.append({
                "combination": combo_name, "method": method_label,
                "one_step_w2": w2,
                "A_err": float(np.linalg.norm(theta_hat.A - A_true)),
                "offdiag_err": float(np.linalg.norm(hat_off - true_off)),
                **_edge_metrics(true_off, hat_off),
                "sigma_err": abs(theta_hat.sigma - effective_sigma(sim)),
                **fit_info,
            })
            A_hats[f"{combo_name}__{method_label}"] = theta_hat.A
            print(f"  w2={w2:.4f}  A_err={rows[-1]['A_err']:.3f}  edge_corr={rows[-1]['edge_corr']:+.3f}  "
                  f"auprc={rows[-1]['auprc']:.3f}  evals={fit_info['n_evals']}  {fit_info['seconds']:.0f}s")

    df = pd.DataFrame(rows)
    csv_path = HERE / "exact_ou_cma_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df)
    _plot_matrices(A_true, A_hats)


def _plot(df: pd.DataFrame) -> None:
    combos = list(COMBINATIONS.keys())
    x = np.arange(len(combos))
    width = 0.35
    methods = ["CMA-ES + Euler-Maruyama", "CMA-ES + exact OU"]
    colors = ["#4393c3", "#1a9641"]

    fig, (ax_corr, ax_auprc, ax_w2) = plt.subplots(1, 3, figsize=(19, 5))

    for ax, col, title in [
        (ax_corr, "edge_corr", "edge_corr (higher better)"),
        (ax_auprc, "auprc", "AUPRC (higher better; chance=0.5)"),
        (ax_w2, "one_step_w2", "one-step-ahead sliced-W2 (lower better)"),
    ]:
        for i, (method, color) in enumerate(zip(methods, colors)):
            vals = [df[(df.combination == c) & (df.method == method)][col].iloc[0] for c in combos]
            ax.bar(x + (i - 0.5) * width, vals, width, label=method, color=color)
        ax.axhline(0.5 if col == "auprc" else 0, color="black", lw=0.8, ls="--" if col == "auprc" else "-")
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"CMA-ES: Euler-Maruyama vs exact OU transition -- single-gene knockout, "
        f"{N_GENES} genes, density={NETWORK_DENSITY}"
    )
    fig.tight_layout()
    png_path = HERE / "exact_ou_cma_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _plot_matrices(A_true: np.ndarray, A_hats: dict[str, np.ndarray]) -> None:
    combos = list(COMBINATIONS.keys())
    methods = ["TRUE", "CMA-ES + Euler-Maruyama", "CMA-ES + exact OU"]
    vmax = np.abs(A_true).max() * 1.2

    fig, axes = plt.subplots(len(methods), len(combos), figsize=(2.0 * len(combos), 2.2 * len(methods) + 0.5))

    im = None
    for row, method in enumerate(methods):
        for col, combo in enumerate(combos):
            ax = axes[row, col]
            mat = A_true if method == "TRUE" else A_hats[f"{combo}__{method}"]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(combo, fontsize=9)
            if col == 0:
                ax.set_ylabel(method, fontsize=8)

    fig.suptitle(f"Recovered A -- {N_GENES} genes, density={NETWORK_DENSITY}, knockout scenario", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="A_ij", pad=0.01)
    png_path = HERE / "exact_ou_cma_recovered_matrices.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
