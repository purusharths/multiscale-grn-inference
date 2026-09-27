"""
Does CMA-ES recover the single-gene-knockout network better than Nelder-
Mead, at a matched evaluation budget? See README.md for the full chain that
led here: Nelder-Mead's default initial simplex was found to be degenerate
for our all-zero-off-diagonal starting point (fixed, in
../interventions/_ours_combinations.py's _initial_simplex), which helped
but plateaued -- edge_corr stuck around +0.36 even at 4x the evaluation
budget. This asks whether swapping the optimizer itself, rather than just
its budget, does better.

Dataset: the same single-gene-knockout scenario as ../jko-testing/ and
../sch-bridge-test/ (8 genes, gene_3 knocked out, density=0.5), knockout
scenario only -- this isn't about the mu_k question those two folders
cover, so the paired constant-mu control isn't needed here.

Both optimizers fit all 6 loss combinations from
../interventions/_ours_combinations.py, at a matched evaluation budget
(CMA-ES's maxfevals=2200 ~= Nelder-Mead's typical evals at maxiter=1500).
Scored on A_err/offdiag_err/edge_corr against the true A, and on one-step-
ahead sliced-W2 against the same independent eval set used everywhere else
in this branch's diagnostics.

Not a test -- a standalone report script; re-run anytime, overwrites this
folder's cma_es_comparison.{csv,png} and cma_es_recovered_matrices.png.

Usage:
    uv run python tests/diagnostics/cma-es-test/compare_cma_es.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.housekeeping.edge_recovery_metrics import (  # noqa: E402
    auprc,
    auroc,
    precision_at_k,
)

from _intervention_ground_truth import (  # noqa: E402
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)
from _ours_combinations import (  # noqa: E402
    COMBINATIONS,
    fit_combination,
    fit_combination_cma,
    off_diag_indices,
    one_step_ahead_w2,
)

# ---------------------------------------------------------------------------
# Config -- matches ../jko-testing/ and ../sch-bridge-test/ (knockout only,
# see README.md for why the control scenario isn't needed here)
# ---------------------------------------------------------------------------

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
NM_MAXITER = 1500
CMA_MAXFEVALS = 2200
CMA_SIGMA0 = 0.15

HERE = Path(__file__).parent
OFF = off_diag_indices(N_GENES)


def _edge_metrics(true_off: np.ndarray, hat_off: np.ndarray) -> dict:
    """edge_corr plus AUPRC/AUROC/precision@k -- see
    src/multsc_grn_inference/housekeeping/edge_recovery_metrics.py for why
    the latter three are the more trustworthy read on edge recovery, and
    the central metric for what this script is actually asking."""
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
        for method_label, fit_fn, fit_kwargs in [
            ("Nelder-Mead", fit_combination, {"maxiter": NM_MAXITER, "method": "Nelder-Mead"}),
            ("CMA-ES", fit_combination_cma, {"maxfevals": CMA_MAXFEVALS, "sigma0": CMA_SIGMA0}),
        ]:
            print(f"[{combo_name} / {method_label}] fitting ...", flush=True)
            theta_hat, fit_info = fit_fn(
                terms, snapshots, mu_known, dt, N_GENES, n_proj=N_PROJ, seed=0, **fit_kwargs,
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
    csv_path = HERE / "cma_es_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df)
    _plot_matrices(A_true, A_hats)


def _plot(df: pd.DataFrame) -> None:
    combos = list(COMBINATIONS.keys())
    x = np.arange(len(combos))
    width = 0.35

    fig, (ax_corr, ax_auprc, ax_w2) = plt.subplots(1, 3, figsize=(19, 5))

    for ax, col, title, better in [
        (ax_corr, "edge_corr", "edge_corr (higher better)", "high"),
        (ax_auprc, "auprc", "AUPRC (higher better; chance=0.5)", "high"),
        (ax_w2, "one_step_w2", "one-step-ahead sliced-W2 (lower better)", "low"),
    ]:
        nm = [df[(df.combination == c) & (df.method == "Nelder-Mead")][col].iloc[0] for c in combos]
        cma = [df[(df.combination == c) & (df.method == "CMA-ES")][col].iloc[0] for c in combos]
        ax.bar(x - width / 2, nm, width, label="Nelder-Mead", color="#4393c3")
        ax.bar(x + width / 2, cma, width, label="CMA-ES", color="#1a9641")
        ax.axhline(0.5 if col == "auprc" else 0, color="black", lw=0.8, ls="--" if col == "auprc" else "-")
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Nelder-Mead vs CMA-ES, matched budget -- single-gene knockout, "
        f"{N_GENES} genes, density={NETWORK_DENSITY}"
    )
    fig.tight_layout()
    png_path = HERE / "cma_es_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _plot_matrices(A_true: np.ndarray, A_hats: dict[str, np.ndarray]) -> None:
    combos = list(COMBINATIONS.keys())
    methods = ["TRUE", "Nelder-Mead", "CMA-ES"]
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
                ax.set_ylabel(method, fontsize=9)

    fig.suptitle(f"Recovered A -- {N_GENES} genes, density={NETWORK_DENSITY}, knockout scenario", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="A_ij", pad=0.01)
    png_path = HERE / "cma_es_recovered_matrices.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
