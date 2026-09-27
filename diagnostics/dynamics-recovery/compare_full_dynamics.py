"""
Does the recovered dynamical system (theta_hat) actually reproduce the
observed dynamics -- not just its own parameter values, and not just one
teacher-forced prediction step?

Everything reported before this script (jko-testing/, sch-bridge-test/,
cma-es-test/, exact-ou-test/) scores recovery with: A_err/sigma_err
(pointwise parameter distance), edge_corr/AUPRC (edge ranking), and
one_step_ahead_w2 (predict ONE interval ahead from the TRUE snapshot at
t_k, every time -- errors never compound). None of those ask whether
theta_hat, run forward as a generative model from the first observed
snapshot, reproduces the full observed trajectory -- i.e. whether the
recovered OU map and FP map are actually a good dynamical system, not
just a good one-step local predictor.

This script re-fits all six loss combinations (CMA-ES, same as
../exact-ou-test/compare_exact_ou_cma.py) under both the Euler-Maruyama
and exact-OU forward models, then scores every resulting theta_hat with:

  - full_ou_trajectory_w2 / full_fp_trajectory_w2 (_dynamics_metrics.py):
    a COMPOUNDING rollout from snapshot 0 through every remaining
    snapshot, scored with the exact transition regardless of which
    forward model theta_hat was fit under -- this also resolves
    ../exact-ou-test/'s flagged confound (there, both fit methods were
    scored with the biased Euler rollout).
  - spectral_recovery: eigenvalues of A_hat vs A_true, since a Frobenius
    A_err can hide very different qualitative dynamics (relaxation rate,
    oscillation) if the parameter error lands on different
    eigen-directions than A_err's flat norm implies.

Not a test -- a standalone report script; slow (re-fits 6 combinations x
2 forward models with CMA-ES, ~2200 evals each). Re-run anytime,
overwrites this folder's full_dynamics_comparison.{csv,png}.

Usage:
    uv run python tests/diagnostics/dynamics-recovery/compare_full_dynamics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "exact-ou-test"))
sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.housekeeping.edge_recovery_metrics import auprc  # noqa: E402

from _dynamics_metrics import (  # noqa: E402
    full_fp_trajectory_w2,
    full_ou_trajectory_w2,
    spectral_recovery,
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

            one_step = one_step_ahead_w2(theta_hat, eval_snapshots, dt, n_proj=N_PROJ, seed=0)
            full_ou = full_ou_trajectory_w2(theta_hat, eval_snapshots, dt, n_proj=N_PROJ, seed=0)
            full_fp = full_fp_trajectory_w2(theta_hat, eval_snapshots, dt, n_proj=N_PROJ, seed=0)
            spectral = spectral_recovery(A_true, theta_hat.A)

            row = {
                "combination": combo_name, "method": method_label,
                "one_step_w2": one_step,
                "full_ou_trajectory_w2": full_ou,
                "full_fp_trajectory_w2": full_fp,
                "A_err": float(np.linalg.norm(theta_hat.A - A_true)),
                "edge_corr": float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0,
                "auprc": auprc(true_off, hat_off),
                "sigma_err": abs(theta_hat.sigma - effective_sigma(sim)),
                **spectral,
                **fit_info,
            }
            rows.append(row)
            print(
                f"  one_step={one_step:.4f}  full_ou={full_ou:.4f}  full_fp={full_fp:.4f}  "
                f"eig_l2_dist={spectral['eig_l2_dist']:.3f}  A_err={row['A_err']:.3f}  "
                f"evals={fit_info['n_evals']}  {fit_info['seconds']:.0f}s"
            )

    df = pd.DataFrame(rows)
    csv_path = HERE / "full_dynamics_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df)


def _plot(df: pd.DataFrame) -> None:
    combos = list(COMBINATIONS.keys())
    x = np.arange(len(combos))
    width = 0.35
    methods = ["CMA-ES + Euler-Maruyama", "CMA-ES + exact OU"]
    colors = ["#4393c3", "#1a9641"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    panels = [
        ("one_step_w2", "one-step-ahead W2 (teacher-forced, lower better)"),
        ("full_ou_trajectory_w2", "full-trajectory OU rollout W2 (compounding, lower better)"),
        ("full_fp_trajectory_w2", "full-trajectory FP rollout W2 (compounding, lower better)"),
        ("eig_l2_dist", "eigenvalue L2 distance, A_hat vs A_true (lower better)"),
    ]

    for ax, (col, title) in zip(axes, panels):
        for i, (method, color) in enumerate(zip(methods, colors)):
            vals = [df[(df.combination == c) & (df.method == method)][col].iloc[0] for c in combos]
            ax.bar(x + (i - 0.5) * width, vals, width, label=method, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7.5)

    fig.suptitle(
        "Recovered-dynamics metrics (all scored with the exact transition): "
        "compounding trajectory rollout + eigenvalue recovery, "
        f"{N_GENES} genes, density={NETWORK_DENSITY}, single-gene knockout"
    )
    fig.tight_layout()
    png_path = HERE / "full_dynamics_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
