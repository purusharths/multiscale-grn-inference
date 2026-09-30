"""
The paper's loss terms (L_OU, L_FP, L_cons from src/.../compute_loss.py) in
combination, across GRN sizes and cell counts -- the counterpart of
run_loss_combo_ablation.py, which tested the _gradfit Gaussian-moment
substitutes instead.

Combinations are ../interventions/_ours_combinations.py's COMBINATIONS: OU, FP,
OU+FP, OU+Cons, FP+Cons, OU+FP+Cons (Cons alone is excluded there: it never
looks at the data). Terms are summed unweighted, as in the paper.

Fixed (the best setting for this track so far, compare_losses_stationary.py
run 22-09-6-11-28-8): CMA-ES, sigma0=0.15, n_proj=300, exact OU forward model
(GRN_FORWARD_MODEL default), A and sigma both fitted, mu known, cold start
(the moment preconditioner came out mixed, 3 of 6 combinations worse).
Data generated with GRN_GEN_SUBSTEPS=50, as in the gradfit ablation.

Budget: that run used 2200 evaluations for 65 parameters at 8 genes, ~34 per
parameter. A fixed 2200 at 20 genes (401 parameters) would confound "bigger
network" with "fewer evaluations per parameter", so the budget scales:
maxfevals = evals_per_param * n_params.

Also reports check 0 on the fit's own objective: gap = L(theta_hat) - L(theta_true)
at the same seed. gap > 0 means CMA-ES stopped short of what the loss prefers.

Usage:
    python run_oufp_combo_ablation.py --combos OU+FP --n-genes 10 --n-cells 3000 --seeds 42 \\
        --run-name oufp-combo
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "interventions"))

import numpy as np  # noqa: E402

import _stationary_ground_truth as gt  # noqa: E402
from _gradfit import metrics  # noqa: E402
from _ours_combinations import (  # noqa: E402
    COMBINATIONS, _make_objective, encode, fit_combination_cma, off_diag_indices,
)
from multsc_grn_inference.compute_loss import FORWARD_MODEL  # noqa: E402
from multsc_grn_inference.housekeeping.edge_recovery_metrics import auprc  # noqa: E402
from multsc_grn_inference.preprocessing import preprocessing  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combos", default="all", help="'all' or comma list of " + ",".join(COMBINATIONS))
    ap.add_argument("--seeds", default="42", help="data seeds (network + noise)")
    ap.add_argument("--n-genes", type=int, default=8)
    ap.add_argument("--n-cells", type=int, default=3000)
    ap.add_argument("--n-snaps", type=int, default=15)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--n-proj", type=int, default=300)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--evals-per-param", type=float, default=34.0)
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "diagnostics", "non-interventional", "oufp"))
    ap.add_argument("--run-name", default=None,
                    help="fixed run folder shared by array tasks "
                         "(merge with merge_gradfit_runs.py)")
    args = ap.parse_args()

    names = list(COMBINATIONS) if args.combos == "all" else args.combos.split(",")
    unknown = [n for n in names if n not in COMBINATIONS]
    if unknown:
        sys.exit(f"unknown combos {unknown}; available: {list(COMBINATIONS)}")
    gen = gt.GEN_SUBSTEPS
    if gen < 50:
        print(f"WARNING: GRN_GEN_SUBSTEPS={gen}; the best setting uses 50", flush=True)
    G = args.n_genes
    off = off_diag_indices(G)
    n_params = G + len(off) + 1
    maxfevals = int(round(args.evals_per_param * n_params))
    run_dir = os.path.join(args.out, args.run_name or (
        time.strftime("%Y%m%d-%H%M%S") + "_oufp-combo"))
    os.makedirs(run_dir, exist_ok=True)
    tag = f"G{G}_c{args.n_cells}"

    for seed in [int(s) for s in args.seeds.split(",")]:
        sim, snaps, _times, mu_known, dt = gt.build(
            n_genes=G, n_cells=args.n_cells, n_snaps=args.n_snaps, dt=args.dt,
            seed=seed, density=args.density)
        sigma_true = gt.effective_sigma(sim)
        true_off = np.array([sim.A[i, j] for i, j in off])
        for name in names:
            print(f"[{tag} seed {seed}] {name}  maxfevals={maxfevals} ...", flush=True)
            theta_hat, info = fit_combination_cma(
                COMBINATIONS[name], snaps, mu_known, dt, G, n_proj=args.n_proj,
                maxfevals=maxfevals, sigma0=args.sigma0, seed=args.fit_seed)
            hat_off = np.array([theta_hat.A[i, j] for i, j in off])

            # check 0: the fit's own objective (same seed = same projections
            # and noise) at theta_hat vs at the truth
            chi = [preprocessing(X) for X in snaps]
            obj = _make_objective(COMBINATIONS[name], snaps, chi, mu_known, dt, G, off,
                                  args.n_proj, args.fit_seed)
            L_fit = obj(encode(theta_hat.A, theta_hat.sigma, off))
            L_true = obj(encode(sim.A, sigma_true, off))

            row = {"preset": name, "n_terms": len(COMBINATIONS[name]),
                   "n_genes": G, "n_cells": args.n_cells, "gen_substeps": gen,
                   "data_seed": seed, "forward_model": FORWARD_MODEL,
                   "n_edges": int((true_off != 0).sum()), "maxfevals": maxfevals,
                   "n_evals": info["n_evals"], "seconds": info["seconds"],
                   **metrics.recovery(theta_hat.A, sim.A),
                   "auprc": auprc(true_off, hat_off),
                   "sigma_rel_err": abs(theta_hat.sigma - sigma_true) / sigma_true,
                   "c0_L_fit": L_fit, "c0_L_true": L_true, "c0_gap": L_fit - L_true,
                   "c0_verdict": "optimizer-limited" if L_fit > L_true else "loss-limited"}
            fname = f"{name}_{tag}_seed{seed}"
            with open(os.path.join(run_dir, fname + ".json"), "w") as fh:
                json.dump({"metrics": row, "A_hat": theta_hat.A.tolist(),
                           "sigma_hat": theta_hat.sigma, "A_true": sim.A.tolist(),
                           "sigma_true": sigma_true, "sigma0": args.sigma0,
                           "n_proj": args.n_proj, "fit_seed": args.fit_seed,
                           "final_objective": info["final_objective"]},
                          fh, indent=1, default=float)
            with open(os.path.join(run_dir, f"metrics_{fname}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            print("   " + "  ".join(f"{k}={v:.4g}" for k, v in row.items()
                                    if isinstance(v, float)), flush=True)
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
