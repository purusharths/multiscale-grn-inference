"""
Loss-combination ablation on the non-interventional dataset, at the best
setting found so far, across GRN sizes and cell counts.

Fixed (the best setting from the gradfit / EM ablations): L-BFGS 500 steps,
posdiag parameterization, diagonal init, exact-OU transition in the fit, data
generated with GRN_GEN_SUBSTEPS=50 so the data are samples of the continuous
model the fit assumes. Run 20260928-152916 had lbfgs_kl at offdiag_corr 0.82
vs 0.69 for lbfgs_sw, at ~5 s vs ~30 min.

Varied: which loss terms are summed (every non-empty subset of TERMS, each term
normalised to 1 at A = I -- see _gradfit/losses.py):

    sw      particle sliced-W2, one step          (the paper's L_OU, micro)
    kl      Gaussian-moment KL, one step          (L_FP analogue, macro)
    bures   Gaussian-moment Bures-W2, one step    (L_FP, other geometry)
    kl_ro   moment KL rolled out from snapshot 0  (multi-step)

The paper's L_cons (particle pushforward vs FP pushforward) has no counterpart:
for the linear-Gaussian model both are the same distribution, so the term is
zero up to sampling noise.

Usage:
    python run_loss_combo_ablation.py --combos moments --n-genes 10 --n-cells 3000 --seeds 42
    python run_loss_combo_ablation.py --combos sw+kl --n-genes 10 --n-cells 3000 --seeds 42 \\
        --run-name loss-combo
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _gradfit  # noqa: E402,F401  -- enables float64 before jax.numpy is used
import _stationary_ground_truth as gt  # noqa: E402
from _gradfit import metrics  # noqa: E402
from _gradfit.pipeline import Config, Stage, fit  # noqa: E402
from _gradfit.problem import from_stationary_build  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TERMS = ("kl", "bures", "kl_ro", "sw")


def all_combos():
    return ["+".join(c) for r in range(1, len(TERMS) + 1)
            for c in itertools.combinations(TERMS, r)]


def select(spec):
    """'all' | 'moments' (no sw) | 'with_sw' | comma list of combos."""
    combos = all_combos()
    if spec == "all":
        return combos
    if spec == "moments":
        return [c for c in combos if "sw" not in c.split("+")]
    if spec == "with_sw":
        return [c for c in combos if "sw" in c.split("+")]
    names = spec.split(",")
    unknown = [n for n in names if n not in combos]
    if unknown:
        sys.exit(f"unknown combos {unknown}; available: {combos}")
    return names


def config(combo):
    return Config(combo, (Stage(loss=combo, optimizer="lbfgs", transition="exact",
                                n_proj=200, opt_kwargs={"steps": 500}),))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combos", default="all")
    ap.add_argument("--seeds", default="42", help="data seeds (network + noise)")
    ap.add_argument("--n-genes", type=int, default=8)
    ap.add_argument("--n-cells", type=int, default=3000)
    ap.add_argument("--n-snaps", type=int, default=15)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "diagnostics", "non-interventional", "gradfit"))
    ap.add_argument("--run-name", default=None,
                    help="fixed run folder shared by array tasks "
                         "(merge with merge_gradfit_runs.py)")
    args = ap.parse_args()

    names = select(args.combos)
    gen = gt.GEN_SUBSTEPS
    if gen < 50:
        print(f"WARNING: GRN_GEN_SUBSTEPS={gen}; the best setting uses 50", flush=True)
    run_dir = os.path.join(args.out, args.run_name or (
        time.strftime("%Y%m%d-%H%M%S") + "_loss-combo"))
    os.makedirs(run_dir, exist_ok=True)
    tag = f"G{args.n_genes}_c{args.n_cells}"

    for seed in [int(s) for s in args.seeds.split(",")]:
        problem = from_stationary_build(n_genes=args.n_genes, n_cells=args.n_cells,
                                        n_snaps=args.n_snaps, dt=args.dt, seed=seed,
                                        density=args.density)
        for name in names:
            cfg = config(name)
            print(f"[{tag} seed {seed}] {name} ...", flush=True)
            res = fit(problem, cfg)
            row = {"preset": name, "n_terms": len(name.split("+")),
                   "n_genes": args.n_genes, "n_cells": args.n_cells,
                   "gen_substeps": gen, "data_seed": seed,
                   "n_edges": int((problem.A_true != 0).sum() - args.n_genes),
                   "seconds": round(res.seconds, 2),
                   **metrics.recovery(res.A_hat, problem.A_true)}
            # check 0 under both yardsticks: particle SW and the moment KL
            for yard in ("sw", "kl"):
                c0 = metrics.check0(problem, res.A_hat, loss=yard,
                                    transition="exact", n_proj=200)
                row.update({f"c0{yard}_{k}": v for k, v in c0.items()})
            fname = f"{name}_{tag}_seed{seed}"
            with open(os.path.join(run_dir, fname + ".json"), "w") as fh:
                json.dump({"config": cfg.describe(), "metrics": row,
                           "A_hat": res.A_hat.tolist(), "A_true": problem.A_true.tolist(),
                           "start_logs": res.start_logs}, fh, indent=1, default=float)
            # one CSV per fit: an array task that dies mid-list keeps what it finished
            with open(os.path.join(run_dir, f"metrics_{fname}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            print("   " + "  ".join(f"{k}={v:.4g}" for k, v in row.items()
                                    if isinstance(v, float)), flush=True)
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
