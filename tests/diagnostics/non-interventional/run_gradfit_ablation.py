"""
Ablation runner for the _gradfit components on the non-interventional dataset.

Each preset differs from a neighbour in one component, grouped by the question
it answers:

  optimizer   (loss fixed: SW, exact transition, 200 fixed projections)
      nm_sw, cma_sw, lbfgs_sw, adam_sw, adam_lbfgs_sw
  loss        (optimizer fixed: L-BFGS)
      lbfgs_sw, lbfgs_kl, lbfgs_kl_rollout, lbfgs_bures
  pipeline    moment warm start then SW refine, +/- L1, +/- multi-start
      kl_then_sw, kl_then_sw_l1, kl_then_sw_ms
  model fidelity inside the SW loss
      lbfgs_sw_em (Euler-Maruyama transition), lbfgs_sw_nofloor
  parameterization
      lbfgs_sw_full (vs posdiag everywhere else)
  sanity
      truth_init_sw -- starts AT A_true; if it walks away, the SW minimum
      is not at the truth (loss bias, not optimizer failure)

Every run reports recovery metrics and "check 0" (see _gradfit/metrics.py)
under a common yardstick: SW, exact transition, 200 projections, held-out seeds.

Usage:
    python run_gradfit_ablation.py                          # all presets, seed 42
    python run_gradfit_ablation.py --presets lbfgs_sw,kl_then_sw --seeds 42,43,44
    GRN_PERTURBATION=heterogeneous GRN_PERTURB_AMP=1.5 python run_gradfit_ablation.py
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

import numpy as np  # noqa: E402

import _gradfit  # noqa: E402,F401  -- enables float64 before jax.numpy is used
import _stationary_ground_truth as gt  # noqa: E402
from _gradfit import metrics  # noqa: E402
from _gradfit.pipeline import Config, Stage, fit  # noqa: E402
from _gradfit.problem import from_stationary_build  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def presets(l1):
    sw = dict(loss="sw", transition="exact", n_proj=200)
    lb = dict(optimizer="lbfgs", opt_kwargs={"steps": 500})
    kl = Stage(loss="kl", **lb)
    sw_lb = Stage(**sw, **lb)
    sw_adam = Stage(**sw, optimizer="adam", resample=True,
                    opt_kwargs={"steps": 1500, "lr": 1e-2})
    P = [
        # optimizer ablation
        Config("nm_sw", (Stage(**sw, optimizer="nelder_mead",
                               opt_kwargs={"max_evals": 20000}),)),
        Config("cma_sw", (Stage(**sw, optimizer="cma",
                                opt_kwargs={"max_evals": 20000, "sigma0": 0.3}),)),
        Config("lbfgs_sw", (sw_lb,)),
        Config("adam_sw", (sw_adam,)),
        Config("adam_lbfgs_sw", (sw_adam, sw_lb)),
        # loss ablation
        Config("lbfgs_kl", (kl,)),
        Config("lbfgs_kl_rollout", (Stage(loss="kl", rollout=True, **lb),)),
        Config("lbfgs_bures", (Stage(loss="bures", **lb),)),
        # pipeline
        Config("kl_then_sw", (kl, sw_lb)),
        Config("kl_then_sw_l1", (kl, Stage(**sw, **lb, lam=l1))),
        Config("kl_then_sw_ms", (kl, sw_lb), n_starts=5),
        # model fidelity
        Config("lbfgs_sw_em", (Stage(**{**sw, "transition": "em"}, **lb),)),
        Config("lbfgs_sw_nofloor", (Stage(**sw, **lb, floor=False),)),
        # parameterization
        Config("lbfgs_sw_full", (sw_lb,), param="full"),
        # sanity
        Config("truth_init_sw", (sw_lb,), init="truth"),
    ]
    return {c.name: c for c in P}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", default="all")
    ap.add_argument("--seeds", default="42", help="data seeds (network + noise)")
    ap.add_argument("--n-genes", type=int, default=8)
    ap.add_argument("--n-cells", type=int, default=3000)
    ap.add_argument("--n-snaps", type=int, default=15)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--l1", type=float, default=1e-4, help="lam for *_l1 presets")
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "diagnostics", "non-interventional", "gradfit"))
    args = ap.parse_args()

    table = presets(args.l1)
    names = list(table) if args.presets == "all" else args.presets.split(",")
    unknown = [n for n in names if n not in table]
    if unknown:
        sys.exit(f"unknown presets {unknown}; available: {list(table)}")

    run_dir = os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S")
                           + f"_{gt.PERTURBATION}_amp{gt.SHIFT_FRACTION}")
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        problem = from_stationary_build(n_genes=args.n_genes, n_cells=args.n_cells,
                                        n_snaps=args.n_snaps, dt=args.dt, seed=seed,
                                        density=args.density)
        for name in names:
            cfg = table[name]
            print(f"[seed {seed}] {name} ...", flush=True)
            res = fit(problem, cfg)
            row = {"preset": name, "data_seed": seed, "seconds": round(res.seconds, 2),
                   **metrics.recovery(res.A_hat, problem.A_true)}
            c0 = metrics.check0(problem, res.A_hat, loss="sw",
                                transition="exact", n_proj=200)
            row.update({f"c0_{k}": v for k, v in c0.items()})
            rows.append(row)
            with open(os.path.join(run_dir, f"{name}_seed{seed}.json"), "w") as fh:
                json.dump({"config": cfg.describe(), "metrics": row,
                           "A_hat": res.A_hat.tolist(), "A_true": problem.A_true.tolist(),
                           "best_start": res.best_start,
                           "start_logs": res.start_logs}, fh, indent=1, default=float)
            print("   " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in row.items() if k not in ("preset", "data_seed")),
                  flush=True)

    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["rel_fro", "offdiag_corr", "edge_auroc", "sign_acc_on_edges", "c0_gap", "seconds"]
    print(f"\n{'preset':<18}" + "".join(f"{c:>18}" for c in cols))
    for name in names:
        sub = [r for r in rows if r["preset"] == name]
        print(f"{name:<18}" + "".join(f"{np.mean([r[c] for r in sub]):>18.4g}" for c in cols))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
