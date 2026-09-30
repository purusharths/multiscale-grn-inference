"""
Per-state GRN inference on the markovmodus two-state time series with the
paper's Algorithm 1 loss terms (OU / FP / Cons) instead of the Gaussian-moment
KL used in ../markovmodus-two-state/.

Everything except the loss is identical to ../markovmodus-two-state/
infer_state_grns.py (same data, same per-state split, same OU model with
fitted A, mu and per-gene sigma, same L1 on off-diagonal A, same L-BFGS,
same summaries), so the two folders compare the LOSS and nothing else.

The loss is _gradfit's "oufp": sliced-W2 on particle clouds, summed over
snapshot intervals, as in src/multsc_grn_inference/compute_loss.py but in JAX
so it has gradients. Projections, KDE resampling and OU noise are drawn once
per fit (common random numbers), so L-BFGS sees a deterministic objective.

Combinations match the knockout sweep (Cons-alone excluded: it never touches
the data):
    OU, FP, OU+FP, OU+Cons, FP+Cons, OU+FP+Cons

Two things to know when reading results:
  - FP at 100 genes: Scott's KDE bandwidth is h = n^(-1/(G+4)) ~ 0.93, so
    the FP arm's input cloud carries ~1.9x the data covariance. Any FP
    combination is fitting through that bias.
  - L1 weights are NOT comparable to the KL folder's: the sliced-W2 sum has a
    different scale. See run_oufp_array.sh for the lambda grid.

One task = one (combination, lambda), fitting both states. Usage from the
repo root:
    .venv/bin/python diagnostics/markovmodus-two-state-oufp/infer_state_grns_oufp.py \\
        --combo OU+FP --lam 1e-4 --run-name test
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "markovmodus-two-state"))

import numpy as np  # noqa: E402

import infer_state_grns as base  # noqa: E402  -- also sets up _gradfit on sys.path
import _mm_data  # noqa: E402
from _gradfit.pipeline import Config, Stage, fit  # noqa: E402

COMBOS = {
    "OU": ("ou",),
    "FP": ("fp",),
    "OU+FP": ("ou", "fp"),
    "OU+Cons": ("ou", "cons"),
    "FP+Cons": ("fp", "cons"),
    "OU+FP+Cons": ("ou", "fp", "cons"),
}
OUT = _mm_data.REPO / "results" / "markovmodus" / "two-state-timeseries-oufp"


def make_config(combo, lam, steps, n_proj, seed):
    return Config(
        name=f"{combo}_lam{lam:g}",
        stages=(Stage(loss="oufp", terms=COMBOS[combo], optimizer="lbfgs", lam=lam,
                      n_proj=n_proj, floor=True, opt_kwargs={"steps": steps}),),
        param="posdiag", init="diag", init_diag=base.A0_DIAG,
        learn_mu=True, learn_sigma=True, seed=seed,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", required=True, choices=list(COMBOS))
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1000, help="L-BFGS iterations")
    ap.add_argument("--n-proj", type=int, default=300,
                    help="sliced-W2 projections (300 = the knockout sweep's setting)")
    ap.add_argument("--seed", type=int, default=0,
                    help="projections / KDE resampling / OU noise")
    ap.add_argument("--data", default=str(_mm_data.DATA))
    ap.add_argument("--run-name", default=None,
                    help="fixed run folder so array tasks share it (default: timestamp)")
    args = ap.parse_args()

    series_by_state, _ = _mm_data.load(args.data)
    run = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out = OUT / run / args.combo / f"lam{args.lam:g}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args.combo, args.lam, args.steps, args.n_proj, args.seed)

    results = {}
    for s, series in series_by_state.items():
        # expression is log1p(counts) >= 0, so the paper's max(., 0) floor applies
        problem = dataclasses.replace(base.make_problem(series), floor=0.0)
        print(f"[{args.combo} lam {args.lam:g}] {s}: t={series.times.tolist()} "
              f"cells {series.n_cells.tolist()}", flush=True)
        res = fit(problem, cfg)
        theta = {k: np.asarray(v) for k, v in res.theta_hat.items()}
        summ = base.summarize(theta, series, problem.dt)
        summ.update(seconds=round(res.seconds, 1),
                    final_objective=res.start_logs[0][-1]["final_value"],
                    n_iter=res.start_logs[0][-1]["n_iter"])
        print("   " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in summ.items()
                                if k not in ("state", "times", "n_cells")), flush=True)
        np.savez(out / f"{s}.npz", A=theta["A"], mu=theta["mu"], sigma=theta["sigma"],
                 genes=np.array(series.genes), times=series.times,
                 n_cells=series.n_cells, dt=problem.dt)
        (out / f"edges_{s}.csv").write_text(base.edge_table(theta["A"], series.genes))
        results[s] = {"theta": theta, "summary": summ, "series": series, "dt": problem.dt}

    (out / "summary.json").write_text(json.dumps(
        {"combo": args.combo, "lam": args.lam, "config": cfg.describe(), "data": args.data,
         "states": {s: r["summary"] for s, r in results.items()}}, indent=1))
    base.plot(results, out / "state_grns.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
