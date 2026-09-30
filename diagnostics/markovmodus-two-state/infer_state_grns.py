"""
Per-state GRN inference on the markovmodus two-state time series.

For each discrete state (state_0, state_1) separately, fit an OU model

    dX = A (mu - X) dt + diag(sigma) dW

to that state's cells across the snapshot times, with the method that has
worked best so far on synthetic data: Gaussian-moment KL + L-BFGS
(diagnostics/non-interventional/_gradfit, preset lbfgs_kl), plus an L1
penalty on the off-diagonal of A for sparsity. Each state gets its own
(A, mu, sigma); mu and sigma are fitted too, since here neither is known.

The two GRNs are saved side by side, not merged. The state_0 -> state_1
transition (rate, flow of cells between states) is deliberately NOT modelled
yet -- see "Known misspecification" below.

Why this is identifiable at all: neither state is stationary. Spliced means
drift ~10 -> ~11.4 counts over t=0..7 in both states, and state_1's total
variance falls 118 -> 88. A single stationary snapshot would pin A only up to
a skew-symmetric term (see _stationary_ground_truth.py); the time course plus
L1 removes that freedom.

Known misspecification (read before interpreting edges):
  1. No regulation in the generator. markovmodus genes are independent
     transcription -> splicing -> decay systems; the state only switches
     transcription rates. The structural truth is A diagonal. Every
     off-diagonal edge found here is an artifact, and off_ratio / n_edges in
     summary.json measure how many the method produces on edge-free data.
  2. state_1 is an open population. Cells keep arriving from state_0 (~4.5%
     of state_0 per unit time), so state_1 at t+1 is NOT state_1 at t pushed
     through an OU step. Arrivals look like a drift towards state_0-like
     expression and can be absorbed into A. Modelling the influx needs the
     transition rate -- the planned next step.
  3. Spliced only. Each gene's true dynamics are 2-D (unspliced, spliced);
     a 1-D OU per gene approximates the spliced marginal.

Usage (from the repo root):
    .venv/bin/python diagnostics/markovmodus-two-state/infer_state_grns.py
    .venv/bin/python diagnostics/markovmodus-two-state/infer_state_grns.py --lam 0,0.001,0.01,0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "non-interventional"))

import numpy as np  # noqa: E402

import _gradfit  # noqa: E402,F401  -- float64 before jax.numpy is used
import _mm_data  # noqa: E402
from _gradfit import transition as tr  # noqa: E402
from _gradfit.pipeline import Config, Stage, fit  # noqa: E402
from _gradfit.problem import Problem  # noqa: E402

A0_DIAG = 0.5   # initial self-decay rate; neutral, NOT the generator's 0.3


def make_problem(series):
    """mu and sigma here are INITS (both are fitted): mu = last snapshot mean,
    sigma from the stationary OU relation var = sigma^2 / (2a) at a = A0_DIAG."""
    dts = np.diff(series.times)
    if not np.allclose(dts, dts[0]):
        raise ValueError(f"{series.state}: uneven snapshot spacing {dts}")
    last = series.snaps[-1]
    return Problem(snaps=series.snaps, mu=last.mean(axis=0),
                   sigma=np.sqrt(2 * A0_DIAG * last.var(axis=0) + 1e-6),
                   dt=float(dts[0]), floor=None, A_true=None)


def make_config(lam, steps):
    return Config(
        name=f"kl_l1_lam{lam:g}",
        stages=(Stage(loss="kl", optimizer="lbfgs", lam=lam, shrinkage="lw",
                      opt_kwargs={"steps": steps}),),
        param="posdiag", init="diag", init_diag=A0_DIAG,
        learn_mu=True, learn_sigma=True,
    )


def mean_rollout(theta, series, dt):
    """Predicted mean trajectory from the first observed snapshot."""
    F, _ = tr.ou_exact(np.asarray(theta["A"]), np.asarray(theta["sigma"]), dt)
    F, mu = np.asarray(F), np.asarray(theta["mu"])
    m = [series.snaps[0].mean(axis=0)]
    for _ in series.snaps[1:]:
        m.append(mu + F @ (m[-1] - mu))
    return np.stack(m)


def summarize(theta, series, dt):
    A = np.asarray(theta["A"])
    G = A.shape[0]
    off = ~np.eye(G, dtype=bool)
    d = np.abs(np.diag(A)).mean()
    obs = np.stack([x.mean(axis=0) for x in series.snaps])
    pred = mean_rollout(theta, series, dt)
    ss_res = ((obs[1:] - pred[1:]) ** 2).sum()
    ss_tot = ((obs[1:] - obs[1:].mean(axis=0)) ** 2).sum()
    if series.target_counts is not None:
        tc = series.target_counts
        target_log = np.log1p(tc / tc.sum() * _mm_data.TARGET_SUM)
        mu_corr = float(np.corrcoef(np.asarray(theta["mu"]), target_log)[0, 1])
    else:
        mu_corr = float("nan")   # no simulator ground truth (real data)
    return {
        "state": series.state,
        "times": series.times.tolist(),
        "n_cells": series.n_cells.tolist(),
        "diag_mean": float(np.diag(A).mean()),
        "diag_range": [float(np.diag(A).min()), float(np.diag(A).max())],
        # ground truth has no edges: these count the artifacts
        "off_ratio": float(np.linalg.norm(A[off]) / np.linalg.norm(np.diag(A))),
        "n_edges_gt_10pct_diag": int((np.abs(A[off]) > 0.1 * d).sum()),
        "n_edges_gt_1pct_diag": int((np.abs(A[off]) > 0.01 * d).sum()),
        "max_abs_offdiag": float(np.abs(A[off]).max()),
        "mean_rollout_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mu_vs_target_corr": mu_corr,
        "sigma_mean": float(np.asarray(theta["sigma"]).mean()),
    }


def edge_table(A, genes, top=200):
    G = A.shape[0]
    rows = [(abs(A[i, j]), genes[j], genes[i], A[i, j])
            for i in range(G) for j in range(G) if i != j]
    rows.sort(reverse=True)
    lines = ["regulator,target,A_ij,abs_A_ij"]
    lines += [f"{r},{t},{a:.6g},{m:.6g}" for m, r, t, a in rows[:top]]
    return "\n".join(lines) + "\n"


def plot(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    states = list(results)
    fig, axes = plt.subplots(2, len(states), figsize=(6.2 * len(states), 11))
    vmax = max(np.abs(np.where(np.eye(r["theta"]["A"].shape[0], dtype=bool), 0,
                               r["theta"]["A"])).max() for r in results.values()) or 1.0
    for c, s in enumerate(states):
        r = results[s]
        A = np.array(r["theta"]["A"])
        np.fill_diagonal(A, np.nan)
        ax = axes[0, c]
        im = ax.imshow(A, cmap=div, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{s}: off-diagonal A  (diag mean {r['summary']['diag_mean']:.2f}, "
                     f"off/diag {r['summary']['off_ratio']:.2f})", fontsize=10, loc="left")
        ax.set_xlabel("regulator j"); ax.set_ylabel("target i")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, c]
        series = r["series"]
        obs = np.stack([x.mean(axis=0) for x in series.snaps])
        pred = mean_rollout(r["theta"], series, r["dt"])
        # the 6 genes that move most over time
        idx = np.argsort(-np.ptp(obs, axis=0))[:6]
        for k, g in enumerate(idx):
            col = f"C{k}"
            ax.plot(series.times, obs[:, g], "o", color=col, ms=5)
            ax.plot(series.times, pred[:, g], "-", color=col, lw=1.5,
                    label=series.genes[g])
        ax.set_title(f"{s}: mean expression, dots = data, lines = fitted rollout "
                     f"(R² {r['summary']['mean_rollout_r2']:.2f})", fontsize=10, loc="left")
        ax.set_xlabel("time"); ax.set_ylabel("log-normalized spliced")
        ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(_mm_data.DATA))
    ap.add_argument("--lam", default="0,0.01",
                    help="comma list of L1 weights; one output folder per value")
    ap.add_argument("--steps", type=int, default=1000, help="L-BFGS iterations")
    ap.add_argument("--out", default=str(_mm_data.REPO / "results" / "markovmodus"
                                         / "two-state-timeseries"))
    args = ap.parse_args()

    series_by_state, _ = _mm_data.load(args.data)
    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    for lam in [float(x) for x in args.lam.split(",")]:
        lam_dir = run_dir / f"lam{lam:g}"
        lam_dir.mkdir(parents=True, exist_ok=True)
        cfg = make_config(lam, args.steps)
        results = {}
        for s, series in series_by_state.items():
            problem = make_problem(series)
            print(f"[lam {lam:g}] {s}: {len(series.snaps)} snapshots at t={series.times.tolist()}, "
                  f"cells {series.n_cells.tolist()}", flush=True)
            res = fit(problem, cfg)
            theta = {k: np.asarray(v) for k, v in res.theta_hat.items()}
            summ = summarize(theta, series, problem.dt)
            summ.update(seconds=round(res.seconds, 1),
                        final_objective=res.start_logs[0][-1]["final_value"],
                        n_iter=res.start_logs[0][-1]["n_iter"])
            print("   " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in summ.items()
                                    if k not in ("state", "times", "n_cells")), flush=True)
            np.savez(lam_dir / f"{s}.npz", A=theta["A"], mu=theta["mu"], sigma=theta["sigma"],
                     genes=np.array(series.genes), times=series.times,
                     n_cells=series.n_cells, dt=problem.dt)
            (lam_dir / f"edges_{s}.csv").write_text(edge_table(theta["A"], series.genes))
            results[s] = {"theta": theta, "summary": summ, "series": series, "dt": problem.dt}
        (lam_dir / "summary.json").write_text(json.dumps(
            {"config": cfg.describe(), "data": args.data,
             "states": {s: r["summary"] for s, r in results.items()}}, indent=1))
        plot(results, lam_dir / "state_grns.png")
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
