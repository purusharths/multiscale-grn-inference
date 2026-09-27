"""
Does the model recover a GRN better WITHOUT an intervention?

The interventional runs (../interventions/single-gene-knockout/) came back at
chance on every edge-detection metric. One hypothesis for why: the knockout
produces a large transient, and the loss is dominated by how well a candidate
theta reproduces THAT, with the much smaller edge signal buried underneath. If
so, removing the intervention should help -- constant mu means the only thing
moving the population is the network itself.

The counter-hypothesis is that interventions are the causal leverage and
removing them makes recovery strictly harder. This run distinguishes them.

Dataset: constant mu, coupled A, cells started away from equilibrium (see
_stationary_ground_truth.py -- the perturbed start is what makes the data
informative at all; at equilibrium the drift is ~0 regardless of A).

Every run writes a self-contained folder runs/dd-mm-y-hh-mm-<genes>/ holding
its dataset picture, config, git commit, metrics and figures, so results stay
interpretable after the code moves on.

Usage:
    uv run python tests/diagnostics/non-interventional/compare_losses_stationary.py
    uv run python .../compare_losses_stationary.py --jobs 6        # parallel
    GRN_N_GENES=12 uv run python .../compare_losses_stationary.py  # sweep config
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "interventions"))

import numpy as np
import pandas as pd

from multsc_grn_inference.housekeeping.dataset_report import describe
from multsc_grn_inference.housekeeping.edge_recovery_metrics import (
    auprc, auroc, edge_labels, precision_at_k, recall_at_k,
)
from multsc_grn_inference.housekeeping.recovery_plots import grn_networks, metrics_and_heatmaps
from multsc_grn_inference.housekeeping.run_dir import Readme, git_commit, make_run_dir

from _ours_combinations import COMBINATIONS, fit_combination_cma, off_diag_indices
from _stationary_ground_truth import build, effective_sigma


def _env(name, default, cast=float):
    raw = os.environ.get(f"GRN_{name}")
    return default if raw is None else cast(raw)


# --- config (all GRN_<NAME> overridable) ------------------------------------
N_GENES = _env("N_GENES", 8, int)
N_CELLS = _env("N_CELLS", 3000, int)
N_SNAPS = _env("N_SNAPS", 15, int)
DT = _env("DT", 0.5)
NETWORK_DENSITY = _env("NETWORK_DENSITY", 0.3)
DATA_SEED = _env("DATA_SEED", 42, int)
FIT_SEED = _env("FIT_SEED", 0, int)
N_PROJ = _env("N_PROJ", 300, int)          # see ../interventions/.../_knockout_shared.py
# Nelder-Mead plateaus well short of full recovery on this 65-dim landscape
# even with its initial-simplex degeneracy fixed (see _ours_combinations.py's
# fit_combination_cma docstring) -- CMA-ES's adapted covariance is what
# actually escapes the diagonal-only local minimum, confirmed on the
# knockout data in ../cma-es-test/.
MAXFEVALS = _env("MAXFEVALS", 2200, int)
SIGMA0 = _env("SIGMA0", 0.15)
OPTIMIZER = "CMA-ES"

CONFIG = dict(n_genes=N_GENES, n_cells=N_CELLS, n_snaps=N_SNAPS, dt=DT,
              density=NETWORK_DENSITY, data_seed=DATA_SEED, fit_seed=FIT_SEED,
              n_proj=N_PROJ, maxfevals=MAXFEVALS, sigma0=SIGMA0, optimizer=OPTIMIZER)


def make_data(cfg):
    return build(n_genes=cfg["n_genes"], n_cells=cfg["n_cells"], n_snaps=cfg["n_snaps"],
                 dt=cfg["dt"], seed=cfg["data_seed"], density=cfg["density"])


def fit_one(args):
    """Top-level so it survives both fork and spawn start methods. Each worker
    rebuilds the dataset deterministically from the seed rather than inheriting
    it, which costs a few seconds and removes any pickling question."""
    name, cfg = args
    sim, snaps, times, mu_known, dt = make_data(cfg)
    off = off_diag_indices(cfg["n_genes"])
    true_off = np.array([sim.A[i, j] for i, j in off])
    k_edges = int(edge_labels(true_off).sum())

    t0 = time.perf_counter()
    theta_hat, info = fit_combination_cma(
        COMBINATIONS[name], snaps, mu_known, dt, cfg["n_genes"],
        n_proj=cfg["n_proj"], maxfevals=cfg["maxfevals"], sigma0=cfg["sigma0"],
        seed=cfg["fit_seed"],
    )
    hat_off = np.array([theta_hat.A[i, j] for i, j in off])
    row = {
        "combination": name,
        "A_err": float(np.linalg.norm(theta_hat.A - sim.A)),
        "offdiag_err": float(np.linalg.norm(hat_off - true_off)),
        "edge_corr": (float(np.corrcoef(true_off, hat_off)[0, 1])
                      if hat_off.std() > 1e-12 else 0.0),
        "auprc": auprc(true_off, hat_off),
        "auroc": auroc(true_off, hat_off),
        "precision_at_k": precision_at_k(true_off, hat_off, k_edges),
        "recall_at_k": recall_at_k(true_off, hat_off, k_edges),
        "sigma_err": abs(theta_hat.sigma - effective_sigma(sim)),
        "final_objective": info["final_objective"],
        "n_evals": info["n_evals"],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return name, theta_hat.A, float(theta_hat.sigma), row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=1,
                    help="fit this many combinations concurrently (6 = all at once)")
    args = ap.parse_args()

    run_dir = make_run_dir(HERE / "runs", N_GENES)
    print(f"run folder: {run_dir}\n", flush=True)

    sim, snaps, times, mu_known, dt = make_data(CONFIG)
    off = off_diag_indices(N_GENES)
    true_off = np.array([sim.A[i, j] for i, j in off])
    k_edges = int(edge_labels(true_off).sum())
    chance = float(edge_labels(true_off).mean())

    fig_path, stats = describe(
        sim, snaps, times, run_dir / "dataset.png", dt=dt,
        title=f"Non-interventional dataset — {N_GENES} genes, constant mu, "
              f"density={NETWORK_DENSITY}, seed={DATA_SEED}",
    )
    print(f"dataset figure: {fig_path}")
    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print(flush=True)

    names = list(COMBINATIONS)
    payload = [(n, CONFIG) for n in names]
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(fit_one, payload):
                print(f"[{r[0]}] done  {r[3]['seconds']}s  AUPRC={r[3]['auprc']:.3f}", flush=True)
                results.append(r)
    else:
        for p in payload:
            print(f"[{p[0]}] optimising ...", flush=True)
            r = fit_one(p)
            print(f"  AUPRC={r[3]['auprc']:.3f}  AUROC={r[3]['auroc']:.3f}  "
                  f"P@{k_edges}={r[3]['precision_at_k']:.3f}  "
                  f"A_err={r[3]['A_err']:.3f}  {r[3]['seconds']}s", flush=True)
            results.append(r)

    results.sort(key=lambda r: names.index(r[0]))
    fitted = {n: A for n, A, _, _ in results}
    df = pd.DataFrame([r[3] for r in results])
    df.to_csv(run_dir / "results.csv", index=False)
    np.savez(run_dir / "recovered_matrices.npz", A_true=sim.A,
             sigma_true=effective_sigma(sim), **fitted)

    scen = (f"non-interventional (constant mu), {N_GENES} genes, "
            f"density={NETWORK_DENSITY}, N={N_CELLS}, {N_SNAPS} snapshots, "
            f"{OPTIMIZER} maxfevals={MAXFEVALS} sigma0={SIGMA0}, n_proj={N_PROJ}")
    metrics_and_heatmaps(df, fitted, sim.A, run_dir / "metrics_and_matrices.png",
                         title=f"Loss-combination comparison — {scen}", chance=chance)
    grn_networks(df, fitted, sim.A, true_off, k_edges, run_dir / "grn_networks.png",
                 title=f"Recovered GRN vs ground truth — {scen}; "
                       f"each panel thresholded to its top {k_edges} edges")

    # ---- README -------------------------------------------------------------
    best = df.loc[df["auprc"].idxmax()]
    above = df[df["auprc"] > chance + 0.05]
    verdict = (f"**{len(above)} of {len(df)} combinations** score more than 0.05 above the "
               f"chance baseline of {chance:.3f} on AUPRC. Best: **{best['combination']}** "
               f"at {best['auprc']:.3f}.")
    if above.empty:
        verdict = (f"**No combination beats chance.** The AUPRC baseline here is "
                   f"{chance:.3f} (the fraction of candidate pairs that are real edges) "
                   f"and every combination lands within 0.05 of it, as does AUROC of 0.5. "
                   f"Edge recovery is not working on this dataset either.")

    (Readme(f"Non-interventional run — {N_GENES} genes")
     .text(f"`{run_dir.name}` · commit `{git_commit()}` · "
           f"generated {time.strftime('%Y-%m-%d %H:%M')}")
     .section("What this run asks",
              "Whether removing the intervention helps GRN recovery. The knockout runs "
              "scored at chance on every edge-detection metric; one explanation is that "
              "the intervention transient dominates the loss and buries the edge signal. "
              "Here mu is constant, so the network itself is the only thing moving the "
              "population.")
     .section("Dataset")
     .text(f"Generated by `tests/diagnostics/non-interventional/_stationary_ground_truth.py` "
           f"(constant mu, coupled A, perturbed start). Deterministic from "
           f"`data_seed={DATA_SEED}` — no data files are stored; rerun the command below "
           f"to reproduce it exactly.")
     .kv(stats)
     .image("dataset.png", "dataset overview")
     .text("_Top-left_ UMAP over all snapshots, coloured by time. _Top-right_ population "
           "mean trajectory in PCA space, with the perturbed start and the mu target. "
           "_Bottom-left_ per-gene relaxation, dotted lines are mu. _Bottom-right_ the "
           "true A being recovered.")
     .section("Config")
     .kv(CONFIG)
     .section("Results")
     .text(verdict)
     .table(df)
     .text(f"**Baselines.** AUPRC and precision@{k_edges} both have a chance level of "
           f"{chance:.3f} (= {k_edges} true edges / {len(off)} candidate pairs); AUROC's "
           f"is 0.500 by construction. Compare every row against those, not against "
           f"each other.")
     .image("grn_networks.png", "recovered networks")
     .text(f"Recovered networks vs ground truth, each thresholded to its top {k_edges} "
           "edges. Solid = correctly recovered, dashed and pale = false positive.")
     .image("metrics_and_matrices.png", "metrics and recovered A")
     .section("Files")
     .text("| file | contents |\n|---|---|\n"
           "| `dataset.png` | dataset overview |\n"
           "| `results.csv` | metrics per loss combination |\n"
           "| `grn_networks.png` | recovered GRNs vs truth |\n"
           "| `metrics_and_matrices.png` | metric bars + recovered A heatmaps |\n"
           "| `recovered_matrices.npz` | fitted A and sigma, for reanalysis without refitting |")
     .section("Reproduce")
     .code(f"GRN_N_GENES={N_GENES} GRN_N_CELLS={N_CELLS} GRN_N_SNAPS={N_SNAPS} "
           f"GRN_DT={DT} GRN_NETWORK_DENSITY={NETWORK_DENSITY} GRN_DATA_SEED={DATA_SEED} "
           f"GRN_N_PROJ={N_PROJ} GRN_MAXFEVALS={MAXFEVALS} GRN_SIGMA0={SIGMA0} \\\n"
           f"  uv run python tests/diagnostics/non-interventional/compare_losses_stationary.py "
           f"--jobs {args.jobs}", "bash")
     .write(run_dir / "README.md"))

    print(f"\n{df.to_string(index=False)}")
    print(f"\nchance baseline (AUPRC, P@k) = {chance:.3f}   AUROC = 0.500")
    print(f"\nWritten: {run_dir}/README.md")


if __name__ == "__main__":
    main()
