"""
Merge the per-task metrics_*.csv files of an array run (run_gradfit_ablation.py
--run-name ...) into one metrics.csv, and print mean +- sd per
(gen_substeps, preset) over data seeds.

    python merge_gradfit_runs.py results/diagnostics/non-interventional/gradfit/em-ablation
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

COLS = ["rel_fro", "offdiag_corr", "edge_auroc", "sign_acc_on_edges",
        "c0_gap", "c0gen_gap", "seconds"]


def main():
    run_dir = sys.argv[1]
    parts = sorted(glob.glob(os.path.join(run_dir, "metrics_*.csv")))
    if not parts:
        sys.exit(f"no metrics_*.csv in {run_dir}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    keys = [k for k in ("gen_substeps", "preset") if k in df]
    cols = [c for c in COLS if c in df]
    g = df.groupby(keys)[cols]
    summary = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    summary.insert(0, "n_seeds", df.groupby(keys).size())
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary)
    summary.to_csv(os.path.join(run_dir, "summary.csv"))
    print(f"\n{len(df)} rows from {len(parts)} parts -> {run_dir}/metrics.csv, summary.csv")


if __name__ == "__main__":
    main()
