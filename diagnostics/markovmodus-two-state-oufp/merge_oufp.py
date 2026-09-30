"""
Collect every <combo>/lam<lam>/summary.json of an array run into one table
(one row per combo x lambda x state), write summary.csv next to them, and
print it.

    .venv/bin/python diagnostics/markovmodus-two-state-oufp/merge_oufp.py \\
        results/markovmodus/two-state-timeseries-oufp/oufp-grid
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

COLS = ["diag_mean", "off_ratio", "n_edges_gt_10pct_diag", "n_edges_gt_1pct_diag",
        "max_abs_offdiag", "mean_rollout_r2", "sigma_mean", "final_objective",
        "n_iter", "seconds"]


def main():
    run = Path(sys.argv[1])
    rows = []
    for f in sorted(run.glob("*/lam*/summary.json")):
        d = json.loads(f.read_text())
        for state, s in d["states"].items():
            rows.append({"combo": d["combo"], "lam": d["lam"], "state": state,
                         **{c: s.get(c) for c in COLS}})
    if not rows:
        sys.exit(f"no */lam*/summary.json under {run}")
    df = pd.DataFrame(rows).sort_values(["state", "combo", "lam"])
    df.to_csv(run / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        print(df.to_string(index=False))
    print(f"\n{len(df)} rows -> {run / 'summary.csv'}")


if __name__ == "__main__":
    main()
