"""
Assemble the part files written by compare_losses_single_gene_knockout.py
--combination into the same outputs the serial run produces.

    uv run python .../merge_knockout_parts.py parts/

Writes loss_comparison_knockout.{csv,png}, grn_networks_knockout.png and
recovered_matrices_knockout.npz next to this file -- identical in form to the
serial run, so downstream readers don't care which way it was produced.

Multi-seed runs: the CSV keeps every (combination, data_seed, fit_seed) row,
and a mean +/- std table across seeds is printed. Figures are drawn for ONE
seed (--figure-seed, default the lowest present), because a network graph
averaged over different ground-truth networks would be meaningless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from _knockout_plots import grn_networks, metrics_and_heatmaps
from _knockout_shared import COMBINATIONS

HERE = Path(__file__).parent

NUMERIC = ["A_err", "offdiag_err", "edge_corr", "auprc", "auroc",
           "precision_at_k", "recall_at_k", "sigma_err", "final_objective",
           "n_evals", "seconds"]


def load_parts(part_dir: Path):
    parts = sorted(part_dir.glob("part_*.npz"))
    if not parts:
        sys.exit(f"No part_*.npz found in {part_dir}")

    rows, mats = [], {}
    for p in parts:
        d = np.load(p, allow_pickle=False)
        row = dict(zip([str(k) for k in d["row_keys"]],
                       [str(v) for v in d["row_vals"]]))
        for col in NUMERIC:
            if col in row:
                row[col] = float(row[col])
        for col in ("data_seed", "fit_seed"):
            row[col] = int(float(row[col]))
        rows.append(row)
        mats[(row["combination"], row["data_seed"], row["fit_seed"])] = (
            d["A_hat"], d["A_true"], d["true_off"], int(d["k_edges"]))

    return pd.DataFrame(rows), mats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part_dir", type=Path, help="directory holding part_*.npz")
    ap.add_argument("--figure-seed", type=int, default=None,
                    help="data_seed to draw figures for (default: lowest present)")
    args = ap.parse_args()

    df, mats = load_parts(args.part_dir)
    order = {n: i for i, n in enumerate(COMBINATIONS)}
    df = df.sort_values(["data_seed", "fit_seed", "combination"],
                        key=lambda s: s.map(order) if s.name == "combination" else s)
    df = df.reset_index(drop=True)

    csv_path = HERE / "loss_comparison_knockout.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(df)} rows)")
    print(df.to_string(index=False))

    seeds = sorted(df["data_seed"].unique())
    if len(seeds) > 1:
        print(f"\nAcross {len(seeds)} data seeds -- mean +/- std:")
        agg = (df.groupby("combination")[["auprc", "auroc", "precision_at_k", "A_err"]]
                 .agg(["mean", "std"]).reindex(list(COMBINATIONS)))
        print(agg.to_string())
        print("\nOverlapping std ranges mean the difference is not resolved by "
              "this many seeds -- read the ranking accordingly.")

    fig_seed = args.figure_seed if args.figure_seed is not None else seeds[0]
    sub = df[df["data_seed"] == fig_seed].reset_index(drop=True)
    fit_seed = int(sub["fit_seed"].iloc[0])

    fitted, A_true, true_off, k_edges = {}, None, None, None
    for name in COMBINATIONS:
        key = (name, fig_seed, fit_seed)
        if key in mats:
            fitted[name], A_true, true_off, k_edges = mats[key]
    if not fitted:
        sys.exit(f"No parts for data_seed={fig_seed}")
    missing = [n for n in COMBINATIONS if n not in fitted]
    if missing:
        print(f"\nWARNING: no part for {', '.join(missing)} at data_seed={fig_seed} "
              f"-- those panels will be blank. Did those array tasks fail?")

    npz_path = HERE / "recovered_matrices_knockout.npz"
    np.savez(npz_path, A_true=A_true, **fitted)
    print(f"\nSaved: {npz_path}")

    metrics_and_heatmaps(sub, fitted, A_true, HERE / "loss_comparison_knockout.png")
    print(f"Saved: {HERE / 'loss_comparison_knockout.png'}  (data_seed={fig_seed})")
    grn_networks(sub, fitted, A_true, true_off, k_edges,
                 HERE / "grn_networks_knockout.png")
    print(f"Saved: {HERE / 'grn_networks_knockout.png'}  (data_seed={fig_seed})")


if __name__ == "__main__":
    main()
