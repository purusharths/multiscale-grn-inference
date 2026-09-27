"""
Which loss combination best recovers the GRN from single-gene knockout data?

Dataset: the coupled single-gene-knockout scenario from
../plot_single_gene_knockout.py (one gene knocked out at t_star, network with
real off-diagonal edges).

What is fitted: the FULL interaction matrix A (diagonal via exp() to keep it
positive, all off-diagonals free) plus sigma -- G^2 + 1 parameters. The
intervention means {mu_k} are held FIXED at their true values, because
Algorithm 1 takes them as given input ("Data: Known Intervention means", paper
line 2). So this measures exactly the thing GRN inference is for: recovering
who regulates whom, given known perturbations.

Combinations compared: OU, FP, OU+FP, OU+Cons, FP+Cons, OU+FP+Cons (imported
from _ours_combinations). Cons-alone is deliberately excluded -- earlier
sweeps (../../../compare_loss_combinations.py) established it is degenerate on
its own: it only checks that the OU and FP forward models agree with EACH
OTHER, never with the data, so it is trivially minimised by wrong dynamics.

Amortized: every combination gets the same optimizer, the same budget, and the
same (deliberately wrong, edge-free) starting point A = 1.2*I.

Outputs (all overwritten in place, next to this file):
    loss_comparison_knockout.csv    metrics per combination
    loss_comparison_knockout.png    metric bars + recovered A heatmaps
    grn_networks_knockout.png       recovered GRNs vs ground truth (networkx)
    recovered_matrices_knockout.npz the fitted A/sigma themselves, so a new
                                    question about the results doesn't cost
                                    another multi-hour refit

Two ways to run
---------------
Serial, all six in one process (~3h at the default config):

    uv run python "tests/diagnostics/interventions/single-gene-knockout/loss-combinations/compare_losses_single_gene_knockout.py"

Array-parallel, one process per combination, then merge. The six fits are
independent, so this finishes in the time of the SLOWEST single combination
rather than their sum:

    ... compare_losses_single_gene_knockout.py --combination OU --out-dir parts/
    ... merge_knockout_parts.py parts/

See run_knockout_array.sh for the SLURM job array that does this.

Not a test -- a standalone report script.
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
from _knockout_shared import (
    COMBINATIONS,
    DATA_SEED,
    FIT_SEED,
    MAXITER,
    N_GENES,
    build_dataset,
    run_one,
    summary_line,
)

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combination", choices=list(COMBINATIONS),
                   help="fit only this one and write a part file (array mode). "
                        "Omit to fit all six and write the full outputs.")
    p.add_argument("--data-seed", type=int, default=DATA_SEED,
                   help="seed for the network + cells. Vary across jobs to ask "
                        "whether results hold on GRNs in general.")
    p.add_argument("--fit-seed", type=int, default=FIT_SEED,
                   help="seed for the objective's own stochasticity.")
    p.add_argument("--out-dir", type=Path, default=HERE,
                   help="where part files are written (array mode).")
    args = p.parse_args()

    d = build_dataset(args.data_seed)
    print(f"fitting {N_GENES**2 + 1} params (full A + sigma), mu fixed at known {{mu_k}}")
    print(f"true sigma={d.sigma_true:.3f} | {d.k_edges} true edges | budget={MAXITER} iters")
    print(f"data_seed={args.data_seed} fit_seed={args.fit_seed}\n", flush=True)

    # ----- array mode: one combination, one part file ----------------------
    if args.combination:
        name = args.combination
        print(f"[{name}] optimising ...", flush=True)
        theta_hat, row = run_one(name, d, fit_seed=args.fit_seed,
                                 data_seed=args.data_seed)
        print(summary_line(row), flush=True)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        # Part filename carries both seeds, so a multi-seed sweep can write
        # every task into one directory without collisions.
        part = args.out_dir / f"part_{name}_d{args.data_seed}_f{args.fit_seed}.npz"
        np.savez(part, A_hat=theta_hat.A, sigma_hat=theta_hat.sigma,
                 A_true=d.A_true, sigma_true=d.sigma_true,
                 true_off=d.true_off, k_edges=d.k_edges,
                 row_keys=np.array(list(row.keys())),
                 row_vals=np.array([str(v) for v in row.values()]))
        print(f"Saved: {part}")
        return

    # ----- serial mode: all six, full outputs -------------------------------
    rows, fitted = [], {}
    for name in COMBINATIONS:
        print(f"[{name}] optimising ...", flush=True)
        theta_hat, row = run_one(name, d, fit_seed=args.fit_seed,
                                 data_seed=args.data_seed)
        fitted[name] = theta_hat.A
        rows.append(row)
        print(summary_line(row), flush=True)

    df = pd.DataFrame(rows)
    csv_path = HERE / "loss_comparison_knockout.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    npz_path = HERE / "recovered_matrices_knockout.npz"
    np.savez(npz_path, A_true=d.A_true, sigma_true=d.sigma_true,
             **{n: fitted[n] for n in fitted})
    print(f"Saved: {npz_path}")

    metrics_and_heatmaps(df, fitted, d.A_true, HERE / "loss_comparison_knockout.png")
    print(f"Saved: {HERE / 'loss_comparison_knockout.png'}")
    grn_networks(df, fitted, d.A_true, d.true_off, d.k_edges,
                 HERE / "grn_networks_knockout.png")
    print(f"Saved: {HERE / 'grn_networks_knockout.png'}")


if __name__ == "__main__":
    main()
