"""
Multiscale GRN inference — entry point.

Pipeline
--------
1. load_dataset          reads CSV snapshots → dict[int, ndarray]
2. compute_chi_hat       fits KDE per timepoint
3. bip_initialise        MAP warm start on linearised OU
4. optimise              outer bilevel Adam loop
5. write outputs         A_hat.csv, sigma_hat.txt, loss_curve.png,
                         theta0_vs_thetahat.png, snapshot_pca.png,
                         grn_comparison.png, edge_scatter.png (if --grn_dir given)

Usage
-----
    python main.py
    python main.py --csv_dir ./csv --n_outer_steps 200 --lr 5e-4
    python main.py --grn_dir ./output-destructive-measurements
    python main.py --w2_backend exact --n_particles 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bip import bip_initialise
from config import GRNConfig
from data import load_dataset, load_true_grns
from optimise import optimise
from preprocess import compute_chi_hat
from visualize import (
    compare_grns,
    plot_a_matrix,
    plot_edge_scatter,
    plot_grn_comparison,
    plot_loss_curve,
    plot_snapshot_pca,
    plot_theta_comparison,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent          # directory containing main.py


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multiscale GRN inference from cross-sectional RNA-seq snapshots"
    )
    p.add_argument("--csv_dir",        default=str(_HERE / "csv"))
    p.add_argument("--results_dir",    default=str(_HERE / "results"))
    p.add_argument("--grn_dir",        default=None,
                   help="Directory with grn_weighted_pop_*.csv for comparison. "
                        "Defaults to output-destructive-measurements/ next to main.py.")
    p.add_argument("--tau",            type=float, default=0.1)
    p.add_argument("--dt",             type=float, default=0.01)
    p.add_argument("--n_em_steps",     type=int,   default=20)
    p.add_argument("--n_jko_steps",    type=int,   default=10)
    p.add_argument("--n_outer_steps",  type=int,   default=500)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--jko_lr",         type=float, default=1e-2)
    p.add_argument("--w2_backend",     default="sinkhorn", choices=["sinkhorn", "exact"])
    p.add_argument("--sinkhorn_blur",  type=float, default=0.05)
    p.add_argument("--n_particles",    type=int,   default=300)
    p.add_argument("--bip_lambda",     type=float, default=1e-2)
    p.add_argument("--bip_delta_t",    type=float, default=1.0)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)
    results_dir = args.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    config = GRNConfig(
        csv_dir=args.csv_dir,
        tau=args.tau,
        dt=args.dt,
        n_em_steps=args.n_em_steps,
        n_jko_steps=args.n_jko_steps,
        n_outer_steps=args.n_outer_steps,
        lr=args.lr,
        jko_lr=args.jko_lr,
        w2_backend=args.w2_backend,
        sinkhorn_blur=args.sinkhorn_blur,
        n_particles=args.n_particles,
        bip_lambda=args.bip_lambda,
        bip_delta_t=args.bip_delta_t,
        device=args.device,
        seed=args.seed,
    )

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading dataset from '{config.csv_dir}' ...")
    dataset = load_dataset(config.csv_dir)
    timepoints = sorted(dataset.keys())
    G = next(iter(dataset.values())).shape[1]
    print(f"  Timepoints: {timepoints}   G={G}   N={[dataset[k].shape[0] for k in timepoints]}")

    # ------------------------------------------------------------------
    # 2. Compute chi_hat (KDE per timepoint)
    # ------------------------------------------------------------------
    print("Fitting KDEs ...")
    chi_hat = compute_chi_hat(dataset)

    # ------------------------------------------------------------------
    # 3. Build mu_list  (known intervention means, one per interval)
    #    Default: grand mean across all timepoints (single fixed attractor).
    #    Replace with ground-truth values if available.
    #    Note: using mu_k = mean(X_{t_k}) degenerates (driving signal = 0).
    # ------------------------------------------------------------------
    grand_mean = np.mean(np.vstack([dataset[k] for k in timepoints]), axis=0)
    mu_list: list[torch.Tensor] = [
        torch.tensor(grand_mean, dtype=torch.float32, device=config.device)
        for _ in timepoints[:-1]
    ]

    # ------------------------------------------------------------------
    # 4. BIP warm start
    # ------------------------------------------------------------------
    print("Computing BIP warm start ...")
    mu_list_np = [m.numpy() for m in mu_list]
    theta0 = bip_initialise(dataset, mu_list_np, config)
    A_init = theta0["A"].copy()
    print(f"  BIP A (norm={np.linalg.norm(A_init):.3f}):")
    print("  ", np.round(A_init, 3))

    # ------------------------------------------------------------------
    # 5. Outer optimisation
    # ------------------------------------------------------------------
    A_hat, log_sigma_hat, history = optimise(
        theta0, dataset, chi_hat, mu_list, config
    )
    sigma_hat = float(torch.exp(log_sigma_hat))
    print(f"\nFinal sigma = {sigma_hat:.4f}")

    # ------------------------------------------------------------------
    # 6. Write numeric outputs
    # ------------------------------------------------------------------
    A_hat_np = A_hat.numpy()
    np.savetxt(Path(results_dir) / "A_hat.csv", A_hat_np, delimiter=",")
    (Path(results_dir) / "sigma_hat.txt").write_text(str(sigma_hat))
    print(f"Wrote A_hat.csv and sigma_hat.txt to '{results_dir}'")

    gene_names = [f"gene_{i}" for i in range(G)]

    # ------------------------------------------------------------------
    # 7. Standard visualisations
    # ------------------------------------------------------------------
    plot_loss_curve(history, output_dir=results_dir)
    plot_theta_comparison(A_init, A_hat_np, output_dir=results_dir, gene_names=gene_names)
    plot_a_matrix(A_hat_np, output_dir=results_dir, gene_names=gene_names)
    try:
        plot_snapshot_pca(dataset, output_dir=results_dir, rng=rng)
    except ImportError:
        print("scikit-learn not installed; skipping snapshot_pca.png")

    # ------------------------------------------------------------------
    # 8. Ground-truth comparison (optional)
    # ------------------------------------------------------------------
    grn_dir = args.grn_dir or str(_HERE / "output-destructive-measurements")
    try:
        true_grns = load_true_grns(grn_dir)
        print(f"\nLoaded {len(true_grns)} ground-truth GRNs from '{grn_dir}'")

        metrics = compare_grns(A_hat_np, true_grns)
        print("\n--- GRN comparison (vs weighted mean) ---")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}" if isinstance(v, float) else f"  {k:<12}: {v}")

        plot_grn_comparison(A_hat_np, true_grns, output_dir=results_dir,
                            gene_names=gene_names)
        plot_edge_scatter(A_hat_np, true_grns, output_dir=results_dir)
    except Exception as e:
        print(f"\nSkipping ground-truth comparison: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
