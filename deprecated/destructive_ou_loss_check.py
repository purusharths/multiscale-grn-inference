"""
Destructive-measurement multi-timestep OU loss check.

Mirrors multi_step_ou_loss_check.py but loads real snapshots from
output/output-destructive-measurements/expression_by_timepoint/ instead of
generating synthetic OU data.

Only cells from a single population are used (default: pop_0, constant mu)
so that the OU assumption (constant A, mu, sigma) holds approximately.

Usage:
    uv run python -m multsc_grn_inference.destructive_ou_loss_check
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multsc_grn_inference.datagen.run_destructive_measurements import (
    COLLECTION_TIMES,
    POPULATIONS,
    NetworkSimulatorPerGeneMu,
)
from archive.multi_step_ou_loss_check import (
    encode,
    ou_fp_loss,
    optimize,
    total_loss,
)
from multsc_grn_inference.housekeeping.ou_check_plots import (
    plot_A_heatmap,
    plot_comparison,
    plot_convergence,
    plot_phase_portrait,
    plot_recovery,
    plot_snapshots_line,
)


# ---------------------------------------------------------------------------
# Step 1 — load snapshots from per-timepoint CSVs
# ---------------------------------------------------------------------------

def load_snapshots(
    timepoints_dir: Path,
    collection_times: list[float],
    populations: list[str] | None = None,
    tol: float = 0.06,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Load per-timepoint CSVs and return (snapshots, actual_times).

    Each snapshot is an (N, G) float array of gene expression values.
    Files are matched to `collection_times` by nearest-time lookup.
    If `populations` is given, only cells from those populations are kept
    and concatenated; pass None to include all populations.
    `tol` is the maximum allowed time gap between requested and found file.
    """
    # Parse all available files: stem "expression_t0_5283" → 0.5283
    available: dict[float, Path] = {}
    for p in sorted(timepoints_dir.glob("expression_t*.csv")):
        raw = p.stem.replace("expression_t", "", 1)   # "0_5283"
        try:
            t = float(raw.replace("_", ".", 1))        # 0.5283
        except ValueError:
            continue
        available[t] = p

    file_times = np.array(sorted(available.keys()))

    snapshots: list[np.ndarray] = []
    actual_times: list[float] = []
    for ct in collection_times:
        nearest_t = file_times[np.argmin(np.abs(file_times - ct))]
        if abs(nearest_t - ct) > tol:
            raise RuntimeError(
                f"No file within tol={tol} of collection time {ct} "
                f"(nearest found: {nearest_t:.4f})"
            )
        df = pd.read_csv(available[nearest_t])
        if populations is not None:
            df = df[df["population"].isin(populations)]
        gene_cols = [c for c in df.columns if c.startswith("gene_")]
        snapshots.append(df[gene_cols].to_numpy(dtype=float))
        actual_times.append(nearest_t)

    return snapshots, actual_times


# ---------------------------------------------------------------------------
# Step 2 — reconstruct ground-truth parameters for a population
# ---------------------------------------------------------------------------

def get_true_params(
    data_dir: Path,
    pop_label: str,
    pop_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (A, mu0, sigma_eff) for the given population.

    A         : loaded from grn_weighted_<pop_label>.csv
    mu0       : constant target means reconstructed from NetworkSimulatorPerGeneMu
    sigma_eff : scalar sigma ≈ sqrt(mean(diag(D)))
    """
    A_df = pd.read_csv(data_dir / f"grn_weighted_{pop_label}.csv", index_col=0)
    true_A = A_df.to_numpy(dtype=float)

    sim = NetworkSimulatorPerGeneMu(num_genes=true_A.shape[0], seed=pop_seed)
    true_mu = sim.mu0.copy()
    sigma_eff = float(np.sqrt(np.diag(sim.D).mean()))

    return true_A, true_mu, sigma_eff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = Path("output/output-destructive-measurements")
    OUT = Path("output/destructive_ou_check")
    OUT.mkdir(parents=True, exist_ok=True)

    OPT_STEPS = 10   # OU simulation sub-steps per dt interval during optimisation
    G = 4            # number of genes

    # Combine control (pop_0, constant mu) and intervention (pop_1, sigmoid mu
    # on gene_0) into each snapshot — mimics a real mixed-population experiment.
    POPS = ["pop_0", "pop_1"]
    pop_meta = {p["label"]: p for p in POPULATIONS}

    # --- Step 1: load combined snapshots --------------------------------------
    print(f"Loading combined snapshots for {POPS} ...")
    snapshots, actual_times = load_snapshots(
        DATA_DIR / "expression_by_timepoint",
        collection_times=COLLECTION_TIMES,
        populations=POPS,
    )
    dts = [actual_times[i + 1] - actual_times[i] for i in range(len(actual_times) - 1)]
    DT = float(np.mean(dts))
    print(f"  {len(snapshots)} snapshots at t = {[f'{t:.4f}' for t in actual_times]}")
    print(f"  Cells per snapshot: {[len(s) for s in snapshots]}")
    print(f"  Mean DT = {DT:.4f}  (range [{min(dts):.4f}, {max(dts):.4f}])")

    # --- Step 2: ground-truth parameters (pop_0 as reference baseline) --------
    # pop_0 has constant mu so its A/mu are the cleanest OU reference.
    # pop_1 shares the same gene set but has a sigmoid perturbation on gene_0.
    TRUE_A, TRUE_MU, TRUE_SIGMA = get_true_params(
        DATA_DIR, "pop_0", pop_meta["pop_0"]["seed"]
    )
    print(f"\nReference (pop_0) true A:\n{TRUE_A.round(3)}")
    print(f"Reference true mu    = {TRUE_MU.round(3)}")
    print(f"Reference true sigma ≈ {TRUE_SIGMA:.3f}")

    # --- Step 3: line plot of combined snapshots ------------------------------
    plot_snapshots_line(snapshots, DT, out_path=OUT / "line_plot.png",
                        true_A=TRUE_A, true_mu=TRUE_MU)

    # --- Step 4: multi-step optimisation with two loss variants ---------------
    init_theta = encode(np.diag([1.0] * G), np.array([2.0] * G), 0.5)

    loss_fns = {
        "ou+fp":      ou_fp_loss,
        "ou+fp+cons": total_loss,
    }

    results: dict = {}
    histories: dict = {}
    for name, fn in loss_fns.items():
        print(f"\nOptimising [{name}] ...")
        (A_opt, mu_opt, sigma_opt), res, hist = optimize(
            fn, snapshots, DT, init_theta.copy(),
            n_steps=OPT_STEPS,
        )
        results[name] = {
            "A": A_opt,
            "mu": mu_opt,
            "sigma": sigma_opt,
            **{f"a{i}{j}": A_opt[i, j] for i in range(G) for j in range(G)},
            **{f"mu{g}": mu_opt[g] for g in range(G)},
        }
        histories[name] = hist
        print(f"  converged={res.success}  fun={res.fun:.5f}")
        print(f"  A_diag={A_opt.diagonal().round(3)}  mu={mu_opt.round(3)}  sigma={sigma_opt:.3f}")

    # --- Step 5: plots --------------------------------------------------------
    plot_convergence(histories, out_path=OUT / "convergence.png")
    plot_comparison(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                    out_path=OUT / "comparison.png")
    plot_recovery(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                  out_path=OUT / "recovery.png")
    plot_phase_portrait(snapshots, TRUE_A, TRUE_MU, DT,
                        out_path=OUT / "phase_portrait.svg",
                        results=results)
    plot_A_heatmap(results, TRUE_A, out_path=OUT / "A_heatmap.png")

    print(f"\nAll outputs in {OUT.resolve()}")
