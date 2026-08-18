"""
Method-of-moments (MOM) initialiser for the 7-parameter OU model.

Given K cross-sectional snapshots at uniform Δt, estimates (A, μ, σ) by
matching empirical first and second moments — no simulation required.

Algorithm
---------
1. μ̂  = sample mean of the last snapshot  (near-equilibrium proxy)
2. A   from mean-velocity regression across all K−1 intervals:
       (m_{k+1} − m_k) / dt  ≈  A (μ̂ − m_k)
   → ordinary least squares on the G×(K−1) system, one row of A at a time
3. σ̂  from the Lyapunov identity at (approximate) stationarity:
       σ²·I  ≈  A·Σ_last + Σ_last·A^T  →  σ = √(mean(diag(A·Σ + Σ·A^T)))
4. Encode (A, μ, σ) as the 7-element θ used in loss_diagnostic.py:
       θ = [log_a00, r01, r10, log_a11, μ0, μ1, log_σ]
   with off-diagonals: a_ij = tanh(r_ij) × 0.9 × a_ii

Usage
-----
    uv run python -m multsc_grn_inference.mom_init

Outputs in output/loss_diagnostic/mom_comparison/:
    comparison.csv
    parameter_recovery_mom.png
    parameter_recovery_generic.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from archive.loss_diagnostic import (
    LOSS_FNS,
    LOSS_ORDER,
    N_CELLS,
    OUT_DIR,
    T_SNAPS,
    _theta_to_named,
    optimise_loss,
    plot_recovery,
    plot_loss_values,
)


# ---------------------------------------------------------------------------
# Core MOM estimation
# ---------------------------------------------------------------------------

def mom_estimate(
    snapshots: list[np.ndarray],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit (A, μ, σ) from snapshot moments.

    Parameters
    ----------
    snapshots : list of (N, G) arrays, length K
    dt        : uniform time interval between consecutive snapshots

    Returns
    -------
    A_hat     : (G, G)  — not guaranteed diagonal-dominant; caller should clip
    mu_hat    : (G,)
    sigma_hat : float
    """
    K = len(snapshots)
    means = [s.mean(axis=0) for s in snapshots]

    # Step 1: μ proxy = mean of last snapshot
    mu_hat = means[-1].copy()

    # Step 2: mean-velocity regression
    #   feature[k] = μ_hat − m_k   (the displacement from current mean to target)
    #   target[k]  = Δm_k / dt     (observed mean velocity)
    #   Solve: features @ A^T ≈ velocities  (K−1 equations, G unknowns per row)
    features   = np.array([mu_hat - means[k]             for k in range(K - 1)])  # (K-1, G)
    velocities = np.array([(means[k+1] - means[k]) / dt  for k in range(K - 1)])  # (K-1, G)

    cond = np.linalg.cond(features)
    if cond > 1e4:
        print(f"  [MOM] feature matrix condition number = {cond:.1f}  "
              f"(genes move collinearly — off-diagonal A may be unreliable)")

    A_hat, _, _, _ = np.linalg.lstsq(features, velocities, rcond=None)
    A_hat = A_hat.T   # (G, G): row j of A_hat fits gene j

    # Ensure positive diagonal (OU stability)
    for i in range(A_hat.shape[0]):
        A_hat[i, i] = max(float(A_hat[i, i]), 0.05)

    # Step 3: σ from Lyapunov at stationarity
    #   D = σ²·I  ≈  A·Σ_last + Σ_last·A^T
    Sigma_last = np.cov(snapshots[-1].T)
    D_approx   = A_hat @ Sigma_last + Sigma_last @ A_hat.T
    sigma_hat  = float(np.sqrt(max(np.mean(np.diag(D_approx)), 1e-6)))

    return A_hat, mu_hat, sigma_hat


def encode_theta(A: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """
    Pack (A, μ, σ) into the 7-element θ used by loss_diagnostic._decode.

    θ = [log_a00, r01, r10, log_a11, μ0, μ1, log_σ]
    where  a_ij = tanh(r_ij) × 0.9 × a_ii  (off-diagonal constraint).
    """
    a00, a01 = float(A[0, 0]), float(A[0, 1])
    a10, a11 = float(A[1, 0]), float(A[1, 1])

    r01 = float(np.arctanh(np.clip(a01 / (0.9 * a00 + 1e-9), -0.99, 0.99)))
    r10 = float(np.arctanh(np.clip(a10 / (0.9 * a11 + 1e-9), -0.99, 0.99)))

    return np.array([
        np.log(max(a00, 0.01)),
        r01,
        r10,
        np.log(max(a11, 0.01)),
        float(mu[0]),
        float(mu[1]),
        np.log(max(sigma, 1e-4)),
    ])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_snapshots(snap_dir: Path) -> tuple[list[np.ndarray], float]:
    """Load snapshots and dt from the CSVs written by loss_diagnostic."""
    meta = pd.read_csv(snap_dir / "metadata.csv")
    dt   = float(meta["dt"].iloc[0])
    snaps, i = [], 0
    while (snap_dir / f"snapshot_t{i}.csv").exists():
        snaps.append(pd.read_csv(snap_dir / f"snapshot_t{i}.csv").values.astype(float))
        i += 1
    return snaps, dt


# ---------------------------------------------------------------------------
# Main: compare MOM-initialised vs generic-initialised optimization
# ---------------------------------------------------------------------------

def main() -> None:
    snap_dir = OUT_DIR / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(
            f"{snap_dir} not found — run loss_diagnostic.py first:\n"
            "    uv run python -m multsc_grn_inference.loss_diagnostic"
        )

    snaps, dt = load_snapshots(snap_dir)
    true_params = pd.read_csv(OUT_DIR / "true_params.csv").iloc[0].to_dict()

    print(f"Loaded {len(snaps)} snapshots, dt={dt}, "
          f"N={snaps[0].shape[0]} cells, G={snaps[0].shape[1]} genes\n")

    # --- MOM estimate -------------------------------------------------------
    print("[MOM] Estimating parameters from moments ...")
    A_hat, mu_hat, sigma_hat = mom_estimate(snaps, dt)
    theta_mom = encode_theta(A_hat, mu_hat, sigma_hat)

    print(f"  A_hat:\n{A_hat.round(3)}")
    print(f"  μ_hat = {mu_hat.round(3)},  σ_hat = {sigma_hat:.3f}")
    print(f"  θ_mom = {theta_mom.round(3)}")
    print(f"\n  True A:\n"
          f"  [[{true_params['a00']:.3f}, {true_params['a01']:.3f}],\n"
          f"   [{true_params['a10']:.3f}, {true_params['a11']:.3f}]]")
    print(f"  True μ = [{true_params['mu0']:.3f}, {true_params['mu1']:.3f}],  "
          f"σ = {true_params['sigma']:.3f}")

    # Generic starting point (same as loss_diagnostic.py)
    theta_generic = np.array([np.log(1.0), 0.0, 0.0, np.log(1.0), 2.0, 2.0, np.log(0.3)])

    # --- Optimise from both starts ------------------------------------------
    out_dir = OUT_DIR / "mom_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for start_name, theta_init in [("mom", theta_mom), ("generic", theta_generic)]:
        print(f"\n[Optimising from {start_name} start]")
        results[start_name] = {}
        for name, loss_fn in LOSS_FNS.items():
            print(f"  [{name}]", flush=True)
            theta_opt, fun = optimise_loss(loss_fn, snaps, dt, theta_init.copy())
            results[start_name][name] = {**_theta_to_named(theta_opt), "objective": fun}

    # --- Save and display comparison ----------------------------------------
    rows = []
    for start_name, res in results.items():
        for loss_name, vals in res.items():
            rows.append({"start": start_name, "loss": loss_name, **vals})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison.csv", index=False)

    params = list(true_params.keys())
    header = f"{'loss':<12} {'start':<8} " + " ".join(f"{p:>7}" for p in params)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    true_row = "  true         " + " ".join(f"{true_params[p]:>7.3f}" for p in params)
    print(true_row)
    print("-" * len(header))
    for start_name in ("mom", "generic"):
        for loss_name in LOSS_ORDER:
            vals = results[start_name][loss_name]
            row = f"{loss_name:<12} {start_name:<8} " + " ".join(
                f"{vals[p]:>7.3f}" for p in params
            )
            print(row)
    print("=" * len(header))

    # --- Recovery plots ------------------------------------------------------
    plot_recovery(results["mom"],     true_params, out_dir / "parameter_recovery_mom.png")
    plot_recovery(results["generic"], true_params, out_dir / "parameter_recovery_generic.png")
    plot_loss_values(results["mom"],     out_dir / "loss_values_mom.png")
    plot_loss_values(results["generic"], out_dir / "loss_values_generic.png")

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
