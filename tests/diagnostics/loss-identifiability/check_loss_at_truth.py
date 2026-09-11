"""
Is the loss's minimum actually at the true parameters -- or does a fitted
theta just beat the truth on this loss, regardless of how well any
optimizer searches?

Where this came from: refitting OU+Cons/knockout (see
../cma-es-test/README.md) and evaluating the SAME objective at three
points, all at the standard config (n_proj=30, 1500 cells):

    loss at TRUE theta:            0.4495
    loss at Nelder-Mead's point:    0.3051
    loss at CMA-ES's point:         0.2127

The true parameters score *worse* than both fitted points. That's not "no
optimizer has found the truth yet" -- it says the truth isn't competitive
with what a mediocre gradient-free search already found. If that gap is
real, no optimizer (Adam, L-BFGS, HMC, exhaustive search) can fix it by
minimizing this loss harder; a better optimizer would just converge more
confidently to the wrong point, the way CMA-ES already does relative to
Nelder-Mead (../cma-es-test/).

This script tests whether that gap is real or an artifact of approximation
noise, on three separate knobs: sliced-W2's own noise (n_proj random
projections), finite cell count, and Euler-Maruyama discretization
(n_substeps in ou_gene_expression/fp_cell_population -- the OU SDE is
LINEAR, so it has an exact closed-form Gaussian transition; the current
implementation approximates it with a single Euler-Maruyama step by
default). Fits ONE reference theta once (CMA-ES, OU+Cons, standard
config), then RE-EVALUATES loss(true) and loss(reference) -- without
re-fitting -- across each knob in turn. If the gap shrinks toward zero (or
reverses) as any knob gets more precise, that knob's approximation was
(part of) the explanation. If it persists everywhere, the true parameters
really are a worse fit to this data than some other point in parameter
space -- a genuine non-identifiability, not an approximation artifact.

Not a test -- a standalone report script; re-run anytime, overwrites this
folder's loss_at_truth.{csv,png}.

Usage:
    uv run python tests/diagnostics/loss-identifiability/check_loss_at_truth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.fp_cell_population import fp_cell_population
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing

from _intervention_ground_truth import (  # noqa: E402
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)
from _ours_combinations import (  # noqa: E402
    _make_objective,
    decode,
    encode,
    fit_combination_cma,
    off_diag_indices,
)

# ---------------------------------------------------------------------------
# Config -- same scenario as ../jko-testing/, ../sch-bridge-test/,
# ../cma-es-test/ (knockout only; this isn't about the mu_k question, and
# isn't about the control scenario either).
# ---------------------------------------------------------------------------

N_GENES = 8
KO_GENE = 3
SEED = 42
T_STAR = 2.0
NETWORK_DENSITY = 0.5
N_SNAPS = 10
T = 7.0

TERMS = {"ou", "cons"}  # OU+Cons: what the original loss-at-truth check used

N_CELLS_STANDARD = 1500
N_CELLS_MAX = 10000
N_CELLS_SWEEP = [1500, 4000, 10000]
N_PROJ_STANDARD = 30
N_PROJ_SWEEP = [30, 100, 300, 1000]
N_PROJ_FOR_CELLS_SWEEP = 100  # fixed, moderate, while varying n_cells below

N_SUBSTEPS_SWEEP = [1, 5, 20, 100]  # 1 = current default (single Euler-Maruyama step)

CMA_MAXFEVALS = 2200
CMA_SIGMA0 = 0.15

HERE = Path(__file__).parent
OFF = off_diag_indices(N_GENES)


def loss_at(snapshots, mu_known, dt, x: np.ndarray, n_proj: int) -> float:
    chi = [preprocessing(X) for X in snapshots]
    objective = _make_objective(TERMS, snapshots, chi, mu_known, dt, N_GENES, OFF, n_proj, seed=0)
    return objective(x)


def loss_ou_cons_at(snapshots, mu_known, dt, x: np.ndarray, n_proj: int, n_substeps: int) -> float:
    """Reimplements loss_ou + loss_cons (compute_loss.py) with n_substeps
    threaded through ou_gene_expression/fp_cell_population -- those don't
    expose it via _make_objective, since compute_loss.py's loss_ou/loss_cons
    don't accept it either. Mirrors their exact structure: a FRESH seeded
    rng per term (matching _make_objective calling loss_ou(..., seed=seed)
    and loss_cons(..., seed=seed) as two independent calls), just with
    n_substeps as an extra knob."""
    theta = decode(x, mu_known, N_GENES, OFF)
    chi = [preprocessing(X) for X in snapshots]
    total = 0.0

    rng = np.random.default_rng(0)
    for k in range(len(snapshots) - 1):
        propagated = ou_gene_expression(snapshots[k], theta, k, dt, n_substeps=n_substeps, rng=rng)
        propagated_density = preprocessing(propagated)
        n = len(propagated)
        cloud_a = propagated_density.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = chi[k + 1].resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)

    rng = np.random.default_rng(0)
    for k in range(len(snapshots) - 1):
        propagated = ou_gene_expression(snapshots[k], theta, k, dt, n_substeps=n_substeps, rng=rng)
        propagated_density = preprocessing(propagated)
        nu_star = fp_cell_population(chi[k], theta, k, dt, n_substeps=n_substeps, rng=rng)
        n = len(snapshots[k])
        cloud_a = propagated_density.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = nu_star.resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)

    return float(total)


def main() -> None:
    sim = make_single_gene_knockout_sim(
        num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
        t_star=T_STAR, network_density=NETWORK_DENSITY,
    )
    # Draw the largest pool once; every smaller n_cells setting below is a
    # nested subsample of it (a proper convergence study), not an
    # independent redraw.
    pool_snapshots, times, mu_known, dt = extract_snapshots(sim, N_CELLS_MAX, N_SNAPS, T=T)
    standard_snapshots = [X[:N_CELLS_STANDARD] for X in pool_snapshots]

    x_true = encode(sim.A, effective_sigma(sim), OFF)

    print("[reference fit] CMA-ES, OU+Cons, standard config (n_proj=30, 1500 cells) ...", flush=True)
    theta_ref, fit_info = fit_combination_cma(
        TERMS, standard_snapshots, mu_known, dt, N_GENES,
        n_proj=N_PROJ_STANDARD, maxfevals=CMA_MAXFEVALS, sigma0=CMA_SIGMA0, seed=0,
    )
    x_ref = encode(theta_ref.A, theta_ref.sigma, OFF)
    print(f"  reference final_objective (n_proj=30, 1500 cells): {fit_info['final_objective']:.4f}")

    rows = []

    print("\n[n_proj sweep] fixed 1500 cells, no re-fitting ...", flush=True)
    for n_proj in N_PROJ_SWEEP:
        loss_true = loss_at(standard_snapshots, mu_known, dt, x_true, n_proj)
        loss_ref = loss_at(standard_snapshots, mu_known, dt, x_ref, n_proj)
        rows.append({"sweep": "n_proj", "value": n_proj, "loss_true": loss_true, "loss_ref": loss_ref})
        print(f"  n_proj={n_proj:>5}  loss(true)={loss_true:.4f}  loss(ref)={loss_ref:.4f}  "
              f"gap={loss_true - loss_ref:+.4f}")

    print(f"\n[cell-count sweep] fixed n_proj={N_PROJ_FOR_CELLS_SWEEP}, no re-fitting ...", flush=True)
    for n_cells in N_CELLS_SWEEP:
        sub_snapshots = [X[:n_cells] for X in pool_snapshots]
        loss_true = loss_at(sub_snapshots, mu_known, dt, x_true, N_PROJ_FOR_CELLS_SWEEP)
        loss_ref = loss_at(sub_snapshots, mu_known, dt, x_ref, N_PROJ_FOR_CELLS_SWEEP)
        rows.append({"sweep": "n_cells", "value": n_cells, "loss_true": loss_true, "loss_ref": loss_ref})
        print(f"  n_cells={n_cells:>6}  loss(true)={loss_true:.4f}  loss(ref)={loss_ref:.4f}  "
              f"gap={loss_true - loss_ref:+.4f}")

    print(f"\n[n_substeps sweep] fixed n_proj={N_PROJ_STANDARD}, {N_CELLS_STANDARD} cells, "
          f"no re-fitting -- is single-step Euler-Maruyama the culprit? ...", flush=True)
    for n_substeps in N_SUBSTEPS_SWEEP:
        loss_true = loss_ou_cons_at(standard_snapshots, mu_known, dt, x_true, N_PROJ_STANDARD, n_substeps)
        loss_ref = loss_ou_cons_at(standard_snapshots, mu_known, dt, x_ref, N_PROJ_STANDARD, n_substeps)
        rows.append({"sweep": "n_substeps", "value": n_substeps, "loss_true": loss_true, "loss_ref": loss_ref})
        print(f"  n_substeps={n_substeps:>4}  loss(true)={loss_true:.4f}  loss(ref)={loss_ref:.4f}  "
              f"gap={loss_true - loss_ref:+.4f}")

    df = pd.DataFrame(rows)
    csv_path = HERE / "loss_at_truth.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df)


def _plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, sweep, xlabel in [
        (axes[0], "n_proj", "n_proj (sliced-W2 random projections)"),
        (axes[1], "n_cells", "n_cells (evaluation sample size)"),
        (axes[2], "n_substeps", "n_substeps (Euler-Maruyama steps/interval)"),
    ]:
        sub = df[df.sweep == sweep].sort_values("value")
        ax.plot(sub.value, sub.loss_true, "o-", color="#a8442f", label="loss(TRUE theta)")
        ax.plot(sub.value, sub.loss_ref, "o-", color="#2a6b67", label="loss(fitted theta)")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("OU+Cons loss")
        ax.legend(fontsize=9)
        ax.set_title(f"Fixed the other axis; varying {sweep}", fontsize=10)

    fig.suptitle(
        "Does more precision close the true-vs-fitted loss gap, or does it persist?\n"
        "(single-gene knockout, 8 genes, density=0.5, OU+Cons)"
    )
    fig.tight_layout()
    png_path = HERE / "loss_at_truth.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
