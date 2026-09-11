"""
Shared config, dataset, fitting and plotting for the single-gene-knockout
loss-combination comparison.

Split out of compare_losses_single_gene_knockout.py so the same code can run
two ways without drifting apart:

  - serial          : compare_losses_single_gene_knockout.py with no args,
                      fits all six combinations in one process (~3h)
  - array-parallel  : the same script with --combination NAME, one process per
                      combination writing a part file, then
                      merge_knockout_parts.py to assemble (~longest single
                      combination, so ~6x faster in wall clock)

The six fits are completely independent -- they share only the dataset, which
every task rebuilds deterministically from DATA_SEED -- so there is nothing to
communicate between them and no reason to run them in sequence when cores are
available.

Nothing here executes on import; the entry points call build_dataset().
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from multsc_grn_inference.housekeeping.edge_recovery_metrics import (
    auprc,
    auroc,
    edge_labels,
    precision_at_k,
    recall_at_k,
)
from multsc_grn_inference.theta import Theta

from _intervention_ground_truth import (
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)
from _ours_combinations import COMBINATIONS, fit_combination, off_diag_indices

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Every constant can be overridden by a GRN_<NAME> environment variable, so a
# cluster sweep over gene count / density / budget needs no code edit and all
# array tasks in one submission are guaranteed to agree on the config.
#   GRN_N_GENES=12 GRN_NETWORK_DENSITY=0.25 sbatch run_knockout_array.sh


def _env(name, default, cast=float):
    raw = os.environ.get(f"GRN_{name}")
    return default if raw is None else cast(raw)


N_GENES = _env("N_GENES", 8, int)
KO_GENE = _env("KO_GENE", 3, int)
N_CELLS = _env("N_CELLS", 3000, int)
N_SNAPS = _env("N_SNAPS", 15, int)
T = _env("T", 7.0)
T_STAR = _env("T_STAR", 2.0)
NETWORK_DENSITY = _env("NETWORK_DENSITY", 0.5)

# Two independent seeds, deliberately separate:
#   DATA_SEED -- generates the network A, mu0, D and the cells. Varying it
#                asks "does this work on GRNs in general?"
#   FIT_SEED  -- the objective's own stochasticity (sliced-W2 projection
#                directions, KDE resampling). Varying it asks "is one fit
#                reproducible?" -- a weaker question.
# Sweeping DATA_SEED is the one worth spending cores on: every result in this
# branch so far comes from the single network at DATA_SEED=42.
DATA_SEED = _env("DATA_SEED", 42, int)
FIT_SEED = _env("FIT_SEED", 0, int)

N_PROJ = _env("N_PROJ", 30, int)
MAXITER = _env("MAXITER", 2000, int)   # same budget for every combination
OPTIMIZER_METHOD = os.environ.get("GRN_OPTIMIZER", "Nelder-Mead")


@dataclass
class Dataset:
    snapshots: list
    times: np.ndarray
    mu_known: list
    dt: float
    A_true: np.ndarray
    sigma_true: float
    off: list
    true_off: np.ndarray
    k_edges: int


def build_dataset(data_seed: int = DATA_SEED) -> Dataset:
    """Deterministic from data_seed, so every array task builds the same thing."""
    sim = make_single_gene_knockout_sim(
        num_genes=N_GENES, knockout_gene=KO_GENE, seed=data_seed,
        t_star=T_STAR, network_density=NETWORK_DENSITY,
    )
    snapshots, times, mu_known, dt = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)
    off = off_diag_indices(N_GENES)
    true_off = np.array([sim.A[i, j] for i, j in off])
    return Dataset(
        snapshots=snapshots, times=times, mu_known=mu_known, dt=dt,
        A_true=sim.A, sigma_true=effective_sigma(sim), off=off, true_off=true_off,
        # k for precision@k / recall@k and for thresholding the drawn networks:
        # the number of edges that actually exist, so each method is judged on
        # the same edge budget as the truth.
        k_edges=int(edge_labels(true_off).sum()),
    )


def score(name: str, theta_hat: Theta, info: dict, d: Dataset,
          data_seed: int, fit_seed: int) -> dict:
    """One CSV row's worth of metrics for a fitted theta."""
    hat_off = np.array([theta_hat.A[i, j] for i, j in d.off])
    return {
        "combination": name,
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "A_err": float(np.linalg.norm(theta_hat.A - d.A_true)),
        "offdiag_err": float(np.linalg.norm(hat_off - d.true_off)),
        "edge_corr": (float(np.corrcoef(d.true_off, hat_off)[0, 1])
                      if hat_off.std() > 1e-12 else 0.0),
        # Detection metrics: rank candidate edges by |A_hat_ij| and score that
        # ranking against which entries really are edges. edge_corr conflates
        # "found the right pairs" with "got the magnitudes right"; these are
        # what the GRN-inference literature (DREAM4/5) reports.
        "auprc": auprc(d.true_off, hat_off),
        "auroc": auroc(d.true_off, hat_off),
        "precision_at_k": precision_at_k(d.true_off, hat_off, d.k_edges),
        "recall_at_k": recall_at_k(d.true_off, hat_off, d.k_edges),
        "sigma_err": abs(theta_hat.sigma - d.sigma_true),
        "final_objective": info["final_objective"],
        "n_evals": info["n_evals"],
        "seconds": info["seconds"],
    }


def run_one(name: str, d: Dataset, *, fit_seed: int = FIT_SEED,
            data_seed: int = DATA_SEED) -> tuple[Theta, dict]:
    """Fit one combination and score it. Returns (theta_hat, row)."""
    theta_hat, info = fit_combination(
        COMBINATIONS[name], d.snapshots, d.mu_known, d.dt, N_GENES,
        n_proj=N_PROJ, maxiter=MAXITER, method=OPTIMIZER_METHOD, seed=fit_seed,
    )
    return theta_hat, score(name, theta_hat, info, d, data_seed, fit_seed)


def summary_line(row: dict) -> str:
    return (f"  A_err={row['A_err']:.3f}  edge_corr={row['edge_corr']:+.3f}  "
            f"AUPRC={row['auprc']:.3f}  AUROC={row['auroc']:.3f}  "
            f"P@k={row['precision_at_k']:.3f}  sigma_err={row['sigma_err']:.3f}  "
            f"evals={row['n_evals']}  {row['seconds']:.0f}s")
