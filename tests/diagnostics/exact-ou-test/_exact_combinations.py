"""
Same fitting logic as ../interventions/_ours_combinations.py (same
encode/decode, same COMBINATIONS, same Nelder-Mead simplex fix), but the
loss uses exact_ou_transition/exact_fp_cell_population (this folder) in
place of ou_gene_expression/fp_cell_population -- i.e. loss_ou/loss_fp/
loss_cons reimplemented with the exact OU transition instead of
Euler-Maruyama. See exact_ou.py and README.md for why.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize

sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.preprocessing import preprocessing

from _ours_combinations import (  # noqa: E402
    COMBINATIONS,
    _initial_simplex,
    decode,
    encode,
    off_diag_indices,
)

from exact_ou import exact_fp_cell_population, exact_ou_transition  # noqa: E402


def _loss_ou_exact(theta, snapshots, chi, dt, n_proj, seed):
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(snapshots) - 1):
        propagated = exact_ou_transition(snapshots[k], theta, k, dt, rng=rng)
        propagated_density = preprocessing(propagated)
        n = len(propagated)
        cloud_a = propagated_density.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = chi[k + 1].resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)
    return total


def _loss_fp_exact(theta, snapshots, chi, dt, n_proj, seed):
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(chi) - 1):
        nu_star = exact_fp_cell_population(chi[k], theta, k, dt, rng=rng)
        n = len(snapshots[k])
        cloud_a = nu_star.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = chi[k + 1].resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)
    return total


def _loss_cons_exact(theta, snapshots, chi, dt, n_proj, seed):
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(len(snapshots) - 1):
        propagated = exact_ou_transition(snapshots[k], theta, k, dt, rng=rng)
        propagated_density = preprocessing(propagated)
        nu_star = exact_fp_cell_population(chi[k], theta, k, dt, rng=rng)
        n = len(snapshots[k])
        cloud_a = propagated_density.resample(n, seed=int(rng.integers(0, 2**31))).T
        cloud_b = nu_star.resample(n, seed=int(rng.integers(0, 2**31))).T
        total += _sliced_w2(cloud_a, cloud_b, n_proj, rng)
    return total


def _make_exact_objective(terms, snapshots, chi, mu_known, dt, n_genes, off, n_proj, seed):
    def objective(x: np.ndarray) -> float:
        theta = decode(x, mu_known, n_genes, off)
        try:
            total = 0.0
            if "ou" in terms:
                total += _loss_ou_exact(theta, snapshots, chi, dt, n_proj, seed)
            if "fp" in terms:
                total += _loss_fp_exact(theta, snapshots, chi, dt, n_proj, seed)
            if "cons" in terms:
                total += _loss_cons_exact(theta, snapshots, chi, dt, n_proj, seed)
            return total
        except np.linalg.LinAlgError:
            return np.nan
    return objective


def fit_combination_exact(
    terms: set[str],
    snapshots: list[np.ndarray],
    mu_known,
    dt: float,
    n_genes: int,
    *,
    n_proj: int = 30,
    maxiter: int = 1500,
    seed: int = 0,
):
    off = off_diag_indices(n_genes)
    chi = [preprocessing(X) for X in snapshots]
    objective = _make_exact_objective(terms, snapshots, chi, mu_known, dt, n_genes, off, n_proj, seed)

    x0 = encode(np.eye(n_genes) * 1.2, 0.5, off)
    options = {
        "maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6, "adaptive": True,
        "initial_simplex": _initial_simplex(x0),
    }
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(objective, x0, method="Nelder-Mead", options=options)
    elapsed = time.perf_counter() - t0
    theta_hat = decode(res.x, mu_known, n_genes, off)
    return theta_hat, {"n_evals": res.nfev, "seconds": round(elapsed, 1), "final_objective": res.fun}


def fit_combination_cma_exact(
    terms: set[str],
    snapshots: list[np.ndarray],
    mu_known,
    dt: float,
    n_genes: int,
    *,
    n_proj: int = 30,
    maxfevals: int = 2200,
    sigma0: float = 0.15,
    seed: int = 0,
):
    """Same as ../cma-es-test/'s fit_combination_cma, but scoring the exact
    OU transition instead of Euler-Maruyama -- see README.md. Tests whether
    Nelder-Mead specifically struggles with the exact-transition landscape
    (in which case CMA-ES should do noticeably better here) or whether the
    exact loss is just harder to optimize regardless of search method."""
    import cma

    off = off_diag_indices(n_genes)
    chi = [preprocessing(X) for X in snapshots]
    objective = _make_exact_objective(terms, snapshots, chi, mu_known, dt, n_genes, off, n_proj, seed)

    x0 = encode(np.eye(n_genes) * 1.2, 0.5, off)
    t0 = time.perf_counter()
    xbest, es = cma.fmin2(
        objective, x0, sigma0,
        options={"maxfevals": maxfevals, "seed": seed + 1, "verbose": -9},
    )
    elapsed = time.perf_counter() - t0
    theta_hat = decode(xbest, mu_known, n_genes, off)
    return theta_hat, {
        "n_evals": es.result.evaluations, "seconds": round(elapsed, 1),
        "final_objective": float(es.result.fbest),
    }
