"""
Shared "ours" (Algorithm 1) fitting logic for the JKOnet* / Schrödinger-
bridge comparison diagnostics (../jko-testing/, ../sch-bridge-test/).

Full A + sigma, mu fixed at the known {mu_k} -- Algorithm 1 takes them as
given input ("Data: Known Intervention means", paper line 2) -- same
parametrization as
single-gene-knockout/loss-combinations/compare_losses_single_gene_knockout.py.

COMBINATIONS matches that script's sweep exactly, including the exclusion
of Cons-alone: it only checks that the OU and FP forward models agree with
EACH OTHER, never with the data, so it's trivially minimized by wrong
dynamics (see that script's module docstring for the full argument).

An earlier version of the two comparison scripts picked a single "best"
combination (OU+Cons, by edge_corr in one past single-seed sweep) to stand
in for "the current implementation". That was weaker than it looked: every
combination's A_err/offdiag_err in that sweep was nearly identical
(~1.99 / ~1.645), edge_corr was the only thing that moved, and it moved
across a single noisy run per combination -- not a real ranking. Comparing
all six here instead of picking one avoids leaning on that noise.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.optimize

from multsc_grn_inference.compute_loss import _sliced_w2, loss_cons, loss_fp, loss_ou
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

COMBINATIONS = {
    "OU":          {"ou"},
    "FP":          {"fp"},
    "OU+FP":       {"ou", "fp"},
    "OU+Cons":     {"ou", "cons"},
    "FP+Cons":     {"fp", "cons"},
    "OU+FP+Cons":  {"ou", "fp", "cons"},
}


def off_diag_indices(n_genes: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_genes) for j in range(n_genes) if i != j]


def encode(A: np.ndarray, sigma: float, off: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([np.log(np.diag(A)), [A[i, j] for i, j in off], [np.log(sigma)]])


def decode(x: np.ndarray, mu_known, n_genes: int, off: list[tuple[int, int]]) -> Theta:
    A = np.diag(np.exp(x[:n_genes]))
    for n, (i, j) in enumerate(off):
        A[i, j] = x[n_genes + n]
    return Theta(A=A, mu=mu_known, sigma=float(np.exp(x[-1])))


def _initial_simplex(x0: np.ndarray, delta: float = 0.06) -> np.ndarray:
    """
    scipy's default Nelder-Mead initial simplex perturbs each x0[k] by
    x0[k]*1.05, EXCEPT where x0[k]==0, where it uses a hardcoded absolute
    step of 0.00025 (scipy.optimize._optimize._minimize_neldermead:
    nonzdelt=0.05, zdelt=0.00025). x0 here (see fit_combination) sets all
    56 off-diagonal entries to exactly 0.0 -- so scipy's own simplex has an
    edge length of 0.00025 in exactly those directions, ~2000-4000x smaller
    than the true off-diagonal scale (up to ~0.9). That made those
    directions effectively unsearchable within the evaluation budget: the
    "recovered A is diagonal-only regardless of loss combination" finding
    (../jko-testing/jko_recovered_matrices.png) traces back to this, not to
    the loss landscape itself -- confirmed by refitting and reading out
    A_hat directly (every off-diagonal entry came back within 1e-4 of its
    starting value of exactly 0).

    Rebuilds scipy's own construction, but with an absolute delta for
    zero-valued coordinates instead of scipy's 0.00025. Default delta=0.06
    matches the step scipy's own nonzero branch gives the diagonal's ~1.2
    starting value (1.2 * 0.05 = 0.06), so off-diagonal and diagonal
    directions get comparably-sized room to move.
    """
    n = len(x0)
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for k in range(n):
        y = x0.copy()
        y[k] = y[k] * 1.05 if y[k] != 0 else delta
        sim[k + 1] = y
    return sim


def _make_objective(terms, snapshots, chi, mu_known, dt, n_genes, off, n_proj, seed):
    def objective(x: np.ndarray) -> float:
        theta = decode(x, mu_known, n_genes, off)
        try:
            total = 0.0
            if "ou" in terms:
                total += loss_ou(theta, snapshots, chi, dt, n_proj=n_proj, seed=seed)
            if "fp" in terms:
                total += loss_fp(theta, snapshots, chi, dt, n_proj=n_proj, seed=seed)
            if "cons" in terms:
                total += loss_cons(theta, snapshots, chi, dt, n_proj=n_proj, seed=seed)
            return total
        except np.linalg.LinAlgError:
            # A candidate theta can be unstable enough that propagated
            # cells collapse onto a lower-dimensional subspace (e.g. one
            # gene clipped to its floor everywhere via np.maximum), making
            # gaussian_kde's covariance singular. Nelder-Mead's simplex
            # never explores far enough from x0 to hit this; CMA-ES's
            # sampling does. NaN is CMA-ES's documented "reject this
            # candidate, resample, don't count it against the budget"
            # signal (cma.fmin2's docstring); scipy's Nelder-Mead also
            # tolerates it safely since NaN never compares as "better".
            return np.nan
    return objective


def fit_combination(
    terms: set[str],
    snapshots: list[np.ndarray],
    mu_known,
    dt: float,
    n_genes: int,
    *,
    n_proj: int = 30,
    maxiter: int = 1500,
    method: str = "Nelder-Mead",
    seed: int = 0,
) -> tuple[Theta, dict]:
    off = off_diag_indices(n_genes)
    chi = [preprocessing(X) for X in snapshots]
    objective = _make_objective(terms, snapshots, chi, mu_known, dt, n_genes, off, n_proj, seed)

    x0 = encode(np.eye(n_genes) * 1.2, 0.5, off)
    options = {"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6, "adaptive": True}
    if method == "Nelder-Mead":
        # scipy's own initial-simplex construction is degenerate wherever
        # x0==0 -- see _initial_simplex's docstring. Every off-diagonal
        # entry starts at exactly 0, so this matters a lot here.
        options["initial_simplex"] = _initial_simplex(x0)
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(objective, x0, method=method, options=options)
    elapsed = time.perf_counter() - t0
    theta_hat = decode(res.x, mu_known, n_genes, off)
    return theta_hat, {"n_evals": res.nfev, "seconds": round(elapsed, 1), "final_objective": res.fun}


def fit_combination_cma(
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
) -> tuple[Theta, dict]:
    """
    CMA-ES instead of Nelder-Mead -- see ../cma-es-test/README.md for why.
    Short version: fixing Nelder-Mead's degenerate initial simplex (see
    _initial_simplex) recovered real signal (edge_corr -0.02 -> +0.36 on
    one refit) but then plateaued well short of full recovery even at 4x
    the evaluation budget -- consistent with plain Nelder-Mead just not
    covering a 65-dimensional non-convex landscape well, initialization
    aside. CMA-ES's sampling step (sigma0) is a single explicit scalar,
    independent of x0's own per-coordinate values, so it can't inherit
    Nelder-Mead's specific "zero-valued coordinates get a frozen step"
    failure mode -- and its adapted covariance is exactly the mechanism
    Nelder-Mead's simplex only approximates badly in high dimensions.

    maxfevals=2200 matches fit_combination's typical evaluation count at
    maxiter=1500 (1978-2248 evals across the 6 combinations in
    ../jko-testing/jko_comparison.csv), for a budget-matched comparison.
    """
    import cma

    off = off_diag_indices(n_genes)
    chi = [preprocessing(X) for X in snapshots]
    objective = _make_objective(terms, snapshots, chi, mu_known, dt, n_genes, off, n_proj, seed)

    x0 = encode(np.eye(n_genes) * 1.2, 0.5, off)
    t0 = time.perf_counter()
    xbest, es = cma.fmin2(
        objective, x0, sigma0,
        # cma treats seed=0 (and None) as "seed from the clock", not a
        # literal seed -- offset by 1 so seed=0 (this function's default)
        # is still reproducible.
        options={"maxfevals": maxfevals, "seed": seed + 1, "verbose": -9},
    )
    elapsed = time.perf_counter() - t0
    theta_hat = decode(xbest, mu_known, n_genes, off)
    return theta_hat, {
        "n_evals": es.result.evaluations, "seconds": round(elapsed, 1),
        "final_objective": float(es.result.fbest),
    }


def one_step_ahead_w2(theta_hat: Theta, eval_snapshots: list[np.ndarray], dt: float, *, n_proj: int = 30, seed: int = 0) -> float:
    """Roll the eval snapshot at t_k one interval forward under theta_hat,
    sliced-W2 against the eval snapshot at t_{k+1}."""
    rng = np.random.default_rng(seed)
    errs = []
    for k in range(len(eval_snapshots) - 1):
        pred = ou_gene_expression(eval_snapshots[k], theta_hat, k, dt, rng=rng)
        errs.append(_sliced_w2(pred, eval_snapshots[k + 1], n_proj, rng))
    return float(np.mean(errs))
