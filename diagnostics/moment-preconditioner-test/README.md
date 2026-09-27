# Moment-matching preconditioner for CMA-ES

Does giving CMA-ES a data-informed starting point (`x0`), instead of the same
generic `diag(1.2)/all-zero-off-diagonal/sigma=0.5` guess every fit currently
uses, get it to a better recovered `A` -- or get there in fewer evaluations --
on `../non-interventional/`'s dataset?

## What led here

`../non-interventional/compare_losses_stationary.py` switched from
Nelder-Mead to CMA-ES (see `../cma-es-test/README.md`) and started recovering
real off-diagonal structure instead of a diagonal-only fit, but 3 of 6 loss
combinations still only barely beat chance, and none get close to the true
`A` in magnitude (`A_err` ~2.5-3.6 regardless of combination). CMA-ES still
starts completely blind to the data every time; a "Bayesian preconditioner"
was proposed as a way to start it closer to the truth instead.

A full per-particle Kalman filter was considered and rejected: it would need
per-cell correspondence ACROSS snapshots, which real scRNA-seq (and the
OU/FP/Cons losses this branch is testing) doesn't have -- cells are
destroyed at each measurement. The synthetic generator here
(`../non-interventional/_stationary_ground_truth.py`) happens to track the
same particles under the hood for simulation convenience, but using that
would leak information the actual algorithm never has access to.

## Two failed attempts, one that works

`_moment_preconditioner.py`'s docstring has the full account; short version:

1. **Mean trajectory, Euler-linearized** (`dm/dt = A(mu-m)`, regressed by
   finite difference): only needs per-snapshot sample means, so it's
   unpaired-snapshot-safe -- but this dataset's `PERTURBATION="uniform"`
   displaces every cell the same way, so the population mean decays along
   essentially one direction. Regressing a full 8x8 `A` off that is
   rank-deficient: `edge_corr` came back ~0.28 but `offdiag_err` (~4.99) was
   *worse* than just leaving the off-diagonal at zero (~1.90).
2. **Mean trajectory, exact transition** (matrix logarithm of the fitted
   discrete-time map, no Euler approximation): made it worse, not better
   (`offdiag_err` ~13.5) -- `logm` amplifies whatever noise is in the
   already-underdetermined regression.
3. **Covariance trajectory** (`dSigma/dt = -A@Sigma - Sigma@A.T + sigma^2*I`,
   linear in `A` for a known `Sigma` -- see `_cov_drift_operator`): not
   limited by the mean's rank, because `Sigma`'s off-diagonal growth rate
   depends on every pair of genes' coupling through `A`, not on which single
   direction the mean moved. This is the one that works: `edge_corr` ~0.41
   (on par with a full CMA-ES fit run to convergence from the cold start)
   and `offdiag_err` ~1.71, below the cold start's implicit ~1.90. The
   overall diagonal *scale* still comes back biased low (first-order
   discretization at `dt=0.5`, not small next to this dataset's own
   relaxation time -- `dt/tau=0.53` per `../non-interventional/`'s dataset
   stats), which is expected and fine for a warm start: CMA-ES's own search
   is what should fix scale while (hopefully) preserving the structure.

Both the failed mean-based attempts and the working covariance-based one
only ever use per-snapshot sample statistics (mean or covariance) -- never
per-cell correspondence across snapshots -- so all three respect the
unpaired-snapshot assumption; the difference is entirely about which moment
carries enough independent directions to identify a full `A` under this
dataset's particular perturbation.

## Setup

Same dataset as `../non-interventional/compare_losses_stationary.py`'s
default config (8 genes, density=0.3, constant mu, `data_seed=42`).
`fit_combination_cma`'s `x0` parameter (added in
`../interventions/_ours_combinations.py`) is the only thing that differs
between the two fits per combination -- `maxfevals=2200`, `sigma0=0.15`, and
everything else stays matched.

## Files

- `_moment_preconditioner.py` -- `estimate_theta0(snapshots, mu_known, dt,
  n_genes)`, the covariance-based closed-form estimator (zero optimizer
  cost).
- `compare_preconditioner.py` -- fits all 6 combinations twice each (`cold`
  vs `preconditioned` `x0`), matched budget, scores both against the true
  `A`, and reports the preconditioner's own zero-cost accuracy alongside.
  Saves `preconditioner_comparison.{csv,png}` and
  `preconditioner_recovered_matrices.png`.

```bash
uv run python tests/diagnostics/moment-preconditioner-test/compare_preconditioner.py --jobs 12
```
