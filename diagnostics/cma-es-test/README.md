# Nelder-Mead vs CMA-ES

Not a mu_k/baseline comparison like `../jko-testing/` or `../sch-bridge-test/` --
this one stays entirely on "our" side. It tests whether the optimizer, not
the loss or the data, was the reason Algorithm 1's fitted A came back
diagonal-only (see the published report from this branch's session,
`jko_recovered_matrices.png`, `sch_bridge_recovered_matrices.png`).

## What led here

1. All 6 loss combinations, fit with `scipy.optimize.minimize(...,
   method="Nelder-Mead")`, converged to a recovered A indistinguishable
   from the starting guess (`x0 = 1.2*I`) -- off-diagonal entries came back
   ~1e-4, five orders of magnitude below their true values.
2. Root cause found: scipy's default Nelder-Mead initial simplex perturbs
   each `x0[k]` by `x0[k]*1.05`, **except** where `x0[k]==0`, where it uses
   a hardcoded absolute step of `0.00025`
   (`scipy.optimize._optimize._minimize_neldermead`, `zdelt`). Every
   off-diagonal entry in `x0` is exactly 0 (see
   `../interventions/_ours_combinations.py`'s `encode`), so scipy's own
   simplex had ~2000-4000x too small a starting step in exactly those 56
   directions.
3. Fixed with a custom initial simplex (`_initial_simplex` in
   `_ours_combinations.py`) that gives zero-valued coordinates the same
   absolute step size scipy's own nonzero branch gives the diagonal
   (~0.06). Refitting OU+Cons/knockout with the fix: `edge_corr` went from
   -0.02 to +0.36, a real, substantial jump -- confirming the bug was
   costing real accuracy.
4. But it's not the whole story. 4x the evaluation budget (1500 -> 6000
   maxiter, 1855 -> 23444 evals, 110s -> 1381s) barely moved `edge_corr`
   further (0.36 -> 0.37) or `offdiag_err` (1.595 -> 1.570, vs. a true norm
   of 1.645 -- only ~5% recovered). That's a plateau, not a starved budget:
   plain Nelder-Mead, even correctly initialized, isn't covering a
   65-dimensional non-convex landscape well.

## Why CMA-ES specifically

CMA-ES (Covariance Matrix Adaptation Evolution Strategy, via the `cma`
package -- added to `pyproject.toml` here) is the standard answer to
exactly this shape of problem: moderate dimensionality (tens to low
hundreds), non-convex, no gradients available, single point estimate
wanted. Two properties make it a better-targeted next step than just
raising Nelder-Mead's budget further:

- **Can't inherit the same bug.** Its sampling step is a single explicit
  scalar (`sigma0`) applied isotropically at the start, independent of
  `x0`'s own per-coordinate values -- there's no "coordinate starts at
  exactly 0 so its step is frozen" failure mode to begin with.
- **The adapted covariance *is* the mechanism Nelder-Mead's simplex only
  crudely approximates.** Nelder-Mead's simplex has `N+1` vertices tracking
  local geometry indirectly through reflection/expansion/contraction/
  shrink; CMA-ES directly estimates and adapts a full covariance matrix
  over the search distribution across generations, which is the better-
  understood tool for this exact "moderate-D, rugged, no gradient" regime.

## Setup

Same dataset/config as the other two folders (8 genes, gene_3 knocked out,
network_density=0.5, 1500 fit cells + 500 independent eval cells, seed 42)
but **knockout scenario only** -- the paired knockout/control design in the
other two folders exists to isolate the mu_k question, which isn't what's
being tested here. `maxfevals=2200` for CMA-ES matches Nelder-Mead's typical
evaluation count at `maxiter=1500` (1978-2248 across the 6 combinations in
`../jko-testing/jko_comparison.csv`), so the comparison is budget-matched.

## Files

- `compare_cma_es.py` -- fits all 6 combinations with both
  `fit_combination` (Nelder-Mead, simplex fix already applied) and
  `fit_combination_cma` (new, in `../interventions/_ours_combinations.py`),
  scores both against the held-out eval set, and plots recovered A
  side-by-side (TRUE / Nelder-Mead / CMA-ES) per combination.

Usage:

```bash
uv run python tests/diagnostics/cma-es-test/compare_cma_es.py
```
