# Exact OU transition vs Euler-Maruyama

Does fixing the discretization bug found in `../loss-identifiability/`
actually recover the network better when you re-fit under it -- not just
score better on a loss-value check at two fixed points?

## Where this came from

`../loss-identifiability/check_loss_at_truth.py`'s `n_substeps` sweep found
that the true-vs-fitted loss gap flips sign once the OU forward model
(`ou_gene_expression`/`fp_cell_population`) is simulated with even 5
Euler-Maruyama substeps instead of the default 1, and stays flipped through
100 substeps. A quick correctness check (see `exact_ou.py`'s docstring)
showed the single-step default isn't just mean-biased -- on a
representative test matrix it also inflated the transition covariance by
~5x relative to a 2000-substep reference.

Since the OU SDE is linear, there's no need to approximate its transition
at any resolution: it's exactly Gaussian, computable in closed form via
Van Loan's (1978) block-matrix trick (one matrix exponential per interval,
shared across every particle in that interval -- not one per particle, so
this isn't meaningfully more expensive than a handful of Euler-Maruyama
substeps).

## What this folder adds

- `exact_ou.py` -- `exact_ou_transition` (drop-in replacement for
  `ou_gene_expression`) and `exact_fp_cell_population` (mirrors
  `fp_cell_population.py`, propagating particles with the exact
  transition instead).
- `_exact_combinations.py` -- reimplements `loss_ou`/`loss_fp`/`loss_cons`
  (compute_loss.py) with the exact transition in place of
  `ou_gene_expression`, reusing `../interventions/_ours_combinations.py`'s
  `encode`/`decode`/`COMBINATIONS`/simplex fix unchanged. Same NaN-guard
  pattern as `_ours_combinations.py`'s `_make_objective`: an unstable
  candidate A can make `expm` overflow, so a non-finite transition is
  treated as a rejected candidate, not a crash.
- `compare_exact_ou.py` -- fits all 6 loss combinations both ways
  (Euler-Maruyama n_substeps=1 vs the exact transition), same optimizer
  (Nelder-Mead with the simplex fix), same budget (maxiter=1500), same
  dataset as `../cma-es-test/` (knockout scenario only). Scored on
  edge_corr/AUPRC/AUROC/precision@k against the true A, and on one-step-
  ahead sliced-W2 (using the existing Euler-Maruyama rollout for scoring
  in both cases, so the prediction number stays comparable to every other
  report in this branch).

## This is still one dataset, one seed

Same caveat as everywhere else in this branch: single seed, single
config (8 genes, density 0.5). A real improvement here is encouraging,
not a proof it generalizes.

Usage:

```bash
uv run python tests/diagnostics/exact-ou-test/compare_exact_ou.py
```
