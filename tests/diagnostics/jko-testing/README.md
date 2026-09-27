# JKOnet* comparison

Compares this repo's Algorithm-1 implementation (OU / FP / Cons losses, known
intervention means) against [JKOnet*](https://github.com/antonioterpin/jkonet-star)
(Terpin et al., NeurIPS 2024) on the same single-gene-knockout dataset used by
`../interventions/single-gene-knockout/`.

## Setup (one-time per clone)

```bash
bash tests/diagnostics/jko-testing/setup_jkonet_star.sh
```

Vendors JKOnet* into `external/jkonet-star/` with its own `.venv`, pinned to
commit `1741c53a` (2025-03-18). `external/` is gitignored — JKOnet* pins
`jax==0.4.26`, `torch==2.2.2`, `wandb==0.17.0`, none of which belong in this
project's `pyproject.toml`, and it expects to be run from its own repo root
(relative `data/`/`out/`/`config.yaml` paths, absolute `from models import
...` imports) — so it's driven as a subprocess in a second interpreter, not
imported as a library.

## Why this isn't a like-for-like comparison

JKOnet* has no slot for "known intervention means {mu_k}" — Algorithm 1 is
handed mu_k as input (paper line 2); JKOnet* learns a potential purely from
consecutive snapshots. So this measures "how much does knowing mu_k help",
not just "which method fits better". `compare_jkonet_star.py` runs both the
knockout dataset and the constant-mu control (`make_constant_sim`) to
separate that from raw fitting quality.

"Ours" is all 6 loss combinations from `../interventions/_ours_combinations.py`
(OU, FP, OU+FP, OU+Cons, FP+Cons, OU+FP+Cons — same set as
`../interventions/single-gene-knockout/loss-combinations/compare_losses_single_gene_knockout.py`'s
sweep, Cons-alone excluded there as degenerate), not a single picked
combination — see `_ours_combinations.py`'s docstring for why picking one
"best" combination from a single past sweep would have been leaning on
noise rather than a real ranking.

The knockout's mu(t) is a regime switch at `t_star`, which a *static*
potential can't represent. `jkonet-star-time-potential` (potential takes
time as an extra feature) is the structural match; the plain
`jkonet-star-potential-internal` is included as the "doesn't even try to
handle nonstationarity" baseline.

No `interaction` term is used on either side of the comparison — our model
has no particle-particle OT interaction, only shared drift through A, so
`jkonet-star`/`jkonet-star-linear` (full, with interaction) aren't a
structural match.

## Feature-config gotcha for the linear/closed-form solvers

`jkonet-star-linear-*` builds its feature basis as a **full per-dimension
power grid**, for both pieces of `config.yaml`'s feature config
(`models/jkonet_star.py`, `JKOnetStarLinear.__init__`):

- RBF centers: `itertools.product(linspace(domain, n_centers_per_dim), repeat=data_dim)`
  — the stock `n_centers_per_dim=10` at 5-8 genes means 10^5-10^8 centers. Hangs.
- Polynomials: `itertools.product(range(degree + 1), repeat=data_dim)` (only
  the all-zero exponent tuple is dropped) — **not** a total-degree cutoff
  like `sklearn.PolynomialFeatures`. `degree` caps each variable's own
  exponent, but every combination across dimensions is still included, so
  the feature count is `(degree+1)**data_dim - 1` regardless of total
  degree. `degree=2` is only 8 features at 2 genes but 3**8-1=6560 at 8 —
  the per-batch feature-outer-product (`train_step`'s
  `jnp.einsum('ijk,ijh->ikh', yt, yt)`, a `(features_dim, features_dim)`
  matrix per sample) OOMs and gets SIGKILLed at that size.

`_jko_star_driver.py` overrides this per run: RBFs disabled entirely
(`types: []`, `n_centers_per_dim: 1`), polynomial `degree: 2`. Degree 2 is
*sufficient* for our quadratic OU potential (Psi(c) = (1/2)(c-mu)^T A
(c-mu)) at any gene count -- the fitted quadratic coefficients read back out
as an A_hat comparable to the OU fit's -- but only *tractable* below roughly
5-6 genes, since (degree+1)**G is what actually governs cost, not degree
alone. `compare_jkonet_star.py` therefore excludes
`jkonet-star-linear-potential-internal` from `JKO_SOLVERS` at N_GENES=8; add
it back for a smaller-gene-count run.

## Shared metric

Both methods' one-step-ahead predicted clouds (true snapshot at t_k rolled
one interval forward) are compared to the true snapshot at t_{k+1} using
*this* repo's `_sliced_w2` (`multsc_grn_inference.compute_loss`), computed
entirely on our side — `_jko_star_driver.py` only dumps raw predicted
particle arrays (`.npy`), not a JAX-side error number, precisely so both
methods are scored with the same estimator instead of two different
Wasserstein approximations.

## Files

- `setup_jkonet_star.sh` — one-time vendoring (see above).
- `_jko_star_driver.py` — runs *inside* `external/jkonet-star/.venv`
  (invoked with `cwd=external/jkonet-star`). Trains the requested solver(s)
  on a pre-built dataset dir, rolls out one-step-ahead predictions with the
  fitted potential, and dumps predicted clouds + fitted params to
  `out/jko-testing/<run>/`.
- `compare_jkonet_star.py` — the report script (run in *this* project's
  venv). Builds the knockout + constant-mu datasets, converts snapshots to
  JKOnet*'s `data.npy`/`sample_labels.npy` format, shells out to
  `data_generator.py` (their coupling/density preprocessing) and
  `_jko_star_driver.py`, fits all 6 of our loss combinations on the
  identical snapshots (`../interventions/_ours_combinations.py`), and
  produces `jko_comparison.{csv,png}`.

Usage:

```bash
uv run python tests/diagnostics/jko-testing/compare_jkonet_star.py
```
