# Schrödinger-bridge-family comparison (ott-jax flow matching)

Same question as `../jko-testing/`: how much does knowing the intervention
means {mu_k} actually buy Algorithm 1, on the single-gene-knockout dataset?
This time the "doesn't know mu_k" baseline is OT-coupled flow matching from
`ott-jax` instead of JKOnet*.

## Why ott-jax instead of JKOnet* or CytoBridge

Considered three packages (see the conversation this folder came out of):

- **JKOnet\*** (`../jko-testing/`) -- already integrated. Needed a whole
  vendored repo + second Python venv (torch/wandb/pinned jax) driven over a
  subprocess boundary, and its potential has no first-class way to take a
  known drift target -- doing that would mean patching their library code.
- **CytoBridge** (NeurIPS 2025 / ICLR 2025, `pip install CytoBridge`) -- the
  most literal "Schrödinger bridge for single-cell population dynamics"
  match, but PyTorch-based (same vendoring problem as JKOnet*) and, per its
  README, no built-in support for a fixed/known drift term either.
- **ott-jax's flow matching** (`ott.neural.methods.flow_matching`) --
  already a project dependency (just needed the `neural` extra: flax,
  diffrax -- added to `pyproject.toml` here). JAX-native, no second venv, no
  subprocess. Its velocity-field network signature is `(t, x_t, cond) ->
  v_t` with `cond` as a first-class, documented argument -- feeding mu_k in
  is exactly what it's for, not a patch.

ott-jax was also the pick because it's independently useful here for
Wasserstein distance computation (`ott.tools.sinkhorn_divergence`,
`ott.solvers.linear`) beyond just this comparison.

## What's actually implemented, honestly

`ott.neural.methods.flow_matching` gives you the *machinery* (batch
construction, one training step, an ODE rollout) for conditional flow
matching -- straight-line interpolation between paired (x0, x1) with target
velocity x1-x0. It does **not** ship the "Schrödinger bridge" part itself;
that comes from how the (x0, x1) pairs are chosen. Here they're drawn from
an entropic (Sinkhorn) OT coupling between consecutive TRUE snapshots
(`ott.solvers.linear.sinkhorn.Sinkhorn`, `ott.geometry.pointcloud.PointCloud`)
rather than i.i.d. minibatch pairing -- this is minibatch-OT conditional
flow matching / rectified flow, the standard practical stand-in for the full
stochastic Schrödinger bridge (Tong et al., "Improving and generalizing
flow-based generative models with minibatch optimal transport", 2023).

What this is **not**: a full entropic Schrödinger bridge with a stochastic
(Brownian-bridge) path measure between endpoints -- the interpolation here
is deterministic (straight line), so it approximates the *Benamou-Brenier
OT map* between consecutive marginals rather than sampling full SB path
noise. Good enough to test the same "does knowing mu_k help" question, but
don't read the numbers here as "the Schrödinger bridge solution" in the
strict sense.

## Two variants, one codebase

Because conditioning is a toggle in the same API (not a different library),
this comparison is tighter than the JKOnet* one:

- **informed**: `cond = mu_k` per interval (the network sees the known
  intervention target, same information Algorithm 1 gets).
- **uninformed**: `cond = None` (has to infer everything from the OT
  couplings alone, like JKOnet*).

Same paired knockout/control scenarios and same config constants
(N_GENES=8, N_CELLS=1500, N_SNAPS=10, ...) as `../jko-testing/`, and the
same held-out-eval-set + `_sliced_w2` one-step-ahead metric, so the two
folders' `*.csv` are directly comparable.

"Ours" is all 6 loss combinations from `../interventions/_ours_combinations.py`
(same module `../jko-testing/compare_jkonet_star.py` uses), not a single
picked combination -- see that module's docstring for why picking one
"best" combination from a single past sweep would have been leaning on
noise rather than a real ranking.

## Files

- `compare_sch_bridge.py` -- the report script. No setup step needed (pure
  in-process JAX, unlike `../jko-testing/`): just
  `uv run python tests/diagnostics/sch-bridge-test/compare_sch_bridge.py`.
