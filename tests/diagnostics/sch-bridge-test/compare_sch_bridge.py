"""
Same question as ../jko-testing/compare_jkonet_star.py: how much does
knowing the intervention means {mu_k} buy Algorithm 1, on the same
single-gene-knockout dataset -- except the "doesn't know mu_k" baseline
here is minibatch-OT conditional flow matching (ott-jax) instead of
JKOnet*. See README.md for why ott-jax was picked over JKOnet*/CytoBridge,
and for the honest caveat that this is deterministic OT-CFM (approximates
the Benamou-Brenier OT map between consecutive marginals), not a full
stochastic Schrödinger bridge.

Dataset: identical scenario construction to ../jko-testing/ (same config
constants, same paired knockout/effectively-constant-mu control, same
independent held-out eval ensemble) so the two folders' CSVs are directly
comparable.

Methods per scenario:
  - ours: all 6 loss combinations from ../interventions/_ours_combinations.py
    (same module ../jko-testing/compare_jkonet_star.py uses -- see that
    module's docstring for why a single picked "best" combination isn't
    used here instead).
  - flow-matching, informed: cond=mu_k fed to the velocity field --
    the flow-matching analogue of "knows the intervention target".
  - flow-matching, uninformed: cond=None -- has to infer everything from
    the OT-coupled snapshots alone, structurally like JKOnet*.
  Because conditioning is a toggle in one codebase rather than a different
  library, informed-vs-uninformed here isolates the effect of knowing mu_k
  more cleanly than the JKOnet* comparison did.

Metric: one-step-ahead sliced-W2 (this repo's _sliced_w2), scored against
the same independent eval set for every method.

Not a test -- a standalone report script; re-run anytime, overwrites this
folder's sch_bridge_comparison.{csv,png}. No setup step (pure in-process
JAX): `uv run python tests/diagnostics/sch-bridge-test/compare_sch_bridge.py`
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import nnx
from ott.geometry import pointcloud
from ott.neural.methods.flow_matching import (
    evaluate_velocity_field,
    flow_matching_step,
    interpolate_samples,
)
from ott.neural.networks.velocity_field.mlp import MLP
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.compute_loss import _sliced_w2
from multsc_grn_inference.housekeeping.edge_recovery_metrics import (
    auprc,
    auroc,
    precision_at_k,
)

from _intervention_ground_truth import (  # noqa: E402
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)
from _ours_combinations import (  # noqa: E402
    COMBINATIONS,
    fit_combination,
    off_diag_indices,
    one_step_ahead_w2,
)

# ---------------------------------------------------------------------------
# Config -- matches ../jko-testing/compare_jkonet_star.py so the two
# folders' results are directly comparable.
# ---------------------------------------------------------------------------

N_GENES = 8
KO_GENE = 3
N_CELLS = 1500
N_EVAL_CELLS = 500
N_SNAPS = 10
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.5
SEED = 42

N_PROJ = 30
MAXITER = 1500
OPTIMIZER_METHOD = "Nelder-Mead"

# Flow matching
HIDDEN_DIMS = (64, 64)          # matches JKOnet*'s config.yaml model.layers
FM_LR = 1e-3                     # matches JKOnet*'s config.yaml energy.optim.lr
FM_TRAIN_STEPS = 2000
FM_BATCH_SIZE = 256
FM_ROLLOUT_STEPS = 20            # fixed Euler steps for evaluate_velocity_field

HERE = Path(__file__).parent

OFF = off_diag_indices(N_GENES)


def _edge_metrics(true_off: np.ndarray, hat_off: np.ndarray) -> dict:
    """edge_corr plus AUPRC/AUROC/precision@k -- see
    src/multsc_grn_inference/housekeeping/edge_recovery_metrics.py for why
    the latter three are the more trustworthy read on edge recovery."""
    k = int((true_off != 0).sum())
    return {
        "edge_corr": float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0,
        "auprc": auprc(true_off, hat_off),
        "auroc": auroc(true_off, hat_off),
        "precision_at_k": precision_at_k(true_off, hat_off, k),
    }


# ---------------------------------------------------------------------------
# Scenario construction (identical to ../jko-testing/)
# ---------------------------------------------------------------------------

def build_scenario(t_star: float | None):
    """t_star=None -> pushed past T (effectively-constant-mu control).
    See ../jko-testing/compare_jkonet_star.py:build_scenario for the full
    rationale (paired scenarios, independent eval ensemble)."""
    sim = make_single_gene_knockout_sim(
        num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
        t_star=T_STAR if t_star is None else t_star, network_density=NETWORK_DENSITY,
    )
    if t_star is None:
        sim.mu_kwargs["t_star"] = T + 100.0
    snapshots, times, mu_known, dt = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)
    eval_snapshots, _, _, _ = extract_snapshots(sim, N_EVAL_CELLS, N_SNAPS, T=T)
    return sim, snapshots, eval_snapshots, mu_known, dt


# ---------------------------------------------------------------------------
# Flow matching: OT-coupled pairs -> conditional flow matching -> rollout.
# See README.md for what this is and isn't (deterministic OT-CFM, not full
# stochastic SB).
# ---------------------------------------------------------------------------

def compute_couplings(snapshots: list[np.ndarray]) -> list[np.ndarray]:
    """Sinkhorn transport plan (n, m) between each consecutive snapshot
    pair -- the OT-coupling ingredient that turns plain flow matching into
    minibatch-OT flow matching."""
    plans = []
    for k in range(len(snapshots) - 1):
        geom = pointcloud.PointCloud(jnp.asarray(snapshots[k]), jnp.asarray(snapshots[k + 1]))
        out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
        plans.append(np.asarray(out.matrix))
    return plans


def make_pair_sampler(P: np.ndarray):
    """Precomputed-cumsum sampler: draw (i, j) index pairs proportional to
    P's entries in O(log n) per draw instead of rebuilding the cumsum
    (O(n*m)) on every call."""
    flat_cumsum = np.cumsum(P.ravel())
    flat_cumsum /= flat_cumsum[-1]
    n_cols = P.shape[1]

    def sample(rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        flat_idx = np.searchsorted(flat_cumsum, rng.random(batch_size))
        return np.divmod(flat_idx, n_cols)

    return sample


def train_flow_matching(
    snapshots: list[np.ndarray], mu_known, *, informed: bool, seed: int = 0,
) -> nnx.Module:
    plans = compute_couplings(snapshots)
    samplers = [make_pair_sampler(P) for P in plans]
    n_intervals = len(samplers)

    rngs = nnx.Rngs(seed)
    model = MLP(dim=N_GENES, hidden_dims=HIDDEN_DIMS, cond_dim=N_GENES if informed else 0, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(FM_LR), wrt=nnx.Param)

    np_rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    for _step in range(FM_TRAIN_STEPS):
        k = int(np_rng.integers(n_intervals))
        i_idx, j_idx = samplers[k](np_rng, FM_BATCH_SIZE)
        x0 = jnp.asarray(snapshots[k][i_idx])
        x1 = jnp.asarray(snapshots[k + 1][j_idx])
        cond = jnp.tile(jnp.asarray(mu_known[k]), (FM_BATCH_SIZE, 1)) if informed else None

        key, sub = jax.random.split(key)
        batch = interpolate_samples(sub, x0, x1, cond)
        flow_matching_step(model, optimizer, batch)

    return model


def one_step_ahead_w2_fm(model: nnx.Module, eval_snapshots, mu_known, *, informed: bool) -> float:
    rng = np.random.default_rng(0)
    errs = []
    for k in range(len(eval_snapshots) - 1):
        cond_vec = jnp.asarray(mu_known[k]) if informed else None

        def rollout_one(x, cond_vec=cond_vec):
            sol = evaluate_velocity_field(model, x, cond_vec, t0=0.0, t1=1.0, num_steps=FM_ROLLOUT_STEPS)
            return sol.ys[0]

        pred = np.asarray(jax.vmap(rollout_one)(jnp.asarray(eval_snapshots[k])))
        errs.append(_sliced_w2(pred, eval_snapshots[k + 1], N_PROJ, rng))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []
    A_hats: dict[str, np.ndarray] = {}
    A_trues: dict[str, np.ndarray] = {}

    for scenario_name, t_star in [("knockout", T_STAR), ("control", None)]:
        print(f"\n=== scenario: {scenario_name} ===", flush=True)
        sim, snapshots, eval_snapshots, mu_known, dt = build_scenario(t_star)
        A_true = sim.A
        A_trues[scenario_name] = A_true
        true_off = np.array([A_true[i, j] for i, j in OFF])

        for combo_name, terms in COMBINATIONS.items():
            print(f"[ours: {combo_name}] fitting ...", flush=True)
            theta_hat, fit_info = fit_combination(
                terms, snapshots, mu_known, dt, N_GENES,
                n_proj=N_PROJ, maxiter=MAXITER, method=OPTIMIZER_METHOD, seed=0,
            )
            hat_off = np.array([theta_hat.A[i, j] for i, j in OFF])
            w2_ours = one_step_ahead_w2(theta_hat, eval_snapshots, dt, n_proj=N_PROJ, seed=0)
            rows.append({
                "scenario": scenario_name, "method": f"ours ({combo_name})",
                "one_step_w2": w2_ours,
                "A_err": float(np.linalg.norm(theta_hat.A - A_true)),
                "offdiag_err": float(np.linalg.norm(hat_off - true_off)),
                **_edge_metrics(true_off, hat_off),
                "sigma_err": abs(theta_hat.sigma - effective_sigma(sim)),
                **fit_info,
            })
            A_hats[f"{scenario_name}__ours_{combo_name}"] = theta_hat.A
            print(f"  w2={w2_ours:.4f}  A_err={rows[-1]['A_err']:.3f}  "
                  f"edge_corr={rows[-1]['edge_corr']:+.3f}  auprc={rows[-1]['auprc']:.3f}")

        for informed in (True, False):
            label = "flow-matching (informed, cond=mu_k)" if informed else "flow-matching (uninformed)"
            print(f"[{label}] training ...", flush=True)
            t0 = time.perf_counter()
            model = train_flow_matching(snapshots, mu_known, informed=informed, seed=0)
            elapsed = time.perf_counter() - t0
            w2_fm = one_step_ahead_w2_fm(model, eval_snapshots, mu_known, informed=informed)
            rows.append({
                "scenario": scenario_name, "method": label,
                "one_step_w2": w2_fm, "seconds": round(elapsed, 1),
            })
            print(f"  w2={w2_fm:.4f}  {elapsed:.0f}s")

    df = pd.DataFrame(rows)
    csv_path = HERE / "sch_bridge_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df)
    _plot_matrices(A_trues, A_hats)


def _plot(df: pd.DataFrame) -> None:
    scenarios = df["scenario"].unique()
    n_rows = df.groupby("scenario").size().max()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 0.5 * n_rows + 1.5))
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        colors = ["#1a9641" if m.startswith("ours") else
                  "#d6604d" if "uninformed" in m else "#4393c3" for m in sub["method"]]
        ax.barh(sub["method"], sub["one_step_w2"], color=colors)
        ax.set_title(f"{scenario}: one-step-ahead sliced-W2 (lower better)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Ours -- all 6 loss combinations (knows {{mu_k}}) vs OT-coupled flow matching, "
        f"informed/uninformed -- single-gene knockout, {N_GENES} genes, density={NETWORK_DENSITY}"
    )
    fig.tight_layout()
    png_path = HERE / "sch_bridge_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _plot_matrices(A_trues: dict[str, np.ndarray], A_hats: dict[str, np.ndarray]) -> None:
    """TRUE A vs every recovered A_hat (all 6 "ours" combinations -- flow
    matching has no A_hat, it's a neural velocity field, not a linear form),
    one row per scenario, shared per-scenario color scale."""
    scenarios = list(A_trues.keys())
    panel_labels = ["TRUE A"] + [f"ours ({c})" for c in COMBINATIONS]
    n_cols = len(panel_labels)

    fig, axes = plt.subplots(len(scenarios), n_cols, figsize=(2.0 * n_cols, 2.2 * len(scenarios) + 0.5))
    axes = np.atleast_2d(axes)

    im = None
    for row, scenario in enumerate(scenarios):
        A_true = A_trues[scenario]
        vmax = np.abs(A_true).max() * 1.2
        for col, label in enumerate(panel_labels):
            ax = axes[row, col]
            mat = A_true if label == "TRUE A" else A_hats[f"{scenario}__ours_{label[len('ours ('):-1]}"]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=8, fontweight="bold" if label == "TRUE A" else "normal")
            if col == 0:
                ax.set_ylabel(scenario, fontsize=9)

    fig.suptitle(f"Recovered A -- {N_GENES} genes, density={NETWORK_DENSITY}", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="A_ij", pad=0.01)
    png_path = HERE / "sch_bridge_recovered_matrices.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
