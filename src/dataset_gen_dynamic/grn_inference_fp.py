"""
GRN inference via particle-based Fokker-Planck + Sinkhorn-Wasserstein loss.

Forward model : dX = A(mu - X) dt + diag(D)^(1/2) dW   [Euler-Maruyama, N particles]
Observed data : cross-sectional snapshots at T timepoints (no trajectory required)
Loss          : sum_t  SinkhornCost( simulated_particles(t),  observed_cells(t) )
Optimization  : Adam via optax, gradients from JAX autodiff through the SDE scan

Parameters inferred
-------------------
  A      (G, G)  — GRN interaction / regulation matrix
  mu     (G,)    — constant drift target (effective mean expression)
  log_D  (G,)    — log diagonal diffusion coefficients

Usage
-----
    python grn_inference_fp.py
    python grn_inference_fp.py --data_dir output-distribution-measurements --n_iter 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

DT            = 0.005     # Euler-Maruyama step
N_PARTICLES   = 500       # simulated particles per forward pass
N_SUBSAMPLE   = 500       # observed cells subsampled per timepoint for OT cost
SINKHORN_EPS  = 0.05      # OT regularisation; smaller → truer W2 but slower
N_ITER        = 600
LR            = 1e-3
L2_REG        = 1e-4      # small L2 penalty on A off-diagonal entries
SEED          = 0

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_observations(data_dir: Path) -> tuple[dict[float, np.ndarray], int]:
    """
    Read expression_by_timepoint/*.csv and return:
        obs   — sorted {time: (N_cells, G) float32 array}
        G     — number of gene columns
    """
    tp_dir = data_dir / "expression_by_timepoint"
    result: dict[float, np.ndarray] = {}

    for csv_path in sorted(tp_dir.glob("expression_t*.csv")):
        df       = pd.read_csv(csv_path)
        gene_cols = [c for c in df.columns if c.startswith("gene_")]
        if not gene_cols:
            continue
        # filename encodes time as e.g. "expression_t1_5000" → 1.5
        t_str = csv_path.stem.replace("expression_t", "").replace("_", ".", 1)
        result[float(t_str)] = df[gene_cols].values.astype(np.float32)

    result = dict(sorted(result.items()))
    G      = next(iter(result.values())).shape[1]
    return result, G


def load_true_grns(data_dir: Path) -> dict[str, np.ndarray]:
    grns: dict[str, np.ndarray] = {}
    for path in sorted(data_dir.glob("grn_weighted_pop_*.csv")):
        label      = path.stem.replace("grn_weighted_", "")
        grns[label] = pd.read_csv(path, index_col=0).values.astype(np.float64)
    return grns


def _subsample(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), size=n, replace=False)]


# --------------------------------------------------------------------------
# JAX forward simulation (differentiable w.r.t. A, mu, log_D)
# --------------------------------------------------------------------------

def make_simulate_fn(obs_times: list[float], n_particles: int, dt: float):
    """
    Build and JIT-compile a function:

        simulate(A, mu, log_D, x0, key) -> particles  (n_obs-1, N, G)

    particles[i] is the particle cloud at obs_times[i+1], evolved via
    Euler-Maruyama from x0 placed at obs_times[0].

    Uses jax.lax.scan for memory-efficient backpropagation (O(1) activations).
    """
    # Total steps and per-timepoint record indices (0-based into scan output)
    steps_per_interval = [
        max(1, round((obs_times[i + 1] - obs_times[i]) / dt))
        for i in range(len(obs_times) - 1)
    ]
    n_total    = sum(steps_per_interval)
    record_at  = jnp.array(np.cumsum(steps_per_interval) - 1)   # (n_obs-1,)

    def simulate(A, mu, log_D, x0, key):
        D_sqrt  = jnp.exp(0.5 * log_D)               # (G,)
        sqrt_dt = jnp.sqrt(dt)
        N, G    = x0.shape
        noise   = jax.random.normal(key, (n_total, N, G))

        def euler_step(x, noise_t):
            # drift: (N, G)  each row = A @ (mu - x[n])
            drift = (mu - x) @ A.T
            x_new = x + drift * dt + noise_t * D_sqrt * sqrt_dt
            x_new = jnp.maximum(x_new, 0.05)         # keep positive
            return x_new, x_new

        # scan outputs all_states: (n_total, N, G)
        _, all_states = jax.lax.scan(euler_step, x0, noise)
        return all_states[record_at]                  # (n_obs-1, N, G)

    return jax.jit(simulate), n_total


# --------------------------------------------------------------------------
# Differentiable Sinkhorn cost via OTT-JAX
# --------------------------------------------------------------------------

def _sinkhorn_cost(x: jnp.ndarray, y: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Regularised OT cost between point clouds x (N,G) and y (M,G)."""
    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problem.LinearProblem(geom)
    out  = sinkhorn.Sinkhorn()(prob)
    return out.reg_ot_cost


# --------------------------------------------------------------------------
# Combined loss
# --------------------------------------------------------------------------

def build_loss_fn(
    simulate_fn,
    obs_jnp: list[jnp.ndarray],
    sinkhorn_eps: float,
    l2_reg: float,
):
    """
    Returns:  loss(A, mu, log_D, x0, key) -> scalar

    Loss = Σ_t SinkhornCost(simulated_t, observed_t)  +  l2_reg * ||A_off||²
    """
    def loss(A, mu, log_D, x0, key):
        particles = simulate_fn(A, mu, log_D, x0, key)  # (n_obs-1, N, G)

        ot_cost = jnp.zeros(())
        for t_idx, obs_t in enumerate(obs_jnp):
            ot_cost = ot_cost + _sinkhorn_cost(particles[t_idx], obs_t, sinkhorn_eps)

        # Light L2 regularisation on off-diagonal entries only
        G       = A.shape[0]
        mask    = 1.0 - jnp.eye(G)
        reg     = l2_reg * jnp.sum((A * mask) ** 2)

        return ot_cost + reg

    return loss


# --------------------------------------------------------------------------
# Optimisation loop
# --------------------------------------------------------------------------

def run_inference(
    obs: dict[float, np.ndarray],
    G: int,
    n_particles: int = N_PARTICLES,
    n_subsample:  int = N_SUBSAMPLE,
    dt:           float = DT,
    sinkhorn_eps: float = SINKHORN_EPS,
    n_iter:       int   = N_ITER,
    lr:           float = LR,
    l2_reg:       float = L2_REG,
    seed:         int   = SEED,
) -> dict:
    rng        = np.random.default_rng(seed)
    obs_times  = sorted(obs.keys())

    # Fixed subsampled observations (converted to JAX arrays once)
    obs_sub  = [jnp.array(_subsample(obs[t], n_subsample, rng)) for t in obs_times[1:]]

    # Fixed initial particle cloud sampled from the first observed distribution
    x0 = jnp.array(_subsample(obs[obs_times[0]], n_particles, rng))

    simulate_fn, n_total_steps = make_simulate_fn(obs_times, n_particles, dt)
    loss_fn = build_loss_fn(simulate_fn, obs_sub, sinkhorn_eps, l2_reg)
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))

    # ---------- initialise parameters ----------
    key = jax.random.PRNGKey(seed)
    # A: near-identity diagonal + tiny random off-diagonal
    A_init     = jnp.eye(G) * 1.2 + 0.02 * jax.random.normal(key, (G, G))
    mu_init    = jnp.array(obs[obs_times[0]].mean(axis=0))   # empirical mean at t0
    log_D_init = jnp.full((G,), float(np.log(0.15)))

    optimizer  = optax.adam(lr)
    params     = (A_init, mu_init, log_D_init)
    opt_state  = optimizer.init(params)

    loss_history: list[float] = []

    print(
        f"Inference:  {n_iter} iters | {n_particles} particles | "
        f"{len(obs_times)} timepoints | {G} genes | "
        f"{n_total_steps} SDE steps/forward"
    )

    for i in range(n_iter):
        key, subkey = jax.random.split(key)
        A, mu, log_D = params

        loss_val, grads = loss_and_grad(A, mu, log_D, x0, subkey)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_history.append(float(loss_val))
        if i % 100 == 0 or i == n_iter - 1:
            print(f"  iter {i:4d}  loss = {loss_val:.5f}")

    A_final, mu_final, log_D_final = params
    return {
        "A":            np.array(A_final),
        "mu":           np.array(mu_final),
        "D_diag":       np.exp(np.array(log_D_final)),
        "loss_history": loss_history,
        "obs_times":    obs_times,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_loss(result: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(result["loss_history"], lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sinkhorn-Wasserstein loss")
    ax.set_title("Training loss (log scale)")
    ax.set_yscale("log")
    fig.tight_layout()
    path = out_dir / "loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_grn_recovery(result: dict, true_grns: dict[str, np.ndarray] | None, out_dir: Path) -> None:
    A_inf = result["A"]
    G     = A_inf.shape[0]
    gene_labels = [f"gene_{g}" for g in range(G)]

    if true_grns:
        A_true = list(true_grns.values())[0]
        vmin   = min(A_inf.min(), A_true.min())
        vmax   = max(A_inf.max(), A_true.max())
        frob   = np.linalg.norm(A_inf - A_true)
        mae    = np.mean(np.abs(A_inf - A_true))

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        mats   = [A_true,    A_inf,           np.abs(A_inf - A_true)]
        titles = ["True A",  "Inferred A",    "|Error|"]
        cmaps  = ["magma",   "magma",         "viridis"]

        for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
            kw = {"vmin": vmin, "vmax": vmax} if "Error" not in title else {}
            im = ax.imshow(mat, cmap=cmap, **kw)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(G)); ax.set_xticklabels(gene_labels, rotation=45, fontsize=7)
            ax.set_yticks(range(G)); ax.set_yticklabels(gene_labels, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"GRN recovery  —  ||ΔA||_F = {frob:.3f},  MAE = {mae:.3f}",
            fontsize=11,
        )
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(A_inf, cmap="magma")
        ax.set_title("Inferred A", fontsize=10)
        ax.set_xticks(range(G)); ax.set_xticklabels(gene_labels, rotation=45, fontsize=7)
        ax.set_yticks(range(G)); ax.set_yticklabels(gene_labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = out_dir / "grn_recovery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mu_D(result: dict, true_grns: dict[str, np.ndarray] | None, out_dir: Path) -> None:
    G    = len(result["mu"])
    x    = np.arange(G)
    w    = 0.35

    fig, (ax_mu, ax_D) = plt.subplots(1, 2, figsize=(10, 4))

    ax_mu.bar(x, result["mu"], color="#4393c3", label="Inferred μ")
    ax_mu.set_xticks(x); ax_mu.set_xticklabels([f"gene_{g}" for g in x])
    ax_mu.set_ylabel("μ"); ax_mu.set_title("Inferred drift target μ")
    ax_mu.legend()

    ax_D.bar(x, result["D_diag"], color="#4393c3", label="Inferred D")
    ax_D.set_xticks(x); ax_D.set_xticklabels([f"gene_{g}" for g in x])
    ax_D.set_ylabel("D diagonal"); ax_D.set_title("Inferred diffusion D")
    ax_D.legend()

    fig.tight_layout()
    path = out_dir / "mu_D_inferred.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_per_timepoint_w2(
    result: dict,
    obs: dict[float, np.ndarray],
    n_particles: int = N_PARTICLES,
    n_subsample: int = N_SUBSAMPLE,
    sinkhorn_eps: float = SINKHORN_EPS,
    seed: int = SEED,
) -> tuple[list[float], list[float]]:
    """
    Single forward pass with the inferred parameters; returns the Sinkhorn-W2
    cost between the Fokker-Planck particle cloud and the observed empirical
    distribution at each timepoint after t=0.

    Returns
    -------
    eval_times : timepoints at which costs are measured  (t1, t2, …)
    costs      : Sinkhorn-W2 cost at each timepoint
    """
    rng       = np.random.default_rng(seed)
    obs_times = result["obs_times"]

    simulate_fn, _ = make_simulate_fn(obs_times, n_particles, dt=DT)
    x0     = jnp.array(_subsample(obs[obs_times[0]], n_particles, rng))
    A      = jnp.array(result["A"])
    mu     = jnp.array(result["mu"])
    log_D  = jnp.log(jnp.array(result["D_diag"]))

    # Fixed key for reproducible evaluation
    particles = simulate_fn(A, mu, log_D, x0, jax.random.PRNGKey(seed))  # (T-1, N, G)

    eval_times, costs = [], []
    for t_idx, t in enumerate(obs_times[1:]):
        obs_t = jnp.array(_subsample(obs[t], n_subsample, rng))
        cost  = float(_sinkhorn_cost(particles[t_idx], obs_t, sinkhorn_eps))
        eval_times.append(t)
        costs.append(cost)

    return eval_times, costs


def plot_wasserstein_per_timepoint(
    eval_times: list[float],
    costs: list[float],
    out_dir: Path,
) -> None:
    """
    Bar chart of Sinkhorn-W2 distance between the Fokker-Planck predicted
    distribution and the observed empirical distribution at each timepoint.
    Lower bars mean the model distribution matches observations more closely.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(eval_times) * 1.1), 4))

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(eval_times)))
    bars   = ax.bar(
        [str(t) for t in eval_times],
        costs,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    # Annotate each bar with its value
    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.01,
            f"{cost:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Collection timepoint")
    ax.set_ylabel("Sinkhorn-W₂ cost")
    ax.set_title(
        "Wasserstein distance: Fokker-Planck (simulated) vs observed\n"
        "at each timepoint after t=0"
    )
    ax.set_ylim(0, max(costs) * 1.15)
    fig.tight_layout()

    path = out_dir / "w2_per_timepoint.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary(result: dict, true_grns: dict[str, np.ndarray] | None) -> None:
    print("\n--- Inference summary ---")
    print(f"  Inferred μ:     {result['mu'].round(3)}")
    print(f"  Inferred D_diag:{result['D_diag'].round(4)}")
    if true_grns:
        A_true = list(true_grns.values())[0]
        A_inf  = result["A"]
        frob   = np.linalg.norm(A_inf - A_true)
        mae    = np.mean(np.abs(A_inf - A_true))
        print(f"  ||A_inf − A_true||_F = {frob:.4f}")
        print(f"  MAE(A)               = {mae:.4f}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="output-distribution-measurements")
    parser.add_argument("--n_iter",       type=int,   default=N_ITER)
    parser.add_argument("--n_particles",  type=int,   default=N_PARTICLES)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--eps",          type=float, default=SINKHORN_EPS)
    parser.add_argument("--l2_reg",       type=float, default=L2_REG)
    parser.add_argument("--seed",         type=int,   default=SEED)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path("output-fp-inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, G    = load_observations(data_dir)
    true_grns = load_true_grns(data_dir) if (data_dir / "grn_weighted_pop_0.csv").exists() else None

    print(f"Loaded {len(obs)} timepoints × ~{next(iter(obs.values())).shape[0]} cells, {G} genes")
    if true_grns:
        print(f"Ground-truth GRNs: {list(true_grns.keys())}")

    result = run_inference(
        obs, G,
        n_particles  = args.n_particles,
        dt           = DT,
        sinkhorn_eps = args.eps,
        n_iter       = args.n_iter,
        lr           = args.lr,
        l2_reg       = args.l2_reg,
        seed         = args.seed,
    )

    # Save artefacts
    gene_labels = [f"gene_{g}" for g in range(G)]
    pd.DataFrame(result["A"], index=gene_labels, columns=gene_labels).to_csv(
        out_dir / "A_inferred.csv"
    )
    pd.DataFrame({"loss": result["loss_history"]}).to_csv(
        out_dir / "loss.csv", index=False
    )

    plot_loss(result, out_dir)
    plot_grn_recovery(result, true_grns, out_dir)
    plot_mu_D(result, true_grns, out_dir)

    print("Computing per-timepoint Wasserstein distances ...")
    eval_times, tp_costs = compute_per_timepoint_w2(
        result, obs,
        n_particles  = args.n_particles,
        sinkhorn_eps = args.eps,
        seed         = args.seed,
    )
    pd.DataFrame({"time": eval_times, "w2_cost": tp_costs}).to_csv(
        out_dir / "w2_per_timepoint.csv", index=False
    )
    plot_wasserstein_per_timepoint(eval_times, tp_costs, out_dir)

    print_summary(result, true_grns)
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
