"""
GPU-optimised GRN inference via particle Fokker-Planck + Sinkhorn-Wasserstein.

Key GPU wins over grn_inference_fp.py (CPU version)
─────────────────────────────────────────────────────
1. Sinkhorn costs across all timepoints run in parallel via jax.vmap
   instead of a Python for-loop — avoids serial OT dispatch.
2. Gradient accumulation: K independent noise draws are vmapped per step,
   averaging the stochastic gradient at no extra wall-clock cost on GPU.
3. Larger default particle count (2 000) to saturate GPU SIMD lanes.
4. All JAX arrays pinned to the selected GPU via jax.device_put.
5. Optional bfloat16 for the SDE scan (2× throughput on A100 / H100).

Usage
─────
    python grn_inference_fp_gpu.py
    python grn_inference_fp_gpu.py \\
        --data_dir output-distribution-measurements \\
        --n_particles 4000 --n_grad_accum 8 --dtype bfloat16 --n_iter 600
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
# Defaults  (larger than CPU version to exploit GPU parallelism)
# --------------------------------------------------------------------------

DT            = 0.005
N_PARTICLES   = 2000      # GPU can hold many more particles per forward pass
N_SUBSAMPLE   = 2000      # larger OT sample → better Wasserstein estimate
N_GRAD_ACCUM  = 4         # noise draws per gradient step (averaged)
SINKHORN_EPS  = 0.05
N_ITER        = 600
LR            = 1e-3
L2_REG        = 1e-4
SEED          = 0

# --------------------------------------------------------------------------
# Device helpers
# --------------------------------------------------------------------------

def _get_device(prefer_gpu: bool = True) -> jax.Device:
    """Return a GPU device if available, else fall back to CPU with a warning."""
    gpu_devices = jax.devices("gpu") if prefer_gpu else []
    if gpu_devices:
        dev = gpu_devices[0]
        print(f"Using GPU: {dev}")
        return dev
    cpu = jax.devices("cpu")[0]
    print("WARNING: no GPU found — falling back to CPU. "
          "GPU-specific defaults (N_PARTICLES, N_GRAD_ACCUM) may be slow.")
    return cpu


def _to_device(x: np.ndarray | jnp.ndarray, device: jax.Device) -> jnp.ndarray:
    return jax.device_put(jnp.array(x), device)


# --------------------------------------------------------------------------
# Data loading  (identical logic to CPU version)
# --------------------------------------------------------------------------

def load_observations(data_dir: Path) -> tuple[dict[float, np.ndarray], int]:
    tp_dir = data_dir / "expression_by_timepoint"
    result: dict[float, np.ndarray] = {}
    for csv_path in sorted(tp_dir.glob("expression_t*.csv")):
        df        = pd.read_csv(csv_path)
        gene_cols = [c for c in df.columns if c.startswith("gene_")]
        if not gene_cols:
            continue
        t_str = csv_path.stem.replace("expression_t", "").replace("_", ".", 1)
        result[float(t_str)] = df[gene_cols].values.astype(np.float32)
    result = dict(sorted(result.items()))
    return result, next(iter(result.values())).shape[1]


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
# SDE forward simulation (raw, non-jitted — JIT happens at the loss level)
# --------------------------------------------------------------------------

def _make_simulate_raw(obs_times: list[float], n_particles: int, dt: float, dtype):
    """
    Returns a function:
        simulate_raw(A, mu, log_D, x0, key) -> particles  (n_obs-1, N, G)

    Not JIT-compiled here so it can be vmapped cleanly inside the loss closure.
    The outer loss function is JIT-compiled instead.
    """
    steps_per_interval = [
        max(1, round((obs_times[i + 1] - obs_times[i]) / dt))
        for i in range(len(obs_times) - 1)
    ]
    n_total   = sum(steps_per_interval)
    record_at = jnp.array(np.cumsum(steps_per_interval) - 1)  # (n_obs-1,)

    def simulate_raw(A, mu, log_D, x0, key):
        D_sqrt  = jnp.exp(0.5 * log_D).astype(dtype)
        sqrt_dt = jnp.sqrt(jnp.array(dt, dtype=dtype))
        x0_     = x0.astype(dtype)
        A_      = A.astype(dtype)
        mu_     = mu.astype(dtype)
        N, G    = x0_.shape

        noise = jax.random.normal(key, (n_total, N, G), dtype=dtype)

        def euler_step(x, noise_t):
            drift = (mu_ - x) @ A_.T
            x_new = x + drift * dt + noise_t * D_sqrt * sqrt_dt
            x_new = jnp.maximum(x_new, jnp.array(0.05, dtype=dtype))
            return x_new, x_new

        _, all_states = jax.lax.scan(euler_step, x0_, noise)
        return all_states[record_at].astype(jnp.float32)   # (n_obs-1, N, G)

    return simulate_raw, n_total


# --------------------------------------------------------------------------
# Differentiable Sinkhorn cost  (OTT-JAX, vmappable)
# --------------------------------------------------------------------------

def _sinkhorn_cost(x: jnp.ndarray, y: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Regularised OT cost between point clouds x (N,G) and y (M,G)."""
    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problem.LinearProblem(geom)
    out  = sinkhorn.Sinkhorn()(prob)
    return out.reg_ot_cost


# --------------------------------------------------------------------------
# GPU loss function
# --------------------------------------------------------------------------

def build_loss_fn(
    simulate_raw,
    obs_stack: jnp.ndarray,   # (T, N_sub, G)  — all timepoints stacked
    sinkhorn_eps: float,
    l2_reg: float,
    n_grad_accum: int,
):
    """
    Returns a JIT-compiled loss function:
        loss(A, mu, log_D, x0, key) -> scalar

    GPU optimisations
    -----------------
    • jax.vmap over T timepoints for the Sinkhorn costs — all OT problems
      solved in parallel on the GPU rather than serially.
    • jax.vmap over n_grad_accum noise draws — K forward passes run in
      parallel; their costs are averaged, giving a lower-variance gradient
      estimate with no additional wall-clock overhead.
    """
    T = obs_stack.shape[0]

    def _loss_single_key(A, mu, log_D, x0, key):
        """One forward pass: simulate then vmap Sinkhorn over T timepoints."""
        particles = simulate_raw(A, mu, log_D, x0, key)  # (T, N, G)
        costs     = jax.vmap(
            lambda p, o: _sinkhorn_cost(p, o, sinkhorn_eps)
        )(particles, obs_stack)                           # (T,)
        return costs.sum()

    def loss(A, mu, log_D, x0, key):
        # Split into K subkeys and vmap → (K,) costs, then average
        subkeys  = jax.random.split(key, n_grad_accum)   # (K, 2)
        ot_costs = jax.vmap(
            lambda sk: _loss_single_key(A, mu, log_D, x0, sk)
        )(subkeys)                                         # (K,)
        ot_cost  = ot_costs.mean()

        G    = A.shape[0]
        mask = 1.0 - jnp.eye(G)
        reg  = l2_reg * jnp.sum((A * mask) ** 2)
        return ot_cost + reg

    return jax.jit(loss)


# --------------------------------------------------------------------------
# Optimisation loop
# --------------------------------------------------------------------------

def run_inference(
    obs: dict[float, np.ndarray],
    G: int,
    device: jax.Device,
    n_particles:  int   = N_PARTICLES,
    n_subsample:  int   = N_SUBSAMPLE,
    n_grad_accum: int   = N_GRAD_ACCUM,
    dt:           float = DT,
    sinkhorn_eps: float = SINKHORN_EPS,
    n_iter:       int   = N_ITER,
    lr:           float = LR,
    l2_reg:       float = L2_REG,
    dtype_str:    str   = "float32",
    seed:         int   = SEED,
) -> dict:
    rng       = np.random.default_rng(seed)
    obs_times = sorted(obs.keys())
    dtype     = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32

    # Subsample observations and move to device
    obs_sub_list = [
        _to_device(_subsample(obs[t], n_subsample, rng), device)
        for t in obs_times[1:]
    ]
    # Stack into (T, N_sub, G) for vmapped Sinkhorn
    obs_stack = jnp.stack(obs_sub_list, axis=0)           # (T, N_sub, G)

    # Fixed initial particles from the first observed snapshot
    x0 = _to_device(_subsample(obs[obs_times[0]], n_particles, rng), device)

    simulate_raw, n_total_steps = _make_simulate_raw(obs_times, n_particles, dt, dtype)
    loss_fn       = build_loss_fn(simulate_raw, obs_stack, sinkhorn_eps, l2_reg, n_grad_accum)
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))

    # ---------- initialise parameters on device ----------
    key    = jax.random.PRNGKey(seed)
    A_init = _to_device(
        np.eye(G) * 1.2 + 0.02 * np.random.default_rng(seed).standard_normal((G, G)),
        device,
    )
    mu_init    = _to_device(obs[obs_times[0]].mean(axis=0), device)
    log_D_init = _to_device(np.full(G, np.log(0.15)), device)

    optimizer = optax.adam(lr)
    params    = (A_init, mu_init, log_D_init)
    opt_state = optimizer.init(params)

    loss_history: list[float] = []

    print(
        f"GPU inference:  {n_iter} iters | {n_particles} particles | "
        f"{n_grad_accum} grad-accum draws | {len(obs_times)} timepoints | "
        f"{G} genes | {n_total_steps} SDE steps/forward | dtype={dtype_str}"
    )

    for i in range(n_iter):
        key, subkey = jax.random.split(key)
        subkey = jax.device_put(subkey, device)
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
# Plots  (identical to CPU version)
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
    labels = [f"gene_{g}" for g in range(G)]

    if true_grns:
        A_true = list(true_grns.values())[0]
        vmin, vmax = min(A_inf.min(), A_true.min()), max(A_inf.max(), A_true.max())
        frob = np.linalg.norm(A_inf - A_true)
        mae  = np.mean(np.abs(A_inf - A_true))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, mat, title, cmap in zip(
            axes,
            [A_true, A_inf, np.abs(A_inf - A_true)],
            ["True A", "Inferred A", "|Error|"],
            ["magma", "magma", "viridis"],
        ):
            kw = {"vmin": vmin, "vmax": vmax} if "Error" not in title else {}
            im = ax.imshow(mat, cmap=cmap, **kw)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(G)); ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_yticks(range(G)); ax.set_yticklabels(labels, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"GRN recovery  —  ||ΔA||_F = {frob:.3f},  MAE = {mae:.3f}", fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(A_inf, cmap="magma")
        ax.set_title("Inferred A", fontsize=10)
        ax.set_xticks(range(G)); ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_yticks(range(G)); ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = out_dir / "grn_recovery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mu_D(result: dict, out_dir: Path) -> None:
    G  = len(result["mu"])
    x  = np.arange(G)
    fig, (ax_mu, ax_D) = plt.subplots(1, 2, figsize=(10, 4))
    ax_mu.bar(x, result["mu"],    color="#4393c3")
    ax_D.bar(x,  result["D_diag"], color="#4393c3")
    for ax, lbl in [(ax_mu, "μ"), (ax_D, "D diagonal")]:
        ax.set_xticks(x); ax.set_xticklabels([f"gene_{g}" for g in x])
        ax.set_ylabel(lbl); ax.set_title(f"Inferred {lbl}")
    fig.tight_layout()
    path = out_dir / "mu_D_inferred.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary(result: dict, true_grns: dict[str, np.ndarray] | None) -> None:
    print("\n--- Inference summary ---")
    print(f"  Inferred μ:     {result['mu'].round(3)}")
    print(f"  Inferred D_diag:{result['D_diag'].round(4)}")
    if true_grns:
        A_true = list(true_grns.values())[0]
        frob   = np.linalg.norm(result["A"] - A_true)
        mae    = np.mean(np.abs(result["A"] - A_true))
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
    parser.add_argument("--n_grad_accum", type=int,   default=N_GRAD_ACCUM)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--eps",          type=float, default=SINKHORN_EPS)
    parser.add_argument("--l2_reg",       type=float, default=L2_REG)
    parser.add_argument("--dtype",        default="float32",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--seed",         type=int,   default=SEED)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path("output-fp-inference-gpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    device    = _get_device()
    obs, G    = load_observations(data_dir)
    true_grns = load_true_grns(data_dir) if (data_dir / "grn_weighted_pop_0.csv").exists() else None

    print(f"Loaded {len(obs)} timepoints × ~{next(iter(obs.values())).shape[0]} cells, {G} genes")
    if true_grns:
        print(f"Ground-truth GRNs: {list(true_grns.keys())}")

    result = run_inference(
        obs, G, device,
        n_particles  = args.n_particles,
        n_grad_accum = args.n_grad_accum,
        dt           = DT,
        sinkhorn_eps = args.eps,
        n_iter       = args.n_iter,
        lr           = args.lr,
        l2_reg       = args.l2_reg,
        dtype_str    = args.dtype,
        seed         = args.seed,
    )

    gene_labels = [f"gene_{g}" for g in range(G)]
    pd.DataFrame(result["A"], index=gene_labels, columns=gene_labels).to_csv(
        out_dir / "A_inferred.csv"
    )
    pd.DataFrame({"loss": result["loss_history"]}).to_csv(
        out_dir / "loss.csv", index=False
    )

    plot_loss(result, out_dir)
    plot_grn_recovery(result, true_grns, out_dir)
    plot_mu_D(result, out_dir)
    print_summary(result, true_grns)
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
