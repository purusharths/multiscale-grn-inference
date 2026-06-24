"""
Bayesian OU-process parameter recovery from non-stationary simulation data.

Simulates trajectories using NetworkSimulatorNonStationaryMu (the same
simulator as run_non_stationary.py) and recovers A, mu, D via MCMC (PyMC).

Since the simulator stores full per-cell trajectories (N, T, G), we use
the same transition-batch approach as the reference notebook (10-genes.ipynb):
for each sampled trajectory pick (X_t, X_{t+1}) pairs.

Note on mu: all four populations use time-varying mu (sigmoid, linear,
heaviside, constant).  The constant-mu OU model recovers an effective
time-averaged mu.  Recovery quality is best for the "constant" population
and degrades gracefully for sharper transitions.

Outputs per population
----------------------
  data_overview_<pop>.png    — original dynamics (mean ± std + phase plot)
  A_recovery_<pop>.png       — inferred A | true A | posterior std
  mu_D_recovery_<pop>.png    — mu and D bar charts with true values
  trace_<pop>.png            — posterior traces
  idata_<pop>.nc             — full ArviZ InferenceData
  posterior_summary_<pop>.csv
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from dataset_gen_dynamic.datagen.non_stationary_sim import NetworkSimulatorNonStationaryMu

# --------------------------------------------------------------------------
# Config  (mirrors run_non_stationary.py)
# --------------------------------------------------------------------------

NUM_GENES = 6
SIM_T = 4.0
SIM_DT = 0.005
SAVE_EVERY = 15
NUM_SAMPLES = 300      # trajectories per population

dt_obs = SIM_DT * SAVE_EVERY   # observation interval

POPULATIONS = [
    {"label": "pop_0", "seed": 0, "mu_mode": "sigmoid",   "mu_kwargs": {}},
    {"label": "pop_1", "seed": 1, "mu_mode": "linear",    "mu_kwargs": {}},
    {"label": "pop_2", "seed": 2, "mu_mode": "heaviside", "mu_kwargs": {}},
    {"label": "pop_3", "seed": 3, "mu_mode": "constant",  "mu_kwargs": {}},
]

N_TRAJ_BATCH = 100    # trajectories to subsample for inference
N_TP_BATCH   = 4      # consecutive transition steps per trajectory
MCMC_DRAWS   = 1000
MCMC_TUNE    = 500
MCMC_CHAINS  = 2

# --------------------------------------------------------------------------
# Transition batches 
# --------------------------------------------------------------------------

def build_transition_batches(
    data: np.ndarray,
    n_samples: int,
    n_timepoints: int,
    rng: int | np.random.Generator = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Sample n_samples trajectories; from each take n_timepoints consecutive
    (X_t, X_{t+1}) pairs.

    Parameters
    ----------
    data : (N, T, G)

    Returns
    -------
    X_t, X_t1 : (M, G)  with M = n_samples * n_timepoints
    """
    N, T, G = data.shape
    rng = np.random.default_rng(rng) if isinstance(rng, int) else rng
    n_s = min(n_samples, N)
    n_tp = min(n_timepoints, T - 1)

    sample_idx = rng.choice(N, size=n_s, replace=False)
    t_idx = np.arange(n_tp)

    X_t_list, X_t1_list = [], []
    for s in sample_idx:
        X = data[s]                # (T, G)
        X_t_list.append(X[t_idx])
        X_t1_list.append(X[t_idx + 1])

    X_t  = np.vstack(X_t_list).astype(np.float32)
    X_t1 = np.vstack(X_t1_list).astype(np.float32)
    info = {"sample_idx": sample_idx, "t_idx": t_idx, "M": X_t.shape[0], "G": G}
    return X_t, X_t1, info


# --------------------------------------------------------------------------
# Bayesian model
# --------------------------------------------------------------------------

def build_model(X_t: np.ndarray, X_t1: np.ndarray, dt: float, G: int) -> pm.Model:
    """
    Likelihood: X_{t+1} | X_t, A, μ, D ~ N(X_t + dt·A(μ − X_t), diag(D·dt))
    """
    I = np.eye(G, dtype=np.float32)
    with pm.Model() as model:
        Xt  = pm.Data("X_t",  X_t)
        Xt1 = pm.Data("X_t1", X_t1)

        A_diag = pm.Normal("A_diag", mu=1.2, sigma=0.3, shape=G)
        A_off  = pm.Laplace("A_off", mu=0.0, b=2.0, shape=(G, G))
        A      = pm.Deterministic("A", A_off * (1.0 - I) + pt.diag(A_diag))

        mu = pm.TruncatedNormal("mu", mu=3.0, sigma=0.5,
                                lower=0.5, upper=6.0, shape=G)
        log_D = pm.TruncatedNormal("log_D", mu=np.log(0.15), sigma=0.3,
                                   lower=np.log(0.01), upper=np.log(1.0), shape=G)
        D_diag = pm.Deterministic("D_diag", pt.exp(log_D))

        pred  = pm.Deterministic("pred", Xt + dt * ((mu - Xt) @ A.T))
        sigma = pt.sqrt(D_diag * dt)

        pm.Normal("obs", mu=pred, sigma=sigma, observed=Xt1)

    return model


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def plot_data_overview(
    data: np.ndarray,
    time_grid: np.ndarray,
    sim: NetworkSimulatorNonStationaryMu,
    pop_label: str,
    mu_mode: str,
    out_dir: Path,
) -> None:
    """
    Original data visualisation:
      Left panel  — mean ± std of each gene over time
      Right panel — gene_0 vs gene_1 phase plot (50 trajectories, coloured by time)
    """
    G = data.shape[2]
    means = data.mean(axis=0)   # (T, G)
    stds  = data.std(axis=0)

    fig, (ax_dyn, ax_phase) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- dynamics ----
    colors = plt.cm.tab10(np.linspace(0, 1, G))
    for g in range(G):
        ax_dyn.plot(time_grid, means[:, g], color=colors[g],
                    label=f"gene_{g}", lw=1.8)
        ax_dyn.fill_between(time_grid,
                             means[:, g] - stds[:, g],
                             means[:, g] + stds[:, g],
                             color=colors[g], alpha=0.12)
    ax_dyn.set_xlabel("Time")
    ax_dyn.set_ylabel("Expression (mean ± std)")
    ax_dyn.set_title(f"{pop_label} ({mu_mode}) — population dynamics")
    ax_dyn.legend(fontsize=8, ncol=2)

    # ---- phase plot ----
    n_show = min(50, data.shape[0])
    idx = np.random.default_rng(0).choice(data.shape[0], size=n_show, replace=False)
    norm = plt.Normalize(time_grid[0], time_grid[-1])
    cmap = plt.cm.viridis
    for i in idx:
        tx = data[i, :, 0]
        ty = data[i, :, 1]
        for t in range(len(time_grid) - 1):
            ax_phase.plot(tx[t:t+2], ty[t:t+2],
                          color=cmap(norm(time_grid[t])), lw=0.6, alpha=0.35)
    mean_x = data[:, :, 0].mean(axis=0)
    mean_y = data[:, :, 1].mean(axis=0)
    ax_phase.plot(mean_x, mean_y, color="white", lw=2.5, zorder=5, label="mean")
    ax_phase.plot(mean_x, mean_y, color="black", lw=1.0, zorder=4)
    ax_phase.scatter([mean_x[0]],  [mean_y[0]],  color="lime", s=60, zorder=6, label="t=0")
    ax_phase.scatter([mean_x[-1]], [mean_y[-1]], color="red",  s=60, zorder=6,
                     label=f"t={time_grid[-1]:.1f}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_phase, label="time")
    ax_phase.set_xlabel("gene_0"); ax_phase.set_ylabel("gene_1")
    ax_phase.set_title("Phase plot (gene_0 vs gene_1)")
    ax_phase.legend(fontsize=8)
    ax_phase.set_facecolor("#1a1a2e")

    fig.suptitle(f"{pop_label} — original simulation data", fontsize=12)
    fig.tight_layout()
    path = out_dir / f"data_overview_{pop_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_recovery(
    sim: NetworkSimulatorNonStationaryMu,
    idata: az.InferenceData,
    pop_label: str,
    out_dir: Path,
) -> None:
    """3-panel A heatmap + mu / D bar charts with true values."""
    post = idata.posterior
    G = sim.num_genes

    A_mean = post["A"].mean(dim=("chain", "draw")).values
    A_std  = post["A"].std(dim=("chain", "draw")).values
    mu_mean = post["mu"].mean(dim=("chain", "draw")).values
    mu_std  = post["mu"].std(dim=("chain", "draw")).values
    D_mean  = post["D_diag"].mean(dim=("chain", "draw")).values
    D_std   = post["D_diag"].std(dim=("chain", "draw")).values

    A_true  = sim.A
    mu_true = sim.mu0
    D_true  = np.diag(sim.D)

    # ---- A heatmap ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = min(A_mean.min(), A_true.min())
    vmax = max(A_mean.max(), A_true.max())

    for ax, mat, title in zip(
        axes,
        [A_mean, A_true, A_std],
        ["Inferred A (mean)", "True A", "Posterior std(A)"],
    ):
        cmap = "magma" if "std" not in title else "viridis"
        kw = {"vmin": vmin, "vmax": vmax} if "std" not in title else {}
        im = ax.imshow(mat, cmap=cmap, **kw)
        ax.set_title(f"{pop_label} — {title}")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(G)); ax.set_yticks(range(G))

    frob = np.linalg.norm(A_mean - A_true)
    mae  = np.mean(np.abs(A_mean - A_true))
    fig.suptitle(f"||A_inferred − A_true||_F = {frob:.3f},  MAE = {mae:.3f}", fontsize=10)
    fig.tight_layout()
    path = out_dir / f"A_recovery_{pop_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- mu and D ----
    fig, (ax_mu, ax_D) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(G); w = 0.35

    ax_mu.bar(x - w/2, mu_mean, w, yerr=mu_std, capsize=4,
              color="#4393c3", label="Inferred")
    ax_mu.bar(x + w/2, mu_true, w, color="#d6604d", alpha=0.7, label="True μ₀")
    ax_mu.set_xticks(x); ax_mu.set_xticklabels([f"g{g}" for g in x])
    ax_mu.set_ylabel("μ"); ax_mu.set_title(f"{pop_label} — μ recovery")
    ax_mu.legend()

    ax_D.bar(x - w/2, D_mean, w, yerr=D_std, capsize=4,
             color="#4393c3", label="Inferred")
    ax_D.bar(x + w/2, D_true, w, color="#d6604d", alpha=0.7, label="True D_diag")
    ax_D.set_xticks(x); ax_D.set_xticklabels([f"g{g}" for g in x])
    ax_D.set_ylabel("D diag"); ax_D.set_title(f"{pop_label} — D recovery")
    ax_D.legend()

    fig.tight_layout()
    path = out_dir / f"mu_D_recovery_{pop_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_posterior_traces(
    idata: az.InferenceData, pop_label: str, out_dir: Path
) -> None:
    az.plot_trace(idata, var_names=["mu", "A_diag", "log_D"])
    plt.suptitle(f"{pop_label} — posterior traces", y=1.01, fontsize=10)
    plt.tight_layout()
    path = out_dir / f"trace_{pop_label}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("output-bayes-non-stationary/")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for pop in POPULATIONS:
        label    = pop["label"]
        mu_mode  = pop["mu_mode"]
        print(f"\n{'='*60}")
        print(f"Population: {label}  (mu_mode={mu_mode})")
        print(f"{'='*60}")

        # ---- simulate ----
        sim = NetworkSimulatorNonStationaryMu(
            num_genes=NUM_GENES,
            network_density=0.3,
            seed=pop["seed"],
            mu_mode=mu_mode,
            mu_kwargs=pop["mu_kwargs"],
        )
        print(f"  Simulating {NUM_SAMPLES} trajectories ...")
        data, time_grid = sim.simulate(
            T=SIM_T, num_samples=NUM_SAMPLES,
            save_every=SAVE_EVERY, dt=SIM_DT,
        )
        print(f"  data shape: {data.shape}")   # (N, T, G)

        # ---- plot original data ----
        plot_data_overview(data, time_grid, sim, label, mu_mode, out_dir)

        # ---- build transition batches ----
        X_t, X_t1, info = build_transition_batches(
            data,
            n_samples=N_TRAJ_BATCH,
            n_timepoints=N_TP_BATCH,
            rng=pop["seed"] + 42,
        )
        print(f"  Transition batch: {info}")

        # ---- MCMC ----
        model = build_model(X_t, X_t1, dt_obs, G=NUM_GENES)
        print("  Sampling ...")
        with model:
            idata = pm.sample(
                draws=MCMC_DRAWS,
                tune=MCMC_TUNE,
                chains=MCMC_CHAINS,
                target_accept=0.9,
                random_seed=42,
                progressbar=True,
            )

        idata.to_netcdf(out_dir / f"idata_{label}.nc")

        # ---- diagnostics ----
        summary = az.summary(idata, var_names=["A", "mu", "D_diag"], hdi_prob=0.95)
        summary["population"] = label
        all_summaries.append(summary)
        summary.to_csv(out_dir / f"posterior_summary_{label}.csv")

        # ---- plots ----
        plot_recovery(sim, idata, label, out_dir)
        plot_posterior_traces(idata, label, out_dir)

        # ---- console report ----
        post = idata.posterior
        A_mean = post["A"].mean(dim=("chain", "draw")).values
        frob = np.linalg.norm(A_mean - sim.A)
        print(f"  ||A_inferred − A_true||_F = {frob:.4f}")
        print(f"  μ_inferred: {post['mu'].mean(dim=('chain','draw')).values.round(3)}")
        print(f"  μ_true:     {sim.mu0.round(3)}")
        print(f"  D_inferred: {post['D_diag'].mean(dim=('chain','draw')).values.round(3)}")
        print(f"  D_true:     {np.diag(sim.D).round(3)}")

    pd.concat(all_summaries).to_csv(out_dir / "all_posterior_summaries.csv")
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
