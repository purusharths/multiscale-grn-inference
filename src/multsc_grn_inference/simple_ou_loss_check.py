import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# OU simulation
# ---------------------------------------------------------------------------

def simulate_ou(X0, A, mu, sigma, dt, n_steps=1, rng=None):
    """
    Advance particle cloud X0 (N, G) forward by dt under the OU SDE.

        dc = A(μ − c) dt + σ dW
    """
    if rng is None:
        rng = np.random.default_rng()
    X = X0.copy()
    sub_dt = dt / n_steps
    for _ in range(n_steps):
        drift = (mu - X) @ A.T
        noise = sigma * np.sqrt(sub_dt) * rng.standard_normal(X.shape)
        X = np.maximum(X + drift * sub_dt + noise, 0.0)
    return X


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def kde_sample(X, n_samples, rng):
    """Draw n_samples from a Gaussian KDE fitted to particle cloud X (N, G)."""
    try:
        kde = gaussian_kde(X.T)
    except np.linalg.LinAlgError:
        # degenerate cloud (all particles collapsed) — add tiny jitter and retry
        X = X + rng.standard_normal(X.shape) * 1e-3
        kde = gaussian_kde(X.T)
    seed_val = int(rng.integers(0, 2**31))
    return np.maximum(kde.resample(n_samples, seed=seed_val).T, 0.0)


def w2(X, Y, epsilon=0.01):
    """
    Sinkhorn divergence (debiased) approximation of W2.
    Returns a Python float.
    """
    out = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        jnp.array(X), jnp.array(Y),
        epsilon=epsilon,
    )
    return float(out[0])


# ---------------------------------------------------------------------------
# Loss functions  (single timestep)
# ---------------------------------------------------------------------------

def ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """
    L_OU = W2( KDE(Φ_OU(X_t0; θ)),  KDE(X_t1) )
    """
    rng = np.random.default_rng(seed)
    X_sim     = simulate_ou(X_t0, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_sim_kde = kde_sample(X_sim, len(X_sim), rng)
    X_t1_kde  = kde_sample(X_t1,  len(X_t1),  rng)
    return w2(X_sim_kde, X_t1_kde, epsilon=epsilon)


def fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """
    L_FP = W2( KDE(Φ_OU( KDE_sample(X_t0) ; θ )),  KDE(X_t1) )
    """
    rng = np.random.default_rng(seed)
    X_fp      = kde_sample(X_t0, len(X_t0), rng)
    X_fp_sim  = simulate_ou(X_fp, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_fp_kde  = kde_sample(X_fp_sim, len(X_fp_sim), rng)
    X_t1_kde  = kde_sample(X_t1,     len(X_t1),     rng)
    return w2(X_fp_kde, X_t1_kde, epsilon=epsilon)


def consistency_loss_single_step(X_t0, _X_t1, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """
    L_cons = W2( KDE(Φ_OU(X_t0; θ)),  KDE(Φ_OU(KDE_sample(X_t0); θ)) )

    Does not use X_t1 — measures self-consistency of the two push-forwards.
    """
    rng = np.random.default_rng(seed)
    X_ou     = simulate_ou(X_t0, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_ou_kde = kde_sample(X_ou, len(X_ou), rng)

    X_fp     = kde_sample(X_t0, len(X_t0), rng)
    X_fp_sim = simulate_ou(X_fp, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_fp_kde = kde_sample(X_fp_sim, len(X_fp_sim), rng)

    return w2(X_ou_kde, X_fp_kde, epsilon=epsilon)


def ou_fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """L_OU + L_FP"""
    return (
        ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
        + fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
    )


def total_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0):
    """L_OU + L_FP + L_cons"""
    return (
        ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
        + fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
        + consistency_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
    )


def weighted_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, epsilon=0.01, seed=0, lam=1.0):
    """L_OU + L_FP + lam * L_cons"""
    return (
        ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
        + fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
        + lam * consistency_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, epsilon=epsilon, seed=seed)
    )


# ---------------------------------------------------------------------------
# Parameter encoding  (diagonal A, 2 genes)
# θ = [log_a0, log_a1, mu0, mu1, log_sigma]
# ---------------------------------------------------------------------------

def encode(A, mu, sigma):
    return np.array([
        np.log(A[0, 0]), np.log(A[1, 1]),
        mu[0], mu[1],
        np.log(sigma),
    ])


def decode(theta):
    log_a0, log_a1, mu0, mu1, log_sigma = theta
    A     = np.diag([np.exp(log_a0), np.exp(log_a1)])
    mu    = np.array([mu0, mu1])
    sigma = float(np.exp(log_sigma))
    return A, mu, sigma


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize(loss_fn, X_t0, X_t1, dt, init_theta,
             n_steps=50, epsilon=0.01, seed=0):
    """
    Minimize loss_fn over θ = [log_a0, log_a1, mu0, mu1, log_sigma].

    Returns
    -------
    A_opt, mu_opt, sigma_opt : recovered parameters
    res                      : raw scipy OptimizeResult
    """
    def objective(theta):
        A, mu, sigma = decode(theta)
        return loss_fn(X_t0, X_t1, A, mu, sigma, dt,
                       n_steps=n_steps, epsilon=epsilon, seed=seed)

    res = scipy.optimize.minimize(
        objective, init_theta, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    return decode(res.x), res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_snapshots_line(snapshots, dt, out_path="line_plot.png",
                        true_A=None, true_mu=None):
    """
    Line plot of snapshot distributions vs ground truth OU mean trajectory.

    One subplot per gene:
      - scatter + mean ± std from observed snapshots
      - dashed line: OU-predicted mean under true parameters (if provided)
    """
    times   = [i * dt for i in range(len(snapshots))]
    n_genes = snapshots[0].shape[1]
    rng0    = np.random.default_rng(0)

    fig, axes = plt.subplots(1, n_genes, figsize=(5 * n_genes, 4), sharey=False)
    if n_genes == 1:
        axes = [axes]

    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(snapshots) - 1, 1)) for i in range(len(snapshots))]

    n_show = min(60, len(snapshots[0]))
    idx    = rng0.choice(len(snapshots[0]), n_show, replace=False)

    # ground truth mean trajectory: propagate m_{k+1} = m_k + A(μ - m_k)*dt
    gt_means = None
    if true_A is not None and true_mu is not None:
        m = snapshots[0].mean(axis=0).copy()
        gt_means = [m.copy()]
        for _ in range(len(snapshots) - 1):
            m = m + (true_mu - m) @ true_A.T * dt
            gt_means.append(m.copy())

    for g, ax in enumerate(axes):
        # faint cell lines (unpaired — visual guide only)
        for cell_vals in zip(*[s[idx, g] for s in snapshots]):
            ax.plot(times, cell_vals, color="gray", alpha=0.12, lw=0.6)

        # scatter per timepoint
        for t, snap, col in zip(times, snapshots, colors):
            jit = rng0.uniform(-0.01 * dt, 0.01 * dt, len(snap))
            ax.scatter(np.full(len(snap), t) + jit, snap[:, g],
                       s=4, alpha=0.35, color=col)

        # observed mean ± std ribbon
        means = [s[:, g].mean() for s in snapshots]
        stds  = [s[:, g].std()  for s in snapshots]
        ax.plot(times, means, color="black", lw=2, marker="o", ms=7, zorder=5,
                label="observed mean")
        ax.fill_between(times,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color="black", label="±1 std")

        # ground truth OU mean trajectory
        if gt_means is not None:
            gt_vals = [m[g] for m in gt_means]
            ax.plot(times, gt_vals, color="tab:red", lw=2, ls="--",
                    marker="s", ms=6, zorder=6, label="OU true mean")

        ax.set_xticks(times)
        ax.set_xlabel("time")
        ax.set_ylabel("expression")
        ax.set_title(f"gene {g}")
        ax.legend(fontsize=8)

    fig.suptitle("Snapshot distributions vs ground truth OU mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _mean_trajectory(snapshots, A, mu, dt):
    """Propagate mean forward: m_{k+1} = m_k + A(μ - m_k)*dt."""
    m = snapshots[0].mean(axis=0).copy()
    traj = [m.copy()]
    for _ in range(len(snapshots) - 1):
        m = m + (mu - m) @ A.T * dt
        traj.append(m.copy())
    return traj


def plot_phase_portrait(snapshots, A, mu, dt, out_path="phase_portrait.png",
                        results=None):
    """
    Phase portrait in gene-0 × gene-1 space.

    Left panel  — cell scatter coloured by timepoint + OU drift streamlines.
    Right panel — mean trajectories: observed (black), true OU (green dashed),
                  and one dotted line per entry in results (recovered params).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(snapshots) - 1, 1)) for i in range(len(snapshots))]

    all_X = np.vstack(snapshots)
    x_lo, x_hi = all_X[:, 0].min(), all_X[:, 0].max()
    y_lo, y_hi = all_X[:, 1].min(), all_X[:, 1].max()
    pad = 0.3
    gx = np.linspace(x_lo - pad, x_hi + pad, 20)
    gy = np.linspace(y_lo - pad, y_hi + pad, 20)
    GX, GY = np.meshgrid(gx, gy)
    pts   = np.stack([GX.ravel(), GY.ravel()], axis=1)
    drift = (mu - pts) @ A.T
    DX    = drift[:, 0].reshape(GX.shape)
    DY    = drift[:, 1].reshape(GY.shape)
    speed = np.sqrt(DX**2 + DY**2) + 1e-9

    for ax in axes:
        ax.streamplot(gx, gy, DX, DY, color=speed, cmap="Greens",
                      linewidth=0.8, arrowsize=0.8, density=1.2)
        for i, (snap, col) in enumerate(zip(snapshots, colors)):
            ax.scatter(snap[:, 0], snap[:, 1], s=4, alpha=0.3, color=col,
                       label=f"t{i}")
        ax.scatter(*mu, marker="*", s=180, color="white", edgecolors="black",
                   zorder=5, label="μ (target)")
        ax.set_xlabel("gene 0")
        ax.set_ylabel("gene 1")

    axes[0].set_title("Drift field  +  snapshot scatter")
    axes[0].legend(fontsize=7, markerscale=2, loc="upper left")

    # right panel: mean trajectories
    means_obs = [s.mean(axis=0) for s in snapshots]
    axes[1].plot([m[0] for m in means_obs], [m[1] for m in means_obs],
                 color="black", lw=2, marker="o", ms=6, zorder=5,
                 label="observed mean")

    true_traj = _mean_trajectory(snapshots, A, mu, dt)
    axes[1].plot([m[0] for m in true_traj], [m[1] for m in true_traj],
                 color="tab:green", lw=2, ls="--", marker="s", ms=6, zorder=4,
                 label="true OU mean")

    if results is not None:
        res_colors = plt.cm.tab10(np.linspace(0, 0.8, len(results)))
        for (name, res), col in zip(results.items(), res_colors):
            A_r  = np.diag([res["a0"], res["a1"]])
            mu_r = np.array([res["mu0"], res["mu1"]])
            traj = _mean_trajectory(snapshots, A_r, mu_r, dt)
            axes[1].plot([m[0] for m in traj], [m[1] for m in traj],
                         color=col, lw=1.5, ls=":", marker="^", ms=5, zorder=3,
                         label=name)

    axes[1].set_title("Mean trajectories")
    axes[1].legend(fontsize=8)

    fig.suptitle("Phase portrait  (OU drift field in gene space)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_loss_landscape(X_t0, X_t1, dt, true_A, true_mu, true_sigma,
                        out_path="loss_landscape.png",
                        n_grid=20, epsilon=0.01, seed=0):
    """
    2D loss heatmap: vary a0 (x-axis) and mu0 (y-axis) while holding all
    other parameters at their true values.  Shows OU+FP, OU+FP+Cons, and
    OU+FP+0.5·Cons side by side.
    """
    a0_true  = true_A[0, 0]
    mu0_true = true_mu[0]

    a0_vals  = np.linspace(max(a0_true  * 0.2, 0.1), a0_true  * 2.5, n_grid)
    mu0_vals = np.linspace(max(mu0_true * 0.3, 0.1), mu0_true * 1.7, n_grid)

    loss_fns = {
        "OU+FP":          ou_fp_loss_single_step,
        "OU+FP+Cons":     total_loss_single_step,
        "OU+FP+0.5·Cons": lambda X0, X1, A, mu, sigma, dt, **kw: weighted_loss_single_step(
                               X0, X1, A, mu, sigma, dt, lam=0.5, **kw),
    }
    grids = {name: np.zeros((n_grid, n_grid)) for name in loss_fns}

    total = n_grid * n_grid * len(loss_fns)
    done  = 0
    for j, a0 in enumerate(a0_vals):
        for i, mu0 in enumerate(mu0_vals):
            A_   = np.diag([a0, true_A[1, 1]])
            mu_  = np.array([mu0, true_mu[1]])
            for name, fn in loss_fns.items():
                grids[name][i, j] = fn(
                    X_t0, X_t1, A_, mu_, true_sigma, dt,
                    epsilon=epsilon, seed=seed,
                )
            done += len(loss_fns)
            if done % (n_grid * len(loss_fns)) == 0:
                print(f"  landscape: {done}/{total}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, grid) in zip(axes, grids.items()):
        im = ax.contourf(a0_vals, mu0_vals, grid, levels=20, cmap="RdYlGn_r")
        fig.colorbar(im, ax=ax, label="loss")
        ax.axvline(a0_true,  color="white", lw=1.5, ls="--")
        ax.axhline(mu0_true, color="white", lw=1.5, ls="--")
        ax.scatter([a0_true], [mu0_true], color="white", s=80,
                   marker="*", zorder=5, label="true")
        ax.set_xlabel("a0  (rate gene 0)")
        ax.set_ylabel("μ₀  (target gene 0)")
        ax.set_title(f"{name} loss")
        ax.legend(fontsize=8)

    fig.suptitle("Loss landscape: a0 vs μ₀  (all other params fixed at true)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_recovery(results, true_A, true_mu, true_sigma, out_path="recovery.png"):
    """
    Bar chart of recovered vs true parameters for each loss.
    """
    param_names  = ["a0", "a1", "mu0", "mu1", "sigma"]
    true_vals    = [true_A[0,0], true_A[1,1], true_mu[0], true_mu[1], true_sigma]
    loss_names   = list(results.keys())

    fig, axes = plt.subplots(1, len(param_names), figsize=(14, 3.5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(loss_names)))

    for ax, pname, true_val in zip(axes, param_names, true_vals):
        recovered = [results[ln][pname] for ln in loss_names]
        bars = ax.bar(loss_names, recovered, color=colors, edgecolor="white")
        ax.axhline(true_val, color="black", lw=1.5, ls="--", label="true")
        ax.set_title(pname, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
        for bar, val in zip(bars, recovered):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Parameter recovery by loss", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    OUT = Path("output/simple_ou_check")
    OUT.mkdir(parents=True, exist_ok=True)

    # --- generate simple OU data -------------------------------------------
    TRUE_A     = np.diag([1.5, 1.2])
    TRUE_MU    = np.array([3.0, 2.5])
    TRUE_SIGMA = 0.3
    DT         = 0.5
    N_CELLS    = 300
    SIM_STEPS  = 50       # sub-steps for ground-truth simulation
    OPT_STEPS  = 10       # sub-steps inside the loss (faster)

    rng = np.random.default_rng(42)
    X_t0 = rng.multivariate_normal(TRUE_MU * 0.3, 0.1 * np.eye(2), size=N_CELLS)
    X_t0 = np.maximum(X_t0, 0.05)
    X_t1 = simulate_ou(X_t0, TRUE_A, TRUE_MU, TRUE_SIGMA, DT,
                        n_steps=SIM_STEPS, rng=rng)

    snapshots = [X_t0, X_t1]

    # --- line plot (first) ----------------------------------------------------
    plot_snapshots_line(snapshots, DT, out_path=OUT / "line_plot.png",
                        true_A=TRUE_A, true_mu=TRUE_MU)

    # --- loss landscape -------------------------------------------------------
    print("Computing loss landscape (may take ~1 min) ...")
    plot_loss_landscape(X_t0, X_t1, DT, TRUE_A, TRUE_MU, TRUE_SIGMA,
                        out_path=OUT / "loss_landscape.png",
                        n_grid=18, epsilon=0.05)

    # --- optimise combined losses ---------------------------------------------
    init_theta = encode(
        np.diag([1.0, 1.0]),
        np.array([2.0, 2.0]),
        0.5,
    )

    LAM = 0.5
    loss_fns = {
        "ou+fp":              ou_fp_loss_single_step,
        "ou+fp+cons":         total_loss_single_step,
        f"ou+fp+{LAM}*cons":  lambda X0, X1, A, mu, sigma, dt, **kw: weighted_loss_single_step(
                                  X0, X1, A, mu, sigma, dt, lam=LAM, **kw),
    }

    results = {}
    for name, fn in loss_fns.items():
        print(f"Optimising [{name}] ...")
        (A_opt, mu_opt, sigma_opt), res = optimize(
            fn, X_t0, X_t1, DT, init_theta.copy(),
            n_steps=OPT_STEPS, epsilon=0.05,
        )
        results[name] = {
            "a0": A_opt[0,0], "a1": A_opt[1,1],
            "mu0": mu_opt[0], "mu1": mu_opt[1],
            "sigma": sigma_opt,
            "converged": res.success, "fun": res.fun,
        }
        print(f"  converged={res.success}  fun={res.fun:.5f}")
        print(f"  A={A_opt.diagonal().round(3)}  mu={mu_opt.round(3)}  sigma={sigma_opt:.3f}")

    print(f"\nTrue: A={TRUE_A.diagonal()}  mu={TRUE_MU}  sigma={TRUE_SIGMA}")

    # --- phase portrait (after optimisation so recovered means can be overlaid)
    plot_phase_portrait(snapshots, TRUE_A, TRUE_MU, DT,
                        out_path=OUT / "phase_portrait.png",
                        results=results)

    # --- recovery plot --------------------------------------------------------
    plot_recovery(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                  out_path=OUT / "recovery.png")

    print(f"\nAll outputs in {OUT.resolve()}")
