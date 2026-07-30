import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from scipy.stats import gaussian_kde


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


def w2(X, Y, epsilon=0.05):
    """Sinkhorn divergence (debiased) W2. Accurate but slow — use for analysis."""
    out = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        jnp.array(X), jnp.array(Y),
        epsilon=epsilon,
    )
    return float(out[0])


def sliced_w2(X, Y, n_proj=50, seed=0):
    """Sliced Wasserstein² — ~50× faster than Sinkhorn. Default for optimisation."""
    rng  = np.random.default_rng(seed)
    G    = X.shape[1]
    dirs = rng.standard_normal((n_proj, G))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    n_min = min(len(X), len(Y))
    t_x = np.linspace(0, 1, len(X))
    t_y = np.linspace(0, 1, len(Y))
    t_q = np.linspace(0, 1, n_min)
    total = 0.0
    for d in dirs:
        total += float(np.mean(
            (np.interp(t_q, t_x, np.sort(X @ d)) -
             np.interp(t_q, t_y, np.sort(Y @ d))) ** 2
        ))
    return total / n_proj


# ---------------------------------------------------------------------------
# Loss functions  (single timestep)
# dist_fn : callable (X, Y) -> float   default = sliced_w2 (fast)
#           pass w2 or functools.partial(w2, epsilon=0.05) for Sinkhorn
# ---------------------------------------------------------------------------

def ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, seed=0, dist_fn=None):
    """L_OU = dist( KDE(Φ_OU(X_t0; θ)),  KDE(X_t1) )"""
    _d = dist_fn if dist_fn is not None else sliced_w2
    rng = np.random.default_rng(seed)
    X_sim     = simulate_ou(X_t0, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_sim_kde = kde_sample(X_sim, len(X_sim), rng)
    X_t1_kde  = kde_sample(X_t1,  len(X_t1),  rng)
    return _d(X_sim_kde, X_t1_kde)


def fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, seed=0, dist_fn=None):
    """L_FP = dist( KDE(Φ_OU( KDE_sample(X_t0); θ )),  KDE(X_t1) )"""
    _d = dist_fn if dist_fn is not None else sliced_w2
    rng = np.random.default_rng(seed)
    X_fp      = kde_sample(X_t0, len(X_t0), rng)
    X_fp_sim  = simulate_ou(X_fp, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_fp_kde  = kde_sample(X_fp_sim, len(X_fp_sim), rng)
    X_t1_kde  = kde_sample(X_t1,     len(X_t1),     rng)
    return _d(X_fp_kde, X_t1_kde)


def consistency_loss_single_step(X_t0, _X_t1, A, mu, sigma, dt, n_steps=1, seed=0, dist_fn=None):
    """L_cons = dist( KDE(Φ_OU(X_t0; θ)),  KDE(Φ_OU(KDE_sample(X_t0); θ)) )"""
    _d = dist_fn if dist_fn is not None else sliced_w2
    rng = np.random.default_rng(seed)
    X_ou     = simulate_ou(X_t0, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_ou_kde = kde_sample(X_ou, len(X_ou), rng)
    X_fp     = kde_sample(X_t0, len(X_t0), rng)
    X_fp_sim = simulate_ou(X_fp, A, mu, sigma, dt, n_steps=n_steps, rng=rng)
    X_fp_kde = kde_sample(X_fp_sim, len(X_fp_sim), rng)
    return _d(X_ou_kde, X_fp_kde)


def ou_fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, seed=0, dist_fn=None):
    """L_OU + L_FP"""
    return (
        ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
        + fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
    )


def total_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, seed=0, dist_fn=None):
    """L_OU + L_FP + L_cons"""
    return (
        ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
        + fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
        + consistency_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
    )


def weighted_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=1, seed=0, dist_fn=None, lam=1.0):
    """L_OU + L_FP + lam * L_cons"""
    return (
        ou_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
        + fp_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
        + lam * consistency_loss_single_step(X_t0, X_t1, A, mu, sigma, dt, n_steps=n_steps, seed=seed, dist_fn=dist_fn)
    )



def encode(A, mu, sigma):
    """θ = [log_a00,...,log_aGG (diag), off-diag row-major, mu0,...,muG, log_sigma]."""
    G = len(mu)
    log_diag = [np.log(A[g, g]) for g in range(G)]
    off_diag = [A[i, j] for i in range(G) for j in range(G) if i != j]
    return np.array(log_diag + off_diag + list(mu) + [np.log(sigma)])


def decode(theta):
    """Inverse of encode. Infers G from len(theta) = G² + G + 1."""
    n = len(theta)
    G = int(round((-1 + np.sqrt(1 + 4 * (n - 1))) / 2))
    A = np.zeros((G, G))
    for g in range(G):
        A[g, g] = np.exp(theta[g])
    off_idx = G
    for i in range(G):
        for j in range(G):
            if i != j:
                A[i, j] = theta[off_idx]
                off_idx += 1
    mu    = np.array(theta[G * G: G * G + G])
    sigma = float(np.exp(theta[-1]))
    return A, mu, sigma


def optimize(loss_fn, X_t0, X_t1, dt, init_theta,
             n_steps=50, seed=0, dist_fn=None):
    """
    Minimize loss_fn over θ = [log_a0, log_a1, mu0, mu1, log_sigma].

    dist_fn : distance function (X, Y) -> float
              default = sliced_w2 (fast); pass w2 for Sinkhorn accuracy.

    Returns
    -------
    (A_opt, mu_opt, sigma_opt), res, history
    history : list of objective values at every function evaluation
    """
    history = []

    def objective(theta):
        A, mu, sigma = decode(theta)
        val = loss_fn(X_t0, X_t1, A, mu, sigma, dt,
                      n_steps=n_steps, seed=seed, dist_fn=dist_fn)
        history.append(val)
        return val

    res = scipy.optimize.minimize(
        objective, init_theta, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    return decode(res.x), res, history


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_POP_PALETTE   = ["#3B4CC0", "#C76DB4", "#E07A5F", "#6EC5E9", "#FAC748"]
_RESULT_COLORS = {"ou+fp": "royalblue", "ou+fp+cons": "crimson"}
_ARROW_COLOR   = "#C8C8C8"

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

    colors = [_POP_PALETTE[i % len(_POP_PALETTE)] for i in range(len(snapshots))]

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


def plot_phase_portrait(snapshots, A, mu, dt, out_path="phase_portrait.svg", results=None):
    """
    G×G subplot grid (G = min(n_genes, 3)).

    Diagonal (g, g)     — mean expression vs time for gene g.
    Off-diagonal (i, j) — quiver drift field in gene_j × gene_i space,
                          snapshot scatter, and mean trajectories overlaid.
    """
    G      = min(snapshots[0].shape[1], 3)
    times  = [k * dt for k in range(len(snapshots))]
    all_X  = np.vstack(snapshots)
    mean_X = all_X.mean(axis=0)

    fig, axes = plt.subplots(G, G, figsize=(5 * G, 5 * G))
    if G == 1:
        axes = np.array([[axes]])

    t_colors = [_POP_PALETTE[k % len(_POP_PALETTE)] for k in range(len(snapshots))]

    # precompute trajectories
    means_obs = [s.mean(axis=0) for s in snapshots]
    true_traj = _mean_trajectory(snapshots, A, mu, dt)

    res_colors      = []
    recovered_trajs = []
    if results:
        res_colors = [_RESULT_COLORS.get(n, _POP_PALETTE[i % len(_POP_PALETTE)])
                      for i, n in enumerate(results.keys())]
        for name, res in results.items():
            recovered_trajs.append((name, _mean_trajectory(snapshots, res["A"], res["mu"], dt)))

    for row in range(G):
        for col in range(G):
            ax = axes[row, col]

            if row == col:
                # ---- diagonal: time series for gene row --------------------
                ax.plot(times, [m[row] for m in means_obs],
                        color="black", lw=2.5, marker="o", ms=7, label="observed")
                ax.plot(times, [m[row] for m in true_traj],
                        color="tab:green", lw=2, ls="--", marker="s", ms=6,
                        label="true OU")
                for (name, traj), c in zip(recovered_trajs, res_colors):
                    ax.plot(times, [m[row] for m in traj],
                            color=c, lw=1.8, ls=":", marker="^", ms=6, label=name)
                ax.set_xlabel("time")
                ax.set_ylabel("mean expression")
                ax.set_title(f"gene {row}")
                ax.set_xticks(times)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)

            else:
                # ---- off-diagonal: phase portrait in gene_col × gene_row ---
                xi_lo, xi_hi = all_X[:, col].min(), all_X[:, col].max()
                yi_lo, yi_hi = all_X[:, row].min(), all_X[:, row].max()
                pad = 0.3
                gx = np.linspace(xi_lo - pad, xi_hi + pad, 10)
                gy = np.linspace(yi_lo - pad, yi_hi + pad, 10)
                GX, GY = np.meshgrid(gx, gy)

                DX_g = np.zeros_like(GX)
                DY_g = np.zeros_like(GY)
                for ii in range(GX.shape[0]):
                    for jj in range(GX.shape[1]):
                        x_pt         = mean_X.copy()
                        x_pt[col]    = GX[ii, jj]
                        x_pt[row]    = GY[ii, jj]
                        d            = A @ (mu - x_pt)
                        DX_g[ii, jj] = d[col]
                        DY_g[ii, jj] = d[row]

                spd = np.sqrt(DX_g**2 + DY_g**2) + 1e-9
                ax.quiver(GX, GY, DX_g / spd, DY_g / spd,
                          color=_ARROW_COLOR, alpha=0.7, scale=20, width=0.004)

                # scatter
                for snap, c in zip(snapshots, t_colors):
                    ax.scatter(snap[:, col], snap[:, row], s=3, alpha=0.25, color=c)

                # mean trajectories
                ax.plot([m[col] for m in means_obs], [m[row] for m in means_obs],
                        color="black", lw=2, marker="o", ms=6)
                ax.plot([m[col] for m in true_traj], [m[row] for m in true_traj],
                        color="tab:green", lw=2, ls="--", marker="s", ms=5)
                for (name, traj), c in zip(recovered_trajs, res_colors):
                    ax.plot([m[col] for m in traj], [m[row] for m in traj],
                            color=c, lw=1.5, ls=":", marker="^", ms=5)

                ax.scatter([mu[col]], [mu[row]], marker="*", s=150,
                           color="white", edgecolors="black", zorder=5)
                ax.set_xlabel(f"gene {col}")
                ax.set_ylabel(f"gene {row}")

    fig.suptitle("Pairwise phase portraits + mean trajectories  (diagonal = time series)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
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
                    seed=seed,
                )
            done += len(loss_fns)
            if done % (n_grid * len(loss_fns)) == 0:
                print(f"  landscape: {done}/{total}")

    # quiver subsample — every other grid point to avoid clutter
    step = max(1, n_grid // 10)
    qa   = a0_vals[::step]
    qmu  = mu0_vals[::step]
    QA, QMU = np.meshgrid(qa, qmu)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, grid) in zip(axes, grids.items()):
        im = ax.contourf(a0_vals, mu0_vals, grid, levels=20, cmap="RdYlGn_r")
        fig.colorbar(im, ax=ax, label="loss")

        # gradient (finite diff on the grid) → quiver arrows pointing downhill
        dL_dmu, dL_da = np.gradient(grid, mu0_vals, a0_vals)
        ga = -dL_da[::step, ::step]
        gmu = -dL_dmu[::step, ::step]
        mag = np.sqrt(ga**2 + gmu**2) + 1e-12
        ax.quiver(QA, QMU, ga / mag, gmu / mag,
                  color=_ARROW_COLOR, alpha=0.8, scale=25, width=0.004)

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
    G            = true_A.shape[0]
    param_names  = ([f"a{i}{j}" for i in range(G) for j in range(G)]
                    + [f"mu{g}" for g in range(G)] + ["sigma"])
    true_vals    = ([true_A[i, j] for i in range(G) for j in range(G)]
                    + list(true_mu) + [true_sigma])
    loss_names   = list(results.keys())

    fig, axes = plt.subplots(1, len(param_names),
                             figsize=(2.2 * len(param_names), 3.5), sharey=False)
    colors = [_RESULT_COLORS.get(n, _POP_PALETTE[i % len(_POP_PALETTE)])
              for i, n in enumerate(loss_names)]

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


def plot_convergence(histories, out_path="convergence.png"):
    """
    Objective value vs number of function evaluations, one line per loss.
    Uses a running minimum so the curve is always non-increasing (best-so-far).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [_RESULT_COLORS.get(n, _POP_PALETTE[i % len(_POP_PALETTE)])
              for i, n in enumerate(histories.keys())]

    for (name, hist), col in zip(histories.items(), colors):
        best = np.minimum.accumulate(hist)
        ax.plot(best, label=name, lw=1.8, color=col)

    ax.set_xlabel("function evaluations")
    ax.set_ylabel("best loss so far")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_title("Optimizer convergence", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_comparison(results, true_A, true_mu, true_sigma, out_path="comparison.png"):
    """
    Absolute error |recovered − true| per parameter, grouped by loss.
    Directly shows which loss gives better recovery for each parameter.
    """
    G           = true_A.shape[0]
    param_names = ([f"a{i}{j}" for i in range(G) for j in range(G)]
                   + [f"mu{g}" for g in range(G)] + ["sigma"])
    true_vals   = ([true_A[i, j] for i in range(G) for j in range(G)]
                   + list(true_mu) + [true_sigma])
    loss_names  = list(results.keys())

    n_losses = len(loss_names)
    x        = np.arange(len(param_names))
    width    = 0.8 / n_losses
    colors   = [_RESULT_COLORS.get(n, _POP_PALETTE[i % len(_POP_PALETTE)])
                for i, n in enumerate(loss_names)]

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, col) in enumerate(zip(loss_names, colors)):
        errors = [abs(results[name][p] - tv) for p, tv in zip(param_names, true_vals)]
        offset = (i - n_losses / 2 + 0.5) * width
        bars = ax.bar(x + offset, errors, width, label=name, color=col,
                      alpha=0.85, edgecolor="white")
        for bar, err in zip(bars, errors):
            if err > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{err:.3f}", ha="center", va="bottom", fontsize=7,
                        color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylabel("|recovered − true|")
    ax.set_title("Parameter recovery error by loss  (lower = better)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_A_heatmap(results, true_A, out_path="A_heatmap.png"):
    """
    Side-by-side heatmaps: true A and each recovered A.
    Annotated with entry values; shared colour scale.
    """
    names = ["true"] + list(results.keys())
    mats  = [true_A] + [res["A"] for res in results.values()]
    n     = len(names)
    G     = true_A.shape[0]
    vmax  = max(np.abs(m).max() for m in mats)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2))
    if n == 1:
        axes = [axes]

    im = None
    for ax, name, mat in zip(axes, names, mats):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(G)); ax.set_xticklabels([f"g{j}" for j in range(G)])
        ax.set_yticks(range(G)); ax.set_yticklabels([f"g{i}" for i in range(G)])
        for i in range(G):
            for j in range(G):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(mat[i, j]) > 0.5 * vmax else "black")
        title_col = "black"
        if name in _RESULT_COLORS:
            title_col = _RESULT_COLORS[name]
        ax.set_title(name, fontsize=10, color=title_col)

    fig.colorbar(im, ax=axes[-1], label="A entry", shrink=0.85)
    fig.suptitle("Interaction matrix A: ground truth vs recovered", fontsize=11)
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
    TRUE_A     = np.array([[1.5, 0.3, 0.0],
                           [0.2, 1.2, 0.1],
                           [0.0, 0.2, 0.9]])
    TRUE_MU    = np.array([3.0, 2.5, 2.0])
    TRUE_SIGMA = 0.3
    DT         = 0.5
    N_CELLS    = 300
    SIM_STEPS  = 50
    OPT_STEPS  = 10
    G          = len(TRUE_MU)

    rng = np.random.default_rng(42)
    X_t0 = rng.multivariate_normal(TRUE_MU * 0.3, 0.1 * np.eye(G), size=N_CELLS)
    X_t0 = np.maximum(X_t0, 0.05)
    X_t1 = simulate_ou(X_t0, TRUE_A, TRUE_MU, TRUE_SIGMA, DT,
                        n_steps=SIM_STEPS, rng=rng)

    snapshots = [X_t0, X_t1]

    # --- line plot ------------------------------------------------------------
    plot_snapshots_line(snapshots, DT, out_path=OUT / "line_plot.png",
                        true_A=TRUE_A, true_mu=TRUE_MU)

    # --- optimise: ou+fp vs ou+fp+cons ---------------------------------------
    init_theta = encode(np.diag([1.0] * G), np.array([2.0] * G), 0.5)

    loss_fns = {
        "ou+fp":      ou_fp_loss_single_step,
        "ou+fp+cons": total_loss_single_step,
    }

    results   = {}
    histories = {}
    for name, fn in loss_fns.items():
        print(f"Optimising [{name}] ...")
        (A_opt, mu_opt, sigma_opt), res, hist = optimize(
            fn, X_t0, X_t1, DT, init_theta.copy(),
            n_steps=OPT_STEPS,
        )
        results[name] = {
            "A": A_opt,
            "mu": mu_opt,
            "sigma": sigma_opt,
            **{f"a{i}{j}": A_opt[i, j] for i in range(G) for j in range(G)},
            **{f"mu{g}": mu_opt[g] for g in range(G)},
        }
        histories[name] = hist
        print(f"  converged={res.success}  fun={res.fun:.5f}")
        print(f"  A={A_opt.diagonal().round(3)}  mu={mu_opt.round(3)}  sigma={sigma_opt:.3f}")

    print(f"\nTrue A:\n{TRUE_A}")
    print(f"True mu={TRUE_MU}  sigma={TRUE_SIGMA}")

    # --- plots ----------------------------------------------------------------
    plot_convergence(histories, out_path=OUT / "convergence.png")
    plot_comparison(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                    out_path=OUT / "comparison.png")
    plot_recovery(results, TRUE_A, TRUE_MU, TRUE_SIGMA,
                  out_path=OUT / "recovery.png")
    plot_phase_portrait(snapshots, TRUE_A, TRUE_MU, DT,
                        out_path=OUT / "phase_portrait.svg",
                        results=results)
    plot_A_heatmap(results, TRUE_A, out_path=OUT / "A_heatmap.png")

    print(f"\nAll outputs in {OUT.resolve()}")
