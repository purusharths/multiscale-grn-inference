"""
Diagnostic plots shared by the OU loss-check scripts
(multi_step_ou_loss_check.py, destructive_ou_loss_check.py).

Extracted from the now-removed simple_ou_loss_check.py, whose loss
computation (simulate_ou, *_loss_single_step, encode/decode) was a
duplicate of loss.py -- already covered by tests/loss/ and tests/sde_fp/ --
but these plotting functions had no equivalent elsewhere.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
