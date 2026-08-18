"""
Loss landscape for L_OU, L_FP, and L_OU+L_FP, varying (a0, mu0) = (A[0,0],
mu[0][0]) over a grid around the true values while holding everything else
fixed -- plus the path a Nelder-Mead search takes through that same 2D
slice, starting from a deliberately wrong point, overlaid on the L_OU+L_FP
panel.

Uses the exact same dataset as tests/algorithm/ (see _ground_truth.py:
same make_stationary_sim/make_snapshots functions, same seeds), so this
landscape is the real loss surface those tests are checking properties of,
not a separately-generated one.

Not a test -- a standalone script. Re-run anytime; always overwrites
tests/diagnostics/loss_landscape.png.

Usage:
    uv run python tests/diagnostics/plot_loss_landscape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "algorithm"))

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from multsc_grn_inference.compute_loss import loss_fp, loss_ou
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _ground_truth import make_snapshots, make_stationary_sim

N_CELLS = 500
DT = 0.3
N_SNAPS = 4
N_GRID = 15
N_PROJ = 50
SEED = 0

sim = make_stationary_sim(seed=42)
TRUE_A, TRUE_MU = sim.A, sim.mu0
TRUE_SIGMA = 0.15
a1_true, mu1_true = TRUE_A[1, 1], TRUE_MU[1]

snapshots = make_snapshots(sim, N_CELLS, N_SNAPS, DT, seed=7, shift=-0.6 * TRUE_MU)
chi = [preprocessing(X) for X in snapshots]


def _theta(a0: float, mu0: float) -> Theta:
    A = np.diag([a0, a1_true])
    mu = np.array([mu0, mu1_true])
    return Theta(A=A, mu=[mu] * (N_SNAPS - 1), sigma=TRUE_SIGMA)


def _l_ou(a0, mu0):
    return loss_ou(_theta(a0, mu0), snapshots, chi, DT, n_proj=N_PROJ, seed=SEED)


def _l_fp(a0, mu0):
    return loss_fp(_theta(a0, mu0), snapshots, chi, DT, n_proj=N_PROJ, seed=SEED)


def _l_ou_fp(a0, mu0):
    return _l_ou(a0, mu0) + _l_fp(a0, mu0)


# ---- grid ----
a0_vals = np.linspace(0.3 * TRUE_A[0, 0], 2.0 * TRUE_A[0, 0], N_GRID)
mu0_vals = np.linspace(0.3 * TRUE_MU[0], 1.7 * TRUE_MU[0], N_GRID)

grids = {"L_OU": np.zeros((N_GRID, N_GRID)), "L_FP": np.zeros((N_GRID, N_GRID)), "L_OU+FP": np.zeros((N_GRID, N_GRID))}
for i, mu0 in enumerate(mu0_vals):
    for j, a0 in enumerate(a0_vals):
        l_ou, l_fp = _l_ou(a0, mu0), _l_fp(a0, mu0)
        grids["L_OU"][i, j] = l_ou
        grids["L_FP"][i, j] = l_fp
        grids["L_OU+FP"][i, j] = l_ou + l_fp
print("Grid done.")

# ---- optimizer trajectory through the same (a0, mu0) slice ----
path = []


def _objective(x):
    a0, mu0 = x
    val = _l_ou_fp(a0, mu0)
    path.append((a0, mu0))
    return val


x0 = np.array([0.4 * TRUE_A[0, 0], 1.5 * TRUE_MU[0]])  # deliberately wrong start
scipy.optimize.minimize(_objective, x0, method="Nelder-Mead",
                        options={"maxiter": 60, "xatol": 1e-2, "fatol": 1e-4})
path = np.array(path)
print(f"Optimizer trajectory: {len(path)} evaluations, "
      f"start=({path[0,0]:.3f},{path[0,1]:.3f}) end=({path[-1,0]:.3f},{path[-1,1]:.3f})")

# ---- plot ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, grid) in zip(axes, grids.items()):
    im = ax.contourf(a0_vals, mu0_vals, grid, levels=20, cmap="RdYlGn_r")
    fig.colorbar(im, ax=ax, label="loss")
    ax.scatter([TRUE_A[0, 0]], [TRUE_MU[0]], color="white", edgecolors="black",
               marker="*", s=150, zorder=5, label="true")
    ax.set_xlabel("a0 = A[0,0]")
    ax.set_ylabel("mu0 = mu[0]")
    ax.set_title(name)
    ax.legend(fontsize=8)

axes[2].plot(path[:, 0], path[:, 1], color="black", lw=1.2, alpha=0.8, zorder=4)
axes[2].scatter(path[:, 0], path[:, 1], color="black", s=6, alpha=0.6, zorder=4)
axes[2].scatter([path[0, 0]], [path[0, 1]], color="cyan", edgecolors="black", s=80, zorder=6, label="start")
axes[2].legend(fontsize=8)

fig.suptitle(f"Loss landscape (a0 vs mu0, other params fixed at true)  --  N={N_CELLS} cells x {N_SNAPS} snapshots")
fig.tight_layout()

out_path = Path(__file__).parent / "loss_landscape.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
