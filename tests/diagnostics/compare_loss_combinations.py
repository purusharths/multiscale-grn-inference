"""
Amortized comparison of parameter-recovery quality across all seven
nonempty subsets of {L_OU, L_FP, L_cons}: which loss combination best
recovers the true (A, mu, sigma) under an equal optimization budget?

"Amortized" here means every combination gets the exact same budget
(same optimizer, same maxiter, same n_proj, same seed) starting from the
exact same wrong initial theta, on the exact same dataset -- so the
comparison isn't skewed by one combination getting more compute or an
easier start.

Same dataset as tests/algorithm/: imports make_stationary_sim/
make_snapshots straight from tests/algorithm/_ground_truth.py (same
seeds), so this is the identical data those pytest specs check properties
of, not a separately-generated lookalike.

Not a test -- a standalone report script; nothing here asserts pass/fail.
Located at tests/ root (not tests/algorithm/, so pytest doesn't try to
collect it as a spec; not src/, since it's a one-shot comparison script,
not reusable library code).

Re-run anytime; always overwrites tests/diagnostics/loss_combination_comparison.{csv,png}.

Usage:
    uv run python tests/diagnostics/compare_loss_combinations.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "algorithm"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from multsc_grn_inference.compute_loss import loss_cons, loss_fp, loss_ou
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _ground_truth import make_snapshots, make_stationary_sim

N_CELLS = 500
DT = 0.3
N_SNAPS = 4
N_PROJ = 50
SEED = 0
MAXITER = 300  # same budget for every combination

sim = make_stationary_sim(seed=42)
TRUE_A, TRUE_MU, TRUE_SIGMA = sim.A, sim.mu0, 0.15
G = TRUE_A.shape[0]

snapshots = make_snapshots(sim, N_CELLS, N_SNAPS, DT, seed=7, shift=-0.6 * TRUE_MU)
chi = [preprocessing(X) for X in snapshots]

COMBINATIONS = {
    "OU":            {"ou"},
    "FP":            {"fp"},
    "Cons":          {"cons"},
    "OU+FP":         {"ou", "fp"},
    "OU+Cons":       {"ou", "cons"},
    "FP+Cons":       {"fp", "cons"},
    "OU+FP+Cons":    {"ou", "fp", "cons"},
}


# ---------------------------------------------------------------------------
# theta <-> unconstrained vector, diagonal A only (matches the ground truth's
# diagonal-dominant, zero-off-diagonal structure)
# ---------------------------------------------------------------------------

def encode(A_diag: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    return np.concatenate([np.log(A_diag), mu, [np.log(sigma)]])


def decode(x: np.ndarray) -> Theta:
    A_diag, mu, log_sigma = x[:G], x[G:2 * G], x[-1]
    return Theta(A=np.diag(np.exp(A_diag)), mu=[mu] * (N_SNAPS - 1), sigma=float(np.exp(log_sigma)))


def make_objective(terms: set[str]):
    def objective(x):
        theta = decode(x)
        total = 0.0
        if "ou" in terms:
            total += loss_ou(theta, snapshots, chi, DT, n_proj=N_PROJ, seed=SEED)
        if "fp" in terms:
            total += loss_fp(theta, snapshots, chi, DT, n_proj=N_PROJ, seed=SEED)
        if "cons" in terms:
            total += loss_cons(theta, snapshots, chi, DT, n_proj=N_PROJ, seed=SEED)
        return total
    return objective


# Deliberately wrong shared starting point for every combination
x0 = encode(0.4 * np.diag(TRUE_A), TRUE_MU * 1.4, 0.4)

rows = []
for name, terms in COMBINATIONS.items():
    print(f"[{name}] optimising ...", flush=True)
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(
        make_objective(terms), x0, method="Nelder-Mead",
        options={"maxiter": MAXITER, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
    )
    elapsed = time.perf_counter() - t0
    theta_hat = decode(res.x)

    A_err = float(np.linalg.norm(np.diag(theta_hat.A) - np.diag(TRUE_A)))
    mu_err = float(np.linalg.norm(theta_hat.mu[0] - TRUE_MU))
    sigma_err = abs(theta_hat.sigma - TRUE_SIGMA)

    rows.append({
        "combination": name,
        "a_diag_hat": np.diag(theta_hat.A).round(3).tolist(),
        "mu_hat": theta_hat.mu[0].round(3).tolist(),
        "sigma_hat": round(theta_hat.sigma, 4),
        "A_err": A_err,
        "mu_err": mu_err,
        "sigma_err": sigma_err,
        "final_objective": res.fun,
        "n_evals": res.nfev,
        "converged": res.success,
        "seconds": round(elapsed, 2),
    })
    print(f"  A_err={A_err:.3f}  mu_err={mu_err:.3f}  sigma_err={sigma_err:.3f}  "
          f"obj={res.fun:.4f}  evals={res.nfev}  {elapsed:.1f}s")

df = pd.DataFrame(rows)
csv_path = Path(__file__).parent / "loss_combination_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print(f"\nTrue A_diag={np.diag(TRUE_A).round(3)}  mu={TRUE_MU.round(3)}  sigma={TRUE_SIGMA}")
print(df[["combination", "A_err", "mu_err", "sigma_err", "final_objective", "seconds"]].to_string(index=False))

# ---- bar chart: recovery error per combination ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes, ["A_err", "mu_err", "sigma_err"],
                          ["||A_diag_hat - A_diag_true||", "||mu_hat - mu_true||", "|sigma_hat - sigma_true|"]):
    ax.bar(df["combination"], df[col], color="#4393c3")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

fig.suptitle(f"Parameter-recovery error by loss combination  (equal budget: maxiter={MAXITER})")
fig.tight_layout()
png_path = Path(__file__).parent / "loss_combination_comparison.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_path}")
