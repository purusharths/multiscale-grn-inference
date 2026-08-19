"""
Does JKOnet* recover the single-gene-knockout network as well as this repo's
OU+Cons fit -- and how much of any gap is just JKOnet* not knowing {mu_k}?
See README.md for the full design rationale and caveats.

Dataset: the same coupled single-gene-knockout scenario as
../interventions/single-gene-knockout/loss-combinations/compare_losses_single_gene_knockout.py
(gene_KO_GENE knocked out at t_star, network_density>0 so A has real
off-diagonal edges), run twice:
  - "knockout": as-is, mu(t) shifts at t_star.
  - "control":  identical A/mu0/D (same seed, t_star pushed past T so mu is
    effectively constant) -- isolates "can JKOnet* fit a static potential at
    all" from "can it handle the regime switch".

Our side: OU+Cons fit (full A via log-diag + free off-diagonals, sigma;
mu fixed at the known {mu_k} -- same parametrization as
compare_losses_single_gene_knockout.py). OU+Cons had the best edge_corr in
that script's sweep (loss_comparison_knockout.csv), so it stands in for
"the current implementation" here rather than re-running the whole sweep.

JKOnet* side, two solvers by default (see README.md for why these two, and
why jkonet-star-linear-potential-internal -- the only variant whose fit
reduces to a comparable A_hat -- isn't in the default sweep at our gene
count):
  - jkonet-star-potential-internal   -- static potential + diffusion
  - jkonet-star-time-potential       -- potential(x, t): can represent the
                                         regime switch

Metric: one-step-ahead sliced-W2 (this repo's _sliced_w2, computed here on
both methods' predicted clouds -- see README.md "Shared metric") averaged
over intervals, plus A_err/offdiag_err/edge_corr wherever an A_hat exists.

Not a test -- a standalone report script; re-run anytime, overwrites this
folder's jko_comparison.{csv,png}. Requires one-time setup:
    bash tests/diagnostics/jko-testing/setup_jkonet_star.sh

Usage:
    uv run python "tests/diagnostics/jko-testing/compare_jkonet_star.py"
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

sys.path.insert(0, str(Path(__file__).parent.parent / "interventions"))

from multsc_grn_inference.compute_loss import _sliced_w2, loss_cons, loss_ou
from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta

from _intervention_ground_truth import (  # noqa: E402
    effective_sigma,
    extract_snapshots,
    make_single_gene_knockout_sim,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_GENES = 8
KO_GENE = 3
N_CELLS = 1500
N_EVAL_CELLS = 500  # independent eval ensemble, see build_scenario()
N_SNAPS = 10
T = 7.0
T_STAR = 2.0
NETWORK_DENSITY = 0.5
SEED = 42

N_PROJ = 30
MAXITER = 1500
OPTIMIZER_METHOD = "Nelder-Mead"

JKO_EPOCHS = 300
JKO_SOLVERS = [
    "jkonet-star-potential-internal",
    "jkonet-star-time-potential",
    # jkonet-star-linear-potential-internal deliberately excluded at
    # N_GENES=8: its polynomial feature count is (degree+1)**G - 1 (a full
    # per-dimension power grid, not a total-degree-bounded basis), so
    # degree=2 alone gives 3**8-1=6560 features here -- the driver's
    # per-batch feature-outer-product einsum OOMs. See README.md "Feature-
    # config gotcha". Safe to add back below ~5-6 genes.
]

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent
JKO_REPO = REPO_ROOT / "external" / "jkonet-star"
JKO_PYTHON = JKO_REPO / ".venv" / "bin" / "python"
JKO_OUT = JKO_REPO / "out" / "jko-testing"


def _require_jko_setup() -> None:
    if not JKO_PYTHON.exists():
        raise SystemExit(
            f"JKOnet* not set up: {JKO_PYTHON} missing.\n"
            f"Run: bash {HERE / 'setup_jkonet_star.sh'}"
        )


# ---------------------------------------------------------------------------
# Scenario construction (paired knockout / control, see module docstring)
# ---------------------------------------------------------------------------

def build_scenario(t_star: float | None):
    """
    t_star=None -> pushed past T (effectively-constant-mu control).

    Draws TWO independent cell ensembles from the same sim (same A/mu0/D/
    mu(t), fresh rng draws): `snapshots` for fitting/coupling-generation on
    both sides, `eval_snapshots` for scoring only -- never touched by either
    method's training, so the one-step-ahead comparison is out-of-sample for
    both (see README.md "Shared metric" and _jko_star_driver.py's docstring
    for why this replaces JKOnet*'s own train/test split).
    """
    sim = make_single_gene_knockout_sim(
        num_genes=N_GENES, knockout_gene=KO_GENE, seed=SEED,
        t_star=T_STAR if t_star is None else t_star, network_density=NETWORK_DENSITY,
    )
    if t_star is None:
        sim.mu_kwargs["t_star"] = T + 100.0  # never fires within [0, T]
    snapshots, times, mu_known, dt = extract_snapshots(sim, N_CELLS, N_SNAPS, T=T)
    eval_snapshots, _, _, _ = extract_snapshots(sim, N_EVAL_CELLS, N_SNAPS, T=T)
    return sim, snapshots, eval_snapshots, mu_known, dt


# ---------------------------------------------------------------------------
# Our fit: OU+Cons, full A + sigma, mu fixed at known {mu_k}
# (same parametrization as compare_losses_single_gene_knockout.py)
# ---------------------------------------------------------------------------

OFF = [(i, j) for i in range(N_GENES) for j in range(N_GENES) if i != j]


def encode(A: np.ndarray, sigma: float) -> np.ndarray:
    return np.concatenate([np.log(np.diag(A)), [A[i, j] for i, j in OFF], [np.log(sigma)]])


def decode(x: np.ndarray, mu_known) -> Theta:
    A = np.diag(np.exp(x[:N_GENES]))
    for n, (i, j) in enumerate(OFF):
        A[i, j] = x[N_GENES + n]
    return Theta(A=A, mu=mu_known, sigma=float(np.exp(x[-1])))


def fit_ou_cons(snapshots, mu_known, dt) -> tuple[Theta, dict]:
    chi = [preprocessing(X) for X in snapshots]

    def objective(x):
        theta = decode(x, mu_known)
        return (
            loss_ou(theta, snapshots, chi, dt, n_proj=N_PROJ, seed=0)
            + loss_cons(theta, snapshots, chi, dt, n_proj=N_PROJ, seed=0)
        )

    x0 = encode(np.eye(N_GENES) * 1.2, 0.5)
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(
        objective, x0, method=OPTIMIZER_METHOD,
        options={"maxiter": MAXITER, "xatol": 1e-3, "fatol": 1e-6, "adaptive": True},
    )
    elapsed = time.perf_counter() - t0
    return decode(res.x, mu_known), {"n_evals": res.nfev, "seconds": round(elapsed, 1), "final_objective": res.fun}


def one_step_ahead_w2_ours(theta_hat: Theta, eval_snapshots, dt) -> float:
    """Roll the (held-out) eval snapshot at t_k one interval forward under
    theta_hat, sliced-W2 against the eval snapshot at t_{k+1} -- mirrors
    JKOnet*'s error_wasserstein_one_step_ahead, but scored on the same
    independent eval set as the JKOnet* side (see build_scenario())."""
    rng = np.random.default_rng(0)
    errs = []
    for k in range(len(eval_snapshots) - 1):
        pred = ou_gene_expression(eval_snapshots[k], theta_hat, k, dt, rng=rng)
        errs.append(_sliced_w2(pred, eval_snapshots[k + 1], N_PROJ, rng))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# JKOnet* side: convert snapshots -> their data.npy/sample_labels.npy,
# shell out to data_generator.py (couplings/densities) then our driver
# (training + one-step-ahead rollout dump) -- see README.md.
# ---------------------------------------------------------------------------

def _write_flat(data_dir: Path, prefix: str, snapshots: list[np.ndarray]) -> None:
    t1, n = len(snapshots), snapshots[0].shape[0]
    np.save(data_dir / f"{prefix}.npy", np.concatenate(snapshots, axis=0))
    np.save(data_dir / f"{prefix.replace('data', 'sample_labels')}.npy", np.repeat(np.arange(t1), n))


def write_jko_dataset(name: str, snapshots: list[np.ndarray], eval_snapshots: list[np.ndarray]) -> None:
    data_dir = JKO_REPO / "data" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_flat(data_dir, "data", snapshots)
    _write_flat(data_dir, "eval_data", eval_snapshots)  # read by _jko_star_driver.py


def run_jko_data_generator(name: str) -> None:
    # test-ratio 0: we score on our own independent eval set (written above),
    # not JKOnet*'s train/test split -- see _jko_star_driver.py's docstring.
    subprocess.run(
        [
            str(JKO_PYTHON), "data_generator.py",
            "--load-from-file", name,
            "--test-ratio", "0", "--n-gmm-components", "8",
        ],
        cwd=JKO_REPO, check=True,
    )


def run_jko_solver(name: str, solver: str, data_dim: int) -> dict:
    out_dir = JKO_OUT / name / solver
    subprocess.run(
        [
            str(JKO_PYTHON), str(HERE / "_jko_star_driver.py"),
            "--dataset", name, "--data-dim", str(data_dim), "--solver", solver,
            "--epochs", str(JKO_EPOCHS), "--seed", "0",
            "--out-dir", str(out_dir.relative_to(JKO_REPO)),
        ],
        cwd=JKO_REPO, check=True,
    )
    with open(out_dir / "result.json") as f:
        result = json.load(f)
    return result, out_dir


def one_step_ahead_w2_jko(out_dir: Path, eval_snapshots: list[np.ndarray]) -> float:
    """Loads _jko_star_driver.py's dumped predicted_t{k}.npy clouds (rolled
    forward from the independent eval set) and scores them against the same
    eval set with the SAME _sliced_w2 used for our own OU rollout -- see
    README.md "Shared metric"."""
    rng = np.random.default_rng(0)
    errs = []
    for k in range(len(eval_snapshots) - 1):
        pred = np.load(out_dir / f"predicted_t{k + 1}.npy")
        errs.append(_sliced_w2(pred, eval_snapshots[k + 1], N_PROJ, rng))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _require_jko_setup()
    rows = []
    A_hats: dict[str, np.ndarray] = {}

    for scenario_name, t_star in [("knockout", T_STAR), ("control", None)]:
        print(f"\n=== scenario: {scenario_name} ===", flush=True)
        sim, snapshots, eval_snapshots, mu_known, dt = build_scenario(t_star)
        A_true = sim.A
        true_off = np.array([A_true[i, j] for i, j in OFF])

        print("[ours: OU+Cons] fitting ...", flush=True)
        theta_hat, fit_info = fit_ou_cons(snapshots, mu_known, dt)
        hat_off = np.array([theta_hat.A[i, j] for i, j in OFF])
        w2_ours = one_step_ahead_w2_ours(theta_hat, eval_snapshots, dt)
        rows.append({
            "scenario": scenario_name, "method": "ours (OU+Cons)",
            "one_step_w2": w2_ours,
            "A_err": float(np.linalg.norm(theta_hat.A - A_true)),
            "offdiag_err": float(np.linalg.norm(hat_off - true_off)),
            "edge_corr": float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0,
            "sigma_err": abs(theta_hat.sigma - effective_sigma(sim)),
            **fit_info,
        })
        A_hats[f"{scenario_name}__ours"] = theta_hat.A
        print(f"  w2={w2_ours:.4f}  A_err={rows[-1]['A_err']:.3f}  edge_corr={rows[-1]['edge_corr']:+.3f}")

        dataset_name = f"single_gene_knockout_{scenario_name}"
        write_jko_dataset(dataset_name, snapshots, eval_snapshots)
        print("[jkonet*] computing couplings/densities ...", flush=True)
        run_jko_data_generator(dataset_name)

        for solver in JKO_SOLVERS:
            print(f"[jkonet*: {solver}] training ...", flush=True)
            t0 = time.perf_counter()
            result, out_dir = run_jko_solver(dataset_name, solver, N_GENES)
            elapsed = time.perf_counter() - t0
            w2_jko = one_step_ahead_w2_jko(out_dir, eval_snapshots)

            row = {
                "scenario": scenario_name, "method": f"jkonet* ({solver})",
                "one_step_w2": w2_jko, "seconds": round(elapsed, 1),
                "final_objective": result["final_loss"],
            }
            if result.get("A_hat_from_quadratic_features") is not None:
                A_hat = np.array(result["A_hat_from_quadratic_features"])
                hat_off = np.array([A_hat[i, j] for i, j in OFF])
                row["A_err"] = float(np.linalg.norm(A_hat - A_true))
                row["offdiag_err"] = float(np.linalg.norm(hat_off - true_off))
                row["edge_corr"] = (
                    float(np.corrcoef(true_off, hat_off)[0, 1]) if hat_off.std() > 1e-12 else 0.0
                )
                A_hats[f"{scenario_name}__{solver}"] = A_hat
            rows.append(row)
            print(f"  w2={w2_jko:.4f}  {elapsed:.0f}s")

    df = pd.DataFrame(rows)
    csv_path = HERE / "jko_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df, A_hats)


def _plot(df: pd.DataFrame, A_hats: dict[str, np.ndarray]) -> None:
    scenarios = df["scenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5))
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        colors = ["#1a9641" if m.startswith("ours") else "#4393c3" for m in sub["method"]]
        ax.barh(sub["method"], sub["one_step_w2"], color=colors)
        ax.set_title(f"{scenario}: one-step-ahead sliced-W2 (lower better)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Ours (knows {{mu_k}}) vs JKOnet* (doesn't) -- single-gene knockout, "
        f"{N_GENES} genes, density={NETWORK_DENSITY}"
    )
    fig.tight_layout()
    png_path = HERE / "jko_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
