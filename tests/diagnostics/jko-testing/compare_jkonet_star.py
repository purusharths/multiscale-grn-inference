"""
Does JKOnet* recover the single-gene-knockout network as well as this repo's
Algorithm 1 -- and how much of any gap is just JKOnet* not knowing {mu_k}?
See README.md for the full design rationale and caveats.

Dataset: the same coupled single-gene-knockout scenario as
../interventions/single-gene-knockout/loss-combinations/compare_losses_single_gene_knockout.py
(gene_KO_GENE knocked out at t_star, network_density>0 so A has real
off-diagonal edges), run twice:
  - "knockout": as-is, mu(t) shifts at t_star.
  - "control":  identical A/mu0/D (same seed, t_star pushed past T so mu is
    effectively constant) -- isolates "can JKOnet* fit a static potential at
    all" from "can it handle the regime switch".

Our side: all 6 loss combinations from
../interventions/_ours_combinations.py (== that script's sweep, Cons-alone
excluded as degenerate there). An earlier version of this script picked a
single "best" combination (OU+Cons) to stand in for "the current
implementation" -- see _ours_combinations.py's docstring for why that
was weaker than it looked (every combination's A_err/offdiag_err in the
prior sweep was nearly identical; edge_corr was the only thing that moved,
across a single noisy run per combination). Comparing all six avoids
leaning on that noise.

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
                row.update(_edge_metrics(true_off, hat_off))
                A_hats[f"{scenario_name}__{solver}"] = A_hat
            rows.append(row)
            print(f"  w2={w2_jko:.4f}  {elapsed:.0f}s")

    df = pd.DataFrame(rows)
    csv_path = HERE / "jko_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    _plot(df, A_hats)
    _plot_matrices(A_trues, A_hats)


def _plot(df: pd.DataFrame, A_hats: dict[str, np.ndarray]) -> None:
    scenarios = df["scenario"].unique()
    n_rows = df.groupby("scenario").size().max()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 0.5 * n_rows + 1.5))
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        colors = ["#1a9641" if m.startswith("ours") else "#4393c3" for m in sub["method"]]
        ax.barh(sub["method"], sub["one_step_w2"], color=colors)
        ax.set_title(f"{scenario}: one-step-ahead sliced-W2 (lower better)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Ours -- all 6 loss combinations (knows {{mu_k}}) vs JKOnet* (doesn't) -- "
        f"single-gene knockout, {N_GENES} genes, density={NETWORK_DENSITY}"
    )
    fig.tight_layout()
    png_path = HERE / "jko_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _plot_matrices(A_trues: dict[str, np.ndarray], A_hats: dict[str, np.ndarray]) -> None:
    """TRUE A vs every recovered A_hat (all 6 "ours" combinations, plus any
    JKOnet* linear-solver A_hat if that solver was included), one row per
    scenario, shared per-scenario color scale."""
    scenarios = list(A_trues.keys())
    combo_names = list(COMBINATIONS.keys())
    extra_labels = sorted({
        key.split("__", 1)[1] for key in A_hats
        if not key.split("__", 1)[1].startswith("ours_")
    })
    panel_labels = ["TRUE A"] + [f"ours ({c})" for c in combo_names] + extra_labels
    n_cols = len(panel_labels)

    fig, axes = plt.subplots(len(scenarios), n_cols, figsize=(2.0 * n_cols, 2.2 * len(scenarios) + 0.5))
    axes = np.atleast_2d(axes)

    im = None
    for row, scenario in enumerate(scenarios):
        A_true = A_trues[scenario]
        vmax = np.abs(A_true).max() * 1.2
        for col, label in enumerate(panel_labels):
            ax = axes[row, col]
            if label == "TRUE A":
                mat = A_true
            elif label.startswith("ours ("):
                mat = A_hats[f"{scenario}__ours_{label[len('ours ('):-1]}"]
            else:
                mat = A_hats[f"{scenario}__{label}"]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=8, fontweight="bold" if label == "TRUE A" else "normal")
            if col == 0:
                ax.set_ylabel(scenario, fontsize=9)

    fig.suptitle(f"Recovered A -- {N_GENES} genes, density={NETWORK_DENSITY}", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="A_ij", pad=0.01)
    png_path = HERE / "jko_recovered_matrices.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
