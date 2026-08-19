"""
Runs INSIDE external/jkonet-star/.venv, with cwd=external/jkonet-star --
see README.md. Not importable from the main project's environment (needs
their torch/wandb/pinned-jax stack).

Trains one JKOnet* solver on a dataset already preprocessed by their
data_generator.py (couplings + density_and_grads present under
data/<dataset>/, built from data.npy/sample_labels.npy -- the FIT set),
then rolls the fitted potential forward one step at a time from an
INDEPENDENT eval set (eval_data.npy/eval_sample_labels.npy, written by
compare_jkonet_star.py from a fresh draw of the same sim -- never seen by
training on either side of the comparison) and dumps the raw predicted
particle clouds -- not a distance number -- so the comparison script can
score both methods with the same estimator (_sliced_w2) on its own side.

Deliberately does NOT use dataset.py's PopulationEvalDataset / its
train/test split: that split only exists on the JKOnet* side, so scoring
against it would evaluate JKOnet* out-of-sample while this repo's OU+Cons
fit (scored separately, in compare_jkonet_star.py) would be in-sample --
an unfair comparison. Using one independent eval set for both sides avoids
that asymmetry entirely, which is also why dt is hardcoded to 1.0 here
(PopulationEvalDataset does the same -- "dt does not actually matter for
learning... everything can be scaled accordingly" -- so this isn't a
divergence from their convention).

For the linear/closed-form solvers, the feature config is overridden here
(see README.md "Feature-config gotcha"): the default RBF grid is
combinatorial in the number of genes and hangs past ~4 dims, and a degree-2
polynomial basis is both tractable and exactly expressive enough for our
quadratic OU potential.

Usage (from external/jkonet-star, via that venv's python):
    python _jko_star_driver.py \
        --dataset <name> --data-dim 8 --solver jkonet-star-potential-internal \
        --epochs 500 --seed 0 --out-dir out/jko-testing/<run>/<solver>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())  # repo root (models/, dataset.py, utils/) -- see README.md

from models import EnumMethod, get_model  # noqa: E402
from utils.sde_simulator import get_SDE_predictions  # noqa: E402


def load_eval_trajectory(dataset: str) -> dict[int, np.ndarray]:
    """{timestep: (n_eval, data_dim)} from eval_data.npy/eval_sample_labels.npy."""
    data = np.load(os.path.join("data", dataset, "eval_data.npy"))
    labels = np.load(os.path.join("data", dataset, "eval_sample_labels.npy"))
    trajectory: dict[int, list] = defaultdict(list)
    for value, label in zip(data, labels):
        trajectory[int(label)].append(value)
    return {t: np.array(v) for t, v in trajectory.items()}


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        return [numpy_collate(s) for s in zip(*batch)]
    return np.array(batch)


# Degree-2 polynomials are exact for our quadratic OU potential; RBFs are
# disabled outright (their center grid is combinatorial in data_dim -- see
# README.md). Only touches config['energy']['linear'], which the neural
# solvers (jkonet-star-potential-internal, jkonet-star-time-potential) never
# read.
def _patch_linear_feature_config(config: dict) -> None:
    linear = config["energy"]["linear"]
    linear["features"]["polynomials"]["degree"] = 2
    linear["features"]["polynomials"]["sines"] = False
    linear["features"]["polynomials"]["cosines"] = False
    linear["features"]["rbfs"]["types"] = []
    linear["features"]["rbfs"]["n_centers_per_dim"] = 1


def main(args: argparse.Namespace) -> None:
    key = jax.random.PRNGKey(args.seed)

    config = yaml.safe_load(open("config.yaml"))
    config.update(yaml.safe_load(open("config-jkonet-extra.yaml")))
    config["train"]["epochs"] = args.epochs
    config["train"]["save_locally"] = False
    _patch_linear_feature_config(config)

    solver = EnumMethod(args.solver)

    if solver in (
        EnumMethod.JKO_NET_STAR_LINEAR,
        EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL,
        EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL,
    ):
        # (degree+1)**data_dim - 1 features (full per-dimension power grid,
        # not a total-degree cutoff -- see README.md "Feature-config
        # gotcha"). Past ~2000 features the per-batch (features_dim,
        # features_dim) einsum in train_step OOMs and gets SIGKILLed with no
        # useful traceback, so fail loudly here instead.
        n_features = 3 ** args.data_dim - 1
        if n_features > 2000:
            raise SystemExit(
                f"jkonet-star-linear-* at data_dim={args.data_dim}, degree=2 "
                f"needs {n_features} polynomial features -- will OOM. "
                f"See README.md 'Feature-config gotcha'."
            )

    dt = 1.0  # see module docstring
    model = get_model(solver, config, args.data_dim, dt)
    state = model.create_state(key)
    dataset_train = model.load_dataset(args.dataset)

    torch.manual_seed(args.seed)
    batch_size = config["train"]["batch_size"]
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size if batch_size > 0 else len(dataset_train),
        shuffle=True,
        collate_fn=numpy_collate,
    )

    train_step = model.train_step
    if config["train"]["epochs"] > 1:
        train_step = jax.jit(train_step)

    losses = []
    for _epoch in range(1, config["train"]["epochs"] + 1):
        epoch_loss = 0.0
        for sample in loader_train:
            loss, state = train_step(state, sample)
            epoch_loss += float(loss)
        losses.append(epoch_loss / max(len(loader_train), 1))

    potential = model.get_potential(state)
    beta = model.get_beta(state)
    interaction = model.get_interaction(state)

    # One-step-ahead rollout from the independent eval set (see module
    # docstring for why this isn't dataset.py's train/test split).
    eval_trajectory = load_eval_trajectory(args.dataset)
    n_eval_steps = len(eval_trajectory) - 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key, key_eval = jax.random.split(key)
    for t in range(n_eval_steps):
        init = eval_trajectory[t]
        predictions = get_SDE_predictions(
            str(solver), dt, 1, t + 1, potential, beta, interaction, key_eval, init,
        )
        np.save(out_dir / f"predicted_t{t + 1}.npy", np.asarray(predictions[-1]))

    # Quadratic coefficients readable only for the degree<=2, RBF-free linear
    # solvers -- see README.md. For JKOnetStarLinear, state IS
    # (theta1, theta2, theta3) already unpacked by train_step
    # (models/jkonet_star.py); theta1 is the potential-energy block, one
    # coefficient per feature, in the same order self.fns/self.features(x)
    # were built: itertools.product(range(degree+1), repeat=G) with
    # sum(e)>0. degree=2 here (patched above), so the degree-2 diagonal
    # terms (e_i=2) map to A_hat's diagonal and the degree-(1,1) cross terms
    # (e_i=e_j=1) map to its off-diagonals.
    quadratic_readout = None
    if solver in (
        EnumMethod.JKO_NET_STAR_LINEAR,
        EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL,
        EnumMethod.JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL,
    ):
        import itertools

        G = args.data_dim
        theta1 = np.asarray(state[0]).reshape(-1)
        exps = [e for e in itertools.product(range(3), repeat=G) if sum(e) > 0]
        A_hat = np.zeros((G, G))
        for coeff, e in zip(theta1, exps):
            if sum(e) != 2:
                continue
            nz = [i for i, v in enumerate(e) if v > 0]
            if len(nz) == 1:
                A_hat[nz[0], nz[0]] = 2.0 * coeff
            else:
                i, j = nz
                A_hat[i, j] = A_hat[j, i] = coeff
        quadratic_readout = A_hat.tolist()

    with open(out_dir / "result.json", "w") as f:
        json.dump(
            {
                "solver": str(solver),
                "dataset": args.dataset,
                "epochs": config["train"]["epochs"],
                "final_loss": losses[-1] if losses else None,
                "loss_curve": losses,
                "beta": float(beta) if isinstance(beta, (int, float)) or hasattr(beta, "item") else None,
                "A_hat_from_quadratic_features": quadratic_readout,
            },
            f,
            indent=2,
        )
    print(f"[{solver}] done. final_loss={losses[-1] if losses else float('nan'):.4f} -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dim", type=int, required=True)
    parser.add_argument("--solver", type=str, required=True, choices=[e.value for e in EnumMethod])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, required=True)
    main(parser.parse_args())
