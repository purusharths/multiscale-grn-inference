"""
Outer bilevel optimisation loop.

Minimises L(theta) = L_OU + L_FP + L_cons over theta = {A, sigma^2}
using torch.optim.Adam.

The inner JKO argmin is solved with a stop-gradient at each outer step
(see jko.py).  Implicit differentiation through the JKO is a future TODO.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gaussian_kde

from bip import theta_to_params
from config import GRNConfig
from loss import compute_loss


def optimise(
    theta0: dict,
    dataset: dict[int, np.ndarray],
    chi_hat: dict[int, gaussian_kde],
    mu_list: list[torch.Tensor],
    config: GRNConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """
    Run the outer Adam optimisation.

    Parameters
    ----------
    theta0   : BIP initialisation dict from bip_initialise()
    dataset  : timepoint → (N_k, G) snapshots
    chi_hat  : timepoint → scipy KDE
    mu_list  : list of (G,) tensors — known intervention means per interval
    config   : GRNConfig

    Returns
    -------
    A_hat        : (G, G) tensor — inferred drift matrix
    log_sigma_hat: () tensor    — inferred log diffusion
    history      : list of per-step dicts {"step", "loss", "L_OU", "L_FP", "L_cons"}
    """
    G = next(iter(dataset.values())).shape[1]
    A, log_sigma = theta_to_params(theta0, G, config.device)

    optimiser = torch.optim.Adam([A, log_sigma], lr=config.lr)
    rng = np.random.default_rng(config.seed)

    history: list[dict] = []

    print(f"Starting outer optimisation ({config.n_outer_steps} steps) ...")
    for step in range(1, config.n_outer_steps + 1):
        optimiser.zero_grad()
        loss, bd = compute_loss(A, log_sigma, dataset, chi_hat, mu_list, config, rng)
        loss.backward()
        optimiser.step()

        record = {"step": step, "loss": float(loss.detach()), **bd}
        history.append(record)

        if step % max(1, config.n_outer_steps // 20) == 0 or step == 1:
            print(
                f"  step {step:4d}/{config.n_outer_steps}  "
                f"loss={record['loss']:.4f}  "
                f"L_OU={record['L_OU']:.4f}  "
                f"L_FP={record['L_FP']:.4f}  "
                f"L_cons={record['L_cons']:.4f}"
            )

    return A.detach(), log_sigma.detach(), history
