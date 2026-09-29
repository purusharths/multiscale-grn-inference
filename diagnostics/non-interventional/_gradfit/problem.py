"""Data container shared by every loss. Model: dX = A(mu - X) dt + sigma dW."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class Problem:
    snaps: object                 # (K, n, G) array, or a list of (n_k, G) arrays
                                  # (moment losses only; SW needs equal n)
    mu: jnp.ndarray               # (G,), constant over time; the INIT when fitted
    sigma: object                 # scalar or (G,) diffusion; the INIT when fitted
    dt: float                     # snapshot spacing
    floor: float | None           # expression floor used by the generator
    A_true: np.ndarray | None = None

    @property
    def n_genes(self) -> int:
        return int(np.shape(self.snaps[0])[-1])


def from_stationary_build(**build_kwargs) -> Problem:
    """Wrap _stationary_ground_truth.build(); kwargs pass straight through."""
    import _stationary_ground_truth as gt

    sim, snaps, _times, _mu_known, dt = gt.build(**build_kwargs)
    return Problem(
        snaps=jnp.asarray(np.stack(snaps)),
        mu=jnp.asarray(sim.mu0),
        sigma=gt.effective_sigma(sim),
        dt=float(dt),
        floor=gt.FLOOR,
        A_true=np.asarray(sim.A),
    )
