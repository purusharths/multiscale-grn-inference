"""Data container shared by every loss. Model: dX = A(mu - X) dt + sigma dW."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class Problem:
    snaps: jnp.ndarray            # (K, n_cells, G), same cells count per snapshot
    mu: jnp.ndarray               # (G,), constant over time here
    sigma: float                  # isotropic diffusion scale, treated as known
    dt: float                     # snapshot spacing
    floor: float                  # expression floor used by the generator
    A_true: np.ndarray | None = None

    @property
    def n_genes(self) -> int:
        return int(self.snaps.shape[-1])


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
