"""
Ground-truth + synthetic snapshot helper for tests/algorithm/.

Reuses the datagen stationary simulator (NetworkSimulatorNonStationaryMu
with a constant mu and network_density=0.0, giving a diagonal A) for
(A, mu) -- same approach as tests/sde_fp/_stationary_destructive_data.py.
Duplicated locally rather than cross-imported: pytest's per-directory
import mode does not put sibling test directories on sys.path.

mu is expressed as the paper's per-interval {mu_k} (a list, one entry per
snapshot interval), even though every entry is the same vector here --
the stationary case is just a constant sequence. The Euler-Maruyama
stepper here is a minimal, independent implementation used only to build
fixture data (the "known-correct" generative process); it is not the
algorithm code under test.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.datagen.non_stationary_sim import NetworkSimulatorNonStationaryMu


def make_stationary_sim(num_genes: int = 2, seed: int = 42) -> NetworkSimulatorNonStationaryMu:
    """Diagonal-A (network_density=0), constant-mu OU population generator."""
    return NetworkSimulatorNonStationaryMu(
        num_genes=num_genes, network_density=0.0, seed=seed, mu_mode="constant",
    )


def draw_cross_section(sim: NetworkSimulatorNonStationaryMu, n_cells: int) -> np.ndarray:
    """n_cells independently simulated cells' state at t0 (T/dt kept minimal -- cheap)."""
    data, _ = sim.simulate(T=0.05, num_samples=n_cells, save_every=1, dt=0.05)
    return data[:, 0, :]


def _euler_maruyama_step(X, A, mu, sigma, dt, n_substeps, rng):
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        drift = (mu - X) @ A.T
        noise = sigma * np.sqrt(sub_dt) * rng.standard_normal(X.shape)
        X = np.maximum(X + drift * sub_dt + noise, 0.05)
    return X


def make_snapshots(
    sim: NetworkSimulatorNonStationaryMu,
    n_cells: int,
    n_snapshots: int,
    dt: float,
    *,
    n_substeps: int = 5,
    seed: int = 0,
) -> list[np.ndarray]:
    """
    Track the SAME n_cells particles forward n_snapshots-1 intervals under
    sim's own (A, mu0, sigma=sqrt(mean(diag(D)))) -- a same-cell-tracked
    sequence, matching OUGeneExpression's contract (paper lines 23-24:
    c_j(t_k) initialised from observed cell j).
    """
    rng = np.random.default_rng(seed)
    X = draw_cross_section(sim, n_cells)
    sigma = float(np.sqrt(np.diag(sim.D).mean()))
    snaps = [X]
    for _ in range(n_snapshots - 1):
        X = _euler_maruyama_step(X, sim.A, sim.mu0, sigma, dt, n_substeps, rng)
        snaps.append(X)
    return snaps
