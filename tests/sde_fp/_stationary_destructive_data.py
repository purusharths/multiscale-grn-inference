"""
Shared ground-truth + cross-sectional data source for tests/sde_fp/*.

Ground truth (A, mu) and every t0 particle cloud in this test package come
from the datagen module's simulator:

    NetworkSimulatorNonStationaryMu(mu_mode="constant", network_density=0.0)

This is the maintained stand-in for datagen.stationary_sim.NetworkSimulator,
which is currently unusable: its __init__ is misspelled `lebann__init__`,
so it is never invoked and the class has no attributes after construction.
NetworkSimulatorNonStationaryMu with a constant mu is already used this way
elsewhere (bayes_grn_non_stationary.py, loss_diagnostic.py); network_density
=0.0 additionally forces zero off-diagonal edges, giving the diagonal A
these tests need for closed-form OU verification.

`draw_destructive_t0_cross_section` mimics a single destructive-measurement
snapshot (see datagen/run_destructive_measurements.py's
`_sim_to_dataframes_destructive`): every returned cell is an independently
simulated trajectory sampled once, never reused across snapshots or tests.

Note: the simulator's own diffusion `D` (one value per gene) is a different
noise model than loss.py's `_ou_euler_maruyama`, which takes a single scalar
sigma applied isotropically across genes. Callers needing a ground-truth
sigma for `_ou_euler_maruyama` define their own scalar constant rather than
reading `sim.D`.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.datagen.non_stationary_sim import NetworkSimulatorNonStationaryMu


def make_stationary_sim(num_genes: int = 2, seed: int = 42) -> NetworkSimulatorNonStationaryMu:
    """Diagonal-A (network_density=0), constant-mu OU population generator."""
    return NetworkSimulatorNonStationaryMu(
        num_genes=num_genes, network_density=0.0, seed=seed, mu_mode="constant",
    )


def draw_destructive_t0_cross_section(
    sim: NetworkSimulatorNonStationaryMu, n_cells: int,
) -> np.ndarray:
    """
    A destructive-measurement-style snapshot at t0: `n_cells` independently
    simulated cells, each observed exactly once, then discarded.

    T/dt are kept minimal since only the t0 state (recorded before the
    simulation loop advances) is used -- this keeps the call cheap even for
    large n_cells despite the simulator's per-sample Python loop.
    """
    data, _ = sim.simulate(T=0.05, num_samples=n_cells, save_every=1, dt=0.05)
    return data[:, 0, :]


def draw_perturbed_cross_section(
    sim: NetworkSimulatorNonStationaryMu, n_cells: int, shift: np.ndarray,
) -> np.ndarray:
    """
    Same destructive draw as `draw_destructive_t0_cross_section`, recentred
    by `shift`.

    The simulator always initialises each cell at mu + small noise (see
    NetworkSimulatorNonStationaryMu.simulate), so a plain t0 draw carries no
    signal for identifying dynamics parameters (X ~ mu regardless of A).
    Tests that need visibly-nonzero drift -- discriminating true vs. wrong
    parameters, or the mean-displacement regression `mom_estimate` performs
    -- should use this instead, with e.g. shift=-0.6*mu to mimic a population
    sampled shortly after a perturbation (consistent with the destructive/
    gene-perturbation datasets in datagen/run_destructive_measurements.py
    and datagen/run_gene_perturb_major.py, which start from an off-equilibrium
    population and observe it relax over subsequent collection timepoints).
    """
    X = draw_destructive_t0_cross_section(sim, n_cells)
    return np.maximum(X + shift, 0.05)
