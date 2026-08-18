"""
Intervention (knockout) dataset fixture, built on the datagen simulator.

Unlike _ground_truth.py -- which uses a constant mu and drives its own
minimal Euler-Maruyama rollout -- this generates data through
datagen.non_stationary_sim.NetworkSimulatorNonStationaryMu.simulate()
itself, under mu_mode="knockout":

    mu(t) = mu0 - mu0 * sigmoid(k * (t - t_star))     [mu_options.mu_inverse_sigmoid]

i.e. cells sit at baseline mu0, then an intervention at t_star drives the
expression target to ~0. Both the knockout and the constant-mu control are
produced through the same simulator code path here, so comparing them
isolates the effect of the intervention rather than the generator.

This exercises the per-interval {mu_k} that Theta carries and Algorithm 1
lists as input ("Data: Known Intervention means {mu_k}", paper line 2):
mu_per_interval below is exactly that sequence, and it is genuinely
time-varying here, unlike the stationary case where every entry is equal.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.datagen.non_stationary_sim import NetworkSimulatorNonStationaryMu

# Simulator integration settings (dense grid; snapshots are subsampled from it).
# NOTE: NetworkSimulatorNonStationaryMu.simulate() has an off-by-one that leaves
# the final saved slot all-zero whenever int(T/dt) is an exact multiple of
# save_every. T=4.0/dt=0.005/save_every=15 -> steps=800, not a multiple, so it is
# not triggered; extract_snapshots() asserts this rather than trusting it.
SIM_T = 4.0
SIM_DT = 0.005
SAVE_EVERY = 15


def make_knockout_sim(num_genes: int = 4, seed: int = 42, t_star: float | None = None, k: float = 20.0):
    """Diagonal-A simulator whose mu(t) collapses mu0 -> ~0 at t_star (default T/2)."""
    return NetworkSimulatorNonStationaryMu(
        num_genes=num_genes, network_density=0.0, seed=seed,
        mu_mode="knockout", mu_kwargs={"t_star": t_star, "k": k},
    )


def make_constant_sim(num_genes: int = 4, seed: int = 42):
    """Stationary control: same A/mu0/D, but mu(t) == mu0 for all t."""
    return NetworkSimulatorNonStationaryMu(
        num_genes=num_genes, network_density=0.0, seed=seed, mu_mode="constant",
    )


def extract_snapshots(
    sim, n_cells: int, n_snapshots: int, *, T: float = SIM_T,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], float]:
    """
    Run sim.simulate() on the dense grid, then subsample n_snapshots evenly
    spaced timepoints from it.

    Returns (snapshots, times, mu_per_interval, dt) where
      snapshots[k]       -- (n_cells, G) cross-section at times[k]
      mu_per_interval[k] -- mu(times[k]), the known intervention mean for
                            interval [t_k, t_{k+1}] (length n_snapshots-1)
      dt                 -- spacing between consecutive snapshot times
    """
    data, time_grid = sim.simulate(
        T=T, num_samples=n_cells, save_every=SAVE_EVERY, dt=SIM_DT,
    )

    idx = np.linspace(0, len(time_grid) - 1, n_snapshots).astype(int)
    snapshots = [data[:, i, :] for i in idx]
    times = time_grid[idx]

    # Guard against the simulate() off-by-one described above.
    for k, snap in enumerate(snapshots):
        assert not np.allclose(snap, 0.0), (
            f"snapshot {k} (t={times[k]:.3f}) is all-zero -- simulate()'s "
            f"off-by-one was triggered; adjust T/SIM_DT/SAVE_EVERY so that "
            f"int(T/dt) is not a multiple of save_every"
        )

    mu_per_interval = [sim.mu_t(float(t), T) for t in times[:-1]]
    dt = float(times[1] - times[0])
    return snapshots, times, mu_per_interval, dt


def effective_sigma(sim) -> float:
    """Scalar isotropic sigma matching the simulator's per-gene diffusion D."""
    return float(np.sqrt(np.diag(sim.D).mean()))
