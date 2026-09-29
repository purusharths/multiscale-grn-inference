"""
Non-interventional (stationary-mu) dataset: a coupled GRN observed relaxing,
with no perturbation applied at any point.

Difference from ../interventions/_intervention_ground_truth.py: mu(t) = mu0 for
all t. There is no knockout, so the ONLY thing driving the population is the
network itself. That is the point of testing here -- it removes the large
intervention transient, which on the knockout data dominated the loss and may
have been drowning out the much smaller edge signal.

Difference from ../../algorithm/_ground_truth.py: that helper builds its
stationary sim with network_density=0.0, i.e. a purely diagonal A with no edges
at all. Fine for testing the OU machinery, useless for GRN recovery -- there is
nothing to recover. Here density > 0, so A has real off-diagonal structure.

THE PERTURBED START IS NOT OPTIONAL. sim.simulate() initialises cells at
mu + small noise, i.e. already at equilibrium. With constant mu they then stay
there, the drift A(mu - c) is ~0 regardless of what A is, and true and wrong
dynamics produce identical snapshots -- every fit looks equally good and the
comparison measures nothing. Starting the population away from mu (shift below)
gives the network something to act on. This is the same trap documented in
../../algorithm/_ground_truth.py's make_snapshots(shift=...).
"""
from __future__ import annotations

import os

import numpy as np

from multsc_grn_inference.datagen.non_stationary_sim import NetworkSimulatorNonStationaryMu

# How the population is started away from equilibrium.
#
#   "uniform"       every cell displaced the SAME way (-amp * mu0). Produces the
#                   big common ramp: all genes rise together from ~0.4*mu to mu.
#   "heterogeneous" each cell displaced in its OWN random direction, N(0, amp*mu0).
#                   The population MEAN barely moves (random directions cancel)
#                   while the COVARIANCE is far from stationary -- so the mean
#                   expression curves stay flat but the data is still
#                   non-stationary, and A stays identifiable.
#
# The choice is a real tradeoff, not a preference. Measured on the 8-gene
# dataset, edge signal = L(no edges) - L(true A), noise = seed-to-seed sd of
# L(true A), exact transition, 200 projections:
#
#   perturbation     amp   max|mean-mu|    signal   noise sd     S/N   ratio
#   uniform          0.6           2.03   0.09961    0.00023   426.7  11.66x
#   heterogeneous    0.6           0.11   0.01337    0.00167     8.0   3.01x
#   heterogeneous    1.0           0.33   0.02736    0.00197    13.9   3.77x
#   heterogeneous    1.5           0.74   0.05771    0.00217    26.7   4.74x
#   heterogeneous    2.5           1.85   0.16972    0.00259    65.6   6.02x
#
# Uniform wins on signal-to-noise by a wide margin, because the common ramp is
# large and highly reproducible. Heterogeneous buys a flat mean at roughly a
# 30x cost in S/N at matched amplitude -- though since random directions cancel
# regardless of size, amplitude can be raised to claw much of it back without
# the mean moving nearly as much.
#
# WHY NOT JUST START AT EQUILIBRIUM: because then A is not identifiable at all.
# At stationarity the data constrains only the Lyapunov equation
# A*Sigma + Sigma*A^T = sigma^2*I, whose solutions are A = (sigma^2/2 I + S)
# Sigma^-1 for ANY skew-symmetric S -- at 8 genes that is 28 free parameters,
# exactly the edge structure. Verified numerically: skew-perturbed A' matrices
# reproduce Sigma to 1e-16 while sharing only 3-4 of the 16 true edges and
# correlating ~0.0 with the truth. Some departure from stationarity is
# mandatory; the only question is which kind.
PERTURBATION = os.environ.get("GRN_PERTURBATION", "uniform")
SHIFT_FRACTION = float(os.environ.get("GRN_PERTURB_AMP", 0.6))

# Euler-Maruyama substeps per snapshot interval in the GENERATOR. 5 is the
# historical default, but at dt=0.5 that is h=0.1 against decay rates up to ~2,
# coarse enough that the data are measurably not samples of the true SDE: an
# exact-OU fit scores BETTER than A_true on its own loss (gradfit run
# 20260928-152916). Raise it (e.g. 50) for data that match the continuous model.
GEN_SUBSTEPS = int(os.environ.get("GRN_GEN_SUBSTEPS", 5))

# Floor applied by the generative stepper. Matches _ground_truth.py (0.05, not
# 0.0) so a gene's expression never sits exactly at zero -- a fully-zero column
# makes gaussian_kde's covariance singular.
FLOOR = 0.05


def make_stationary_network_sim(num_genes=8, seed=42, network_density=0.3):
    """Constant-mu simulator WITH off-diagonal coupling (unlike the density=0
    helper in tests/algorithm/)."""
    return NetworkSimulatorNonStationaryMu(
        num_genes=num_genes, network_density=network_density, seed=seed,
        mu_mode="constant",
    )


def effective_sigma(sim) -> float:
    """Scalar isotropic sigma matching the simulator's per-gene diffusion D."""
    return float(np.sqrt(np.diag(sim.D).mean()))


def _em_step(X, A, mu, sigma, dt, n_substeps, rng):
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        drift = (mu - X) @ A.T
        noise = sigma * np.sqrt(sub_dt) * rng.standard_normal(X.shape)
        X = np.maximum(X + drift * sub_dt + noise, FLOOR)
    return X


def perturb(X0, mu0, rng, *, mode=None, amp=None):
    """Displace the starting population away from equilibrium. See PERTURBATION."""
    mode = mode or PERTURBATION
    amp = SHIFT_FRACTION if amp is None else amp
    if mode == "uniform":
        return np.maximum(X0 - amp * mu0, FLOOR)
    if mode == "heterogeneous":
        return np.maximum(X0 + rng.standard_normal(X0.shape) * amp * mu0, FLOOR)
    raise ValueError(f"unknown GRN_PERTURBATION {mode!r}; "
                     f"choose 'uniform' or 'heterogeneous'")


def make_snapshots(sim, n_cells, n_snapshots, dt, *, n_substeps=None, seed=0,
                   shift_fraction=SHIFT_FRACTION):
    """
    Track n_cells particles forward from a perturbed start.

    n_substeps in the GENERATIVE process is separate from the fitting model's
    own discretization: the data should be a good sample of the true SDE,
    whatever resolution the model being fitted happens to use. Defaults to
    GEN_SUBSTEPS (see the note there on why 5 is not enough).
    """
    n_substeps = GEN_SUBSTEPS if n_substeps is None else n_substeps
    rng = np.random.default_rng(seed)
    data, _ = sim.simulate(T=0.05, num_samples=n_cells, save_every=1, dt=0.05)
    X = perturb(data[:, 0, :], sim.mu0, rng, amp=shift_fraction)

    sigma = effective_sigma(sim)
    snaps = [X]
    for _ in range(n_snapshots - 1):
        X = _em_step(X, sim.A, sim.mu0, sigma, dt, n_substeps, rng)
        snaps.append(X)
    return snaps


def build(n_genes=8, n_cells=3000, n_snaps=15, dt=0.5, seed=42, density=0.3):
    """Everything a comparison run needs. mu is the paper's per-interval
    {mu_k}: a constant sequence here, one entry per interval."""
    sim = make_stationary_network_sim(n_genes, seed, density)
    snaps = make_snapshots(sim, n_cells, n_snaps, dt, seed=seed)
    mu_known = [sim.mu0] * (n_snaps - 1)
    times = np.arange(n_snaps) * dt
    return sim, snaps, times, mu_known, dt
