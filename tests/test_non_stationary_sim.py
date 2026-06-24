import numpy as np
import pytest
from dataset_gen_dynamic.non_stationary_sim import NetworkSimulatorNonStationaryMu


def is_diagonally_dominant(A):
    # true if A is strictly row diagonally dominant
    for i in range(A.shape[0]):
        off_sum = np.sum(np.abs(A[i])) - A[i, i]
        if A[i, i] <= off_sum:
            return False
    return True


def test_diagonal_dominance_default():
    sim = NetworkSimulatorNonStationaryMu(num_genes=10, network_density=0.3, seed=0)
    assert is_diagonally_dominant(sim.A), "A is not diagonally dominant after enforcement"


def test_diagonal_dominance_high_density():
    sim = NetworkSimulatorNonStationaryMu(num_genes=10, network_density=0.8, seed=42)
    assert is_diagonally_dominant(sim.A)


def test_diagonal_dominance_various_seeds():
    for seed in range(20):
        sim = NetworkSimulatorNonStationaryMu(num_genes=15, network_density=0.5, seed=seed)
        assert is_diagonally_dominant(sim.A), f"A not diagonally dominant for seed={seed}"


def test_eigenvalues_positive_real_parts():
    sim = NetworkSimulatorNonStationaryMu(num_genes=10, network_density=0.6, seed=7)
    eigs = np.linalg.eigvals(sim.A)
    assert np.all(eigs.real > 0), f"Some eigenvalues have non-positive real parts: {eigs.real}"


def test_diagonal_entries_unchanged():
    sim = NetworkSimulatorNonStationaryMu(num_genes=10, network_density=0.3, seed=1)
    for i in range(sim.num_genes):
        assert sim.A[i, i] > 0, f"Diagonal entry A[{i},{i}] should remain positive     "

def test_heaviside_no_nan():
    delta = np.ones(10) * 5.0 
    sim = NetworkSimulatorNonStationaryMu(
        mu_mode="heaviside", mu_kwargs={"delta": delta, "t_star": 1.0}
    )
    data, _ = sim.simulate(T=4.0, num_samples=50)
    assert not np.any(np.isnan(data))
    assert np.all(data < 1e4)
