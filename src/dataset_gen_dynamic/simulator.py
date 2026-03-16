"""Simulate mRNA expression levels using an Ornstein-Uhlenbeck process."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    theta: float = 2.0        # mean-reversion rate
    sigma: float = 0.5        # noise amplitude
    dt: float = 0.05          # time step


def _ou_step(
    x: np.ndarray,
    mu: np.ndarray,
    params: OUParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama step: dX = θ(μ - X)dt + σ dW."""
    dW = rng.standard_normal(x.shape)
    x_new = x + params.theta * (mu - x) * params.dt + params.sigma * np.sqrt(params.dt) * dW
    return np.maximum(x_new, 0.0)


def simulate(
    n_cells: int = 1000,
    n_genes: int = 50,
    n_states: int = 3,
    n_timepoints: int = 200,
    ou_params: OUParams | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate mRNA expression for n_cells x n_genes via OU process.

    Cells are sampled along a trajectory that transitions through n_states
    (e.g. differentiation). Each gene's expression follows an OU process
    around the current state's mean.

    Returns
    -------
    expression : DataFrame (n_cells × n_genes)
        Simulated mRNA counts (non-negative).
    metadata : DataFrame (n_cells × 3)
        Columns: time, state, pseudotime.
    trajectory : DataFrame (n_timepoints × n_genes)
        The underlying OU trajectory (hidden state, before cell sampling).
    """
    if ou_params is None:
        ou_params = OUParams()

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # State-specific mean expression profiles  (n_states × n_genes)
    # ---------------------------------------------------------------
    state_means = rng.uniform(0.5, 4.0, size=(n_states, n_genes))

    # Give each state a set of "marker" genes with elevated expression
    markers = n_genes // n_states
    for s in range(n_states):
        lo, hi = s * markers, (s + 1) * markers
        state_means[s, lo:hi] *= 4.0

    # ---------------------------------------------------------------
    # State schedule: equal-width blocks across timepoints
    # ---------------------------------------------------------------
    state_schedule = np.zeros(n_timepoints, dtype=int)
    for s in range(1, n_states):
        state_schedule[s * n_timepoints // n_states :] = s

    # ---------------------------------------------------------------
    # Simulate OU trajectory
    # ---------------------------------------------------------------
    traj = np.empty((n_timepoints, n_genes))
    x = state_means[0].copy()
    for t in range(n_timepoints):
        x = _ou_step(x, state_means[state_schedule[t]], ou_params, rng)
        traj[t] = x

    # ---------------------------------------------------------------
    # Sample cells at each timepoint  (cells_per_tp copies + noise)
    # ---------------------------------------------------------------
    cells_per_tp = max(1, n_cells // n_timepoints)
    total_cells = cells_per_tp * n_timepoints

    expressions = np.empty((total_cells, n_genes))
    times = np.empty(total_cells)
    states = np.empty(total_cells, dtype=int)

    for t in range(n_timepoints):
        idx = t * cells_per_tp
        noise = rng.standard_normal((cells_per_tp, n_genes)) * 0.2
        expressions[idx : idx + cells_per_tp] = np.maximum(traj[t] + noise, 0.0)
        times[idx : idx + cells_per_tp] = t * ou_params.dt
        states[idx : idx + cells_per_tp] = state_schedule[t]

    t_max = (n_timepoints - 1) * ou_params.dt
    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]

    expression_df = pd.DataFrame(expressions, columns=gene_names)

    metadata_df = pd.DataFrame(
        {
            "time": times,
            "state": [f"state_{s}" for s in states],
            "pseudotime": times / t_max,
        }
    )

    trajectory_df = pd.DataFrame(traj, columns=gene_names)
    trajectory_df["time"] = np.arange(n_timepoints) * ou_params.dt
    trajectory_df["state"] = [f"state_{s}" for s in state_schedule]

    return expression_df, metadata_df, trajectory_df
