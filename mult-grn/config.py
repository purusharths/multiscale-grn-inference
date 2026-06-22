"""
All hyperparameters for the multiscale GRN inference pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRNConfig:
    csv_dir: str = ""          # set by main.py to Path(__file__).parent / "csv"
    tau: float = 0.1            # JKO step size
    dt: float = 0.01            # Euler-Maruyama step size
    n_em_steps: int = 20        # Euler-Maruyama steps per interval
    n_jko_steps: int = 10       # inner JKO particle gradient iterations
    n_outer_steps: int = 500    # outer optimisation iterations
    lr: float = 1e-3            # outer optimiser learning rate
    jko_lr: float = 1e-2        # inner JKO particle learning rate
    w2_backend: str = "sinkhorn"  # "sinkhorn" | "exact"
    sinkhorn_blur: float = 0.05   # Sinkhorn regularisation (smaller → truer W2)
    n_particles: int = 300      # particles subsampled per timepoint for OT
    bip_lambda: float = 1e-2    # ridge regularisation for BIP warm start
    bip_delta_t: float = 1.0    # effective time step between consecutive snapshots
    device: str = "cpu"
    seed: int = 42
    # TODO: extend to diagonal D = diag(sigma_1^2, ..., sigma_G^2)
