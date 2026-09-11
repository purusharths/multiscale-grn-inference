"""
Exact closed-form OU transition (Van Loan's method), replacing Euler-
Maruyama entirely for the linear OU SDE dc = A(mu_k - c)dt + sigma dW.

Where this came from: ../loss-identifiability/check_loss_at_truth.py's
n_substeps sweep found that the true-vs-fitted loss gap flips sign and
stays flipped once the OU forward model is simulated with even 5
Euler-Maruyama substeps instead of the default 1 -- confirming single-step
discretization, not a fundamental non-identifiability, was why the loss's
minimum wasn't at the truth. A quick correctness check (see this folder's
README.md) showed the default single-step approximation isn't just biased
in the mean -- it also inflates the transition covariance by ~5x on a
representative test case. Since the OU SDE is linear, there's no need to
approximate it at any resolution: the transition is exactly Gaussian, in
closed form.

For dc = A(mu_k - c)dt + sigma dW (A general, sigma scalar/isotropic), the
transition over an interval dt is exactly:

    c(t+dt) | c(t) ~ Normal(mu_k + F(c(t) - mu_k), Sigma_dt)
    F        = expm(-A * dt)
    Sigma_dt = integral_0^dt expm(-A*s) (sigma^2 I) expm(-A^T*s) ds

Sigma_dt is computed via Van Loan's (1978) block-matrix trick: form
M = [[-A, sigma^2 I], [0, A^T]] (2G x 2G), compute expm(M*dt) = [[F, Gm],
[0, H]], then Sigma_dt = Gm @ F^T. This needs exactly one matrix
exponential per interval (shared across every particle/cell in that
interval, not one per particle), so it isn't meaningfully more expensive
than a handful of Euler-Maruyama substeps -- and has zero discretization
error.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.stats import gaussian_kde

from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta


def exact_ou_transition(
    X_tk: np.ndarray,
    theta: Theta,
    k: int,
    dt: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Drop-in replacement for ou_gene_expression -- same signature (minus
    n_substeps, which this makes moot), exact instead of approximate."""
    if rng is None:
        rng = np.random.default_rng()

    mu_k = theta.mu_at(k)
    A = theta.A
    G = A.shape[0]
    sigma = theta.sigma

    M = np.block([[-A, sigma**2 * np.eye(G)], [np.zeros((G, G)), A.T]])
    E = expm(M * dt)
    F = E[:G, :G]
    Gm = E[:G, G:]
    Sigma_dt = Gm @ F.T
    Sigma_dt = 0.5 * (Sigma_dt + Sigma_dt.T)  # symmetrize away float error

    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(Sigma_dt))):
        # Candidate A unstable enough that expm overflows -- treat like the
        # existing NaN-guard in _ours_combinations.py's _make_objective:
        # reject this candidate rather than crash.
        raise np.linalg.LinAlgError("non-finite exact OU transition (unstable candidate A)")

    mean = mu_k + (X_tk - mu_k) @ F.T
    noise = rng.multivariate_normal(np.zeros(G), Sigma_dt, size=X_tk.shape[0])
    return np.maximum(mean + noise, 0.0)


def exact_fp_cell_population(
    chi_tk: gaussian_kde,
    theta: Theta,
    k: int,
    dt: float,
    *,
    n_particles: int | None = None,
    rng: np.random.Generator | None = None,
) -> gaussian_kde:
    """Mirrors fp_cell_population.py exactly, but propagates particles with
    exact_ou_transition instead of ou_gene_expression."""
    if rng is None:
        rng = np.random.default_rng()
    n = n_particles or chi_tk.n
    seed_val = int(rng.integers(0, 2**31))
    particles = np.maximum(chi_tk.resample(n, seed=seed_val).T, 0.0)
    propagated = exact_ou_transition(particles, theta, k, dt, rng=rng)
    return preprocessing(propagated)
