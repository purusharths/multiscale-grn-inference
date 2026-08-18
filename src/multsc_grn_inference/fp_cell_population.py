"""
Algorithm 1, function FPCellPopulation (paper lines 30-38):

    function FPCellPopulation(chi_tk; theta)
        # Macro scale: evolve population density forward via JKO scheme
        # d/dt p = div[A(c - mu_k) p] + sigma^2 * Delta p
        nu* <- argmin_{nu in P_2(R^G)} [W2^2(nu, chi_tk) + tau * F[nu; theta]]
        return nu*

    where F[nu; theta] = int nu log(nu) dc + int nu(c) Psi[c; theta] dc
          Psi[c; theta] = (1/2)(c - mu_k)^T A (c - mu_k)   [OU potential]

Implementation: particle JKO, not JKO-ICNN.

For a QUADRATIC potential Psi[c;theta] -- i.e. exactly the OU potential
theta already specifies, not an unknown/learned one -- the JKO minimiser
nu* coincides, in the many-particle limit, with the distribution of
X(t_{k+1}) for dX = A(mu_k - X)dt + sigma dW_t started from X(t_k) ~ chi_tk.
This is the classical particle <-> Fokker-Planck correspondence for Ito
diffusions [Risken 1996; Oksendal 2003], combined with the JKO
time-discretisation result that iterating the proximal step above
converges to the Fokker-Planck solution as the step size -> 0
[Jordan, Kinderlehrer & Otto, 1998, SIAM J. Math. Anal. 29(1):1-17].

So the solver here is: resample particles from chi_tk (no per-cell
continuity required -- valid for destructive/cross-sectional measurement)
and propagate them one interval under the SAME Euler-Maruyama step
OUGeneExpression uses, refitting a density (Preprocessing) on the result.
No ICNN, no inner optimisation loop -- theta already parametrises the
potential, so there's nothing to learn at this step. The JKO step size tau
is taken equal to dt (one pass over the interval, not multiple proximal
substeps); n_substeps controls the SDE's own internal discretisation.

Validity without longitudinal (trajectory) data
-------------------------------------------------
Cross-sectional-only snapshots identify this parametric linear-Gaussian
(OU) model reasonably well precisely because the potential is restricted
to a low-dimensional parametric family (A, mu_k): a handful of marginal
snapshots constrain (A, mu_k, sigma) via moment matching, without ever
pairing a cell with its own future self (see the deprecated
moment-of-moments initialiser, mom_init.py, which relied on exactly this).
Waddington-OT [Schiebinger et al., 2019, Cell 176(4):928-943] and JKONet
[Bunne, Meng-Papaxanthos, Krause & Cuturi, 2022, AISTATS] exploit the same
principle to infer population dynamics from cross-sectional snapshots alone.

Caveat: without any parametric restriction, marginals-only data cannot
uniquely determine a coupling/dynamics -- many different SDEs can produce
the same sequence of one-time marginals [Lavenant, Zhang, Kim &
Schiebinger, "Towards a mathematical theory of trajectory inference",
arXiv:2102.09204; Chizat, Zhang, Heitz & Schiebinger, "Trajectory
Inference via Mean-field Langevin in Path Space", NeurIPS 2022]. What
resolves the ambiguity here is the OU parametric assumption itself, not
the availability of trajectories.

Where this breaks
--------------------
The OU/quadratic-potential assumption is single-well: it cannot represent
a genuinely multimodal or branching landscape (multiple attractors /
cell-fate decisions -- the actual "Waddington landscape" this algorithm's
macro scale is named for). If the true dynamics are non-quadratic (e.g. a
double-well potential at a bifurcation), no linear theta fits them, and
cross-sectional-only data gives no trajectory-level signal to detect the
mismatch (L_OU can't even be computed without tracked cells; L_FP alone
only tells you the best-fit OU model, not whether OU is *right*). At that
point Psi[c;theta] must become a flexible, learned potential rather than a
fixed quadratic form -- exactly what JKO-ICNN / JKONet's neural-potential
parameterisation is for. The particle-JKO shortcut here stops applying
because there's no longer a closed-form (A, mu_k) to simulate forward --
the potential itself has to be fit as part of solving the JKO step.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from multsc_grn_inference.ou_gene_expression import ou_gene_expression
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta


def fp_cell_population(
    chi_tk: gaussian_kde,
    theta: Theta,
    k: int,
    dt: float,
    *,
    n_particles: int | None = None,
    n_substeps: int = 1,
    rng: np.random.Generator | None = None,
) -> gaussian_kde:
    """
    nu* <- particle-JKO-step(chi_tk; A, mu_k, sigma, dt). See module docstring.

    n_particles defaults to chi_tk's own fitted sample count (chi_tk.n).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = n_particles or chi_tk.n
    seed_val = int(rng.integers(0, 2**31))
    particles = np.maximum(chi_tk.resample(n, seed=seed_val).T, 0.0)

    propagated = ou_gene_expression(particles, theta, k, dt, n_substeps=n_substeps, rng=rng)
    return preprocessing(propagated)
