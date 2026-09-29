"""
One-step pushforward of a particle cloud under dX = A(mu - X) dt + sigma dW.

Both transitions take their noise as an argument instead of drawing it, so a
loss can hold the noise fixed across evaluations (common random numbers): the
objective is then a deterministic, differentiable function of A, which is what
lets L-BFGS work.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

JITTER = 1e-10


def ou_exact(A, sigma, dt):
    """
    Exact discretisation: X' - mu = F (X - mu) + N(0, Q) with F = expm(-A dt),
    Q = int_0^dt expm(-A s) sigma^2 expm(-A^T s) ds.

    Van Loan: for M = [[A, sigma^2 I], [0, -A^T]] dt, expm(M) = [[., B12], [0, B22]]
    gives F = B22^T and Q = F B12. One expm, differentiable in A.
    """
    G = A.shape[0]
    eye = jnp.eye(G)
    M = jnp.block([[A, sigma**2 * eye], [jnp.zeros((G, G)), -A.T]]) * dt
    E = expm(M)
    F = E[G:, G:].T
    Q = F @ E[:G, G:]
    return F, 0.5 * (Q + Q.T)


def ou_em(A, sigma, dt, n_sub):
    """
    Moment map of n_sub Euler-Maruyama substeps (floor ignored), same (F, Q)
    contract as ou_exact: with B = I - hA,
        F = B^n,   Q = sigma^2 h sum_{j<n} B^j B^jT.
    """
    G = A.shape[0]
    h = dt / n_sub
    B = jnp.eye(G) - h * A

    def body(carry, _):
        F, Q = carry
        return (B @ F, B @ Q @ B.T + sigma**2 * h * jnp.eye(G)), None

    (F, Q), _ = jax.lax.scan(body, (jnp.eye(G), jnp.zeros((G, G))), None, length=n_sub)
    return F, 0.5 * (Q + Q.T)


def moment_map(name, n_sub=5):
    """(A, sigma, dt) -> (F, Q) for the moment losses."""
    if name == "exact":
        return ou_exact
    if name == "em":
        return lambda A, sigma, dt: ou_em(A, sigma, dt, n_sub)
    raise ValueError(f"unknown transition {name!r}; choose 'exact' or 'em'")


class Exact:
    name = "exact"

    def noise_shape(self, K, n, G):
        return (K, n, G)

    def push(self, X, A, mu, sigma, dt, xi, floor):
        F, Q = ou_exact(A, sigma, dt)
        L = jnp.linalg.cholesky(Q + JITTER * jnp.eye(A.shape[0]))
        Y = mu + (X - mu) @ F.T + xi @ L.T
        return Y if floor is None else jnp.maximum(Y, floor)


class EulerMaruyama:
    """Ablation partner for Exact: discretisation error enters the loss."""

    name = "em"

    def __init__(self, n_sub=5):
        self.n_sub = n_sub

    def noise_shape(self, K, n, G):
        return (K, self.n_sub, n, G)

    def push(self, X, A, mu, sigma, dt, xi, floor):
        h = dt / self.n_sub

        def body(X, e):
            X = X + (mu - X) @ A.T * h + sigma * jnp.sqrt(h) * e
            return (X if floor is None else jnp.maximum(X, floor)), None

        X, _ = jax.lax.scan(body, X, xi)
        return X


def get(name, **kw):
    if name == "exact":
        return Exact()
    if name == "em":
        return EulerMaruyama(**kw)
    raise ValueError(f"unknown transition {name!r}; choose 'exact' or 'em'")
