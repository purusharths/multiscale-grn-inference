"""
Parameterizations of A. Each maps a pytree of unconstrained parameters to A
and back, so optimizers never see constraints.

  full     A itself, G^2 free entries.
  posdiag  diag(exp(logd)) + off-diagonal vector. Keeps self-decay positive
           (the generator draws it in [1, 1.8]) and has no dead entries, which
           matters for CMA / Nelder-Mead where every dimension costs samples.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class Full:
    name = "full"

    def init(self, A0):
        return jnp.asarray(A0)

    def to_A(self, p):
        return p


class PosDiag:
    name = "posdiag"

    def __init__(self, G):
        self.G = G
        self.rows, self.cols = np.where(~np.eye(G, dtype=bool))

    def init(self, A0):
        A0 = np.asarray(A0)
        return {"logd": jnp.log(jnp.clip(jnp.diag(A0), 1e-3, None)),
                "off": jnp.asarray(A0[self.rows, self.cols])}

    def to_A(self, p):
        A = jnp.diag(jnp.exp(p["logd"]))
        return A.at[self.rows, self.cols].set(p["off"])


class Model:
    """
    The full fitted parameter set. Wraps an A-parameterization and optionally
    also fits mu (free vector) and a per-gene sigma (via log sigma). What is not
    fitted is taken from the Problem inside the loss (losses.unpack).

    theta dicts (numpy or jax) have keys "A", "mu", "sigma".
    """

    def __init__(self, A_param, G, learn_mu=False, learn_sigma=False):
        self.A_param, self.G = A_param, G
        self.learn_mu, self.learn_sigma = learn_mu, learn_sigma

    def init(self, theta):
        p = {"A": self.A_param.init(theta["A"])}
        if self.learn_mu:
            p["mu"] = jnp.asarray(theta["mu"], dtype=float)
        if self.learn_sigma:
            s = jnp.broadcast_to(jnp.asarray(theta["sigma"], dtype=float), (self.G,))
            p["log_sigma"] = jnp.log(s)
        return p

    def to_theta(self, p):
        th = {"A": self.A_param.to_A(p["A"])}
        if "mu" in p:
            th["mu"] = p["mu"]
        if "log_sigma" in p:
            th["sigma"] = jnp.exp(p["log_sigma"])
        return th


def get(name, G):
    if name == "full":
        return Full()
    if name == "posdiag":
        return PosDiag(G)
    raise ValueError(f"unknown parameterization {name!r}; choose 'full' or 'posdiag'")


def initial_A(kind, G, rng, A_true=None, diag=1.0, off_scale=0.1):
    """
    diag    diag * I, no edges -- the neutral start.
    random  diag * I + N(0, off_scale) off-diagonal (used for extra multi-starts).
    truth   A_true, for the sanity check "does the optimizer walk away from the
            truth?" -- if it does, the loss minimum is not at A_true.
    """
    if kind == "diag":
        return diag * np.eye(G)
    if kind == "random":
        A = off_scale * rng.standard_normal((G, G))
        np.fill_diagonal(A, diag)
        return A
    if kind == "truth":
        if A_true is None:
            raise ValueError("init='truth' needs A_true")
        return np.array(A_true, dtype=float)
    raise ValueError(f"unknown init {kind!r}; choose 'diag', 'random' or 'truth'")
