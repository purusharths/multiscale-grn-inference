"""
Losses over the snapshot sequence. Every builder returns f(theta, key) -> scalar,
where theta is either A alone (mu and sigma then come from the Problem) or a
dict {"A", optionally "mu", "sigma"} when those are being fitted too.

  sw     particle loss: push snapshot k one step, sliced-W2^2 against snapshot
         k+1. Stochastic in the projections and the transition noise; with
         resample=False both come from a fixed seed and `key` is ignored.
  kl     Gaussian-moment loss: predicted (mean, cov) of snapshot k+1 from the
         EMPIRICAL moments of snapshot k, KL(empirical || predicted). Exact for
         the linear-Gaussian model, deterministic, no particles.
  bures  same moments, Bures-Wasserstein W2^2 instead of KL.

Every loss takes transition="exact" | "em" (with transition_kw={"n_sub": n}),
so the fit's discretisation can be matched to, or ablated against, the
generator's. Moment losses take rollout=True to propagate the predicted moments from
snapshot 0 through the whole sequence instead of restarting from the data at
each interval (multi-step error, more weight on the slow mean ramp).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import transition as tr


def unpack(theta, problem):
    if isinstance(theta, dict):
        return (theta["A"], theta.get("mu", problem.mu),
                theta.get("sigma", problem.sigma))
    return theta, problem.mu, problem.sigma


def _unit_directions(key, n_proj, G):
    th = jax.random.normal(key, (n_proj, G))
    return th / jnp.linalg.norm(th, axis=1, keepdims=True)


def make_sw(problem, *, transition="exact", n_proj=200, resample=False,
            floor=True, seed=0, transition_kw=None):
    trans = tr.get(transition, **(transition_kw or {}))
    snaps = jnp.asarray(problem.snaps)   # SW needs equal cell counts per snapshot
    X0, X1 = snaps[:-1], snaps[1:]
    K, n, G = X0.shape
    dt = problem.dt
    fl = problem.floor if floor else None
    fixed = jax.random.PRNGKey(seed)

    def loss(theta, key=None):
        A, mu, sigma = unpack(theta, problem)
        k = key if resample else fixed
        k_proj, k_noise = jax.random.split(k)
        theta = _unit_directions(k_proj, n_proj, G)
        xi = jax.random.normal(k_noise, trans.noise_shape(K, n, G))
        pred = jax.vmap(lambda X, e: trans.push(X, A, mu, sigma, dt, e, fl))(X0, xi)
        a = jnp.sort(pred @ theta.T, axis=1)          # (K, n, P)
        b = jnp.sort(X1 @ theta.T, axis=1)
        return jnp.mean((a - b) ** 2)

    return loss


def _empirical_moments(snaps, shrinkage=None):
    """
    Per-snapshot mean and covariance. Snapshots may differ in cell count.

    shrinkage: None (sample covariance), "lw" (Ledoit-Wolf), or a float a in
    [0, 1] shrinking towards the diagonal: (1-a) S + a diag(S). Needed when a
    snapshot has fewer cells than genes -- the sample covariance is then
    singular and the KL's log-determinant is -inf.
    """
    ms, Ss = [], []
    for X in snaps:
        X = np.asarray(X, dtype=float)
        ms.append(X.mean(axis=0))
        if shrinkage == "lw":
            from sklearn.covariance import ledoit_wolf
            S = ledoit_wolf(X)[0]
        else:
            S = np.cov(X, rowvar=False)
            if shrinkage:
                S = (1 - shrinkage) * S + shrinkage * np.diag(np.diag(S))
        Ss.append(np.atleast_2d(S))
    return np.stack(ms), np.stack(Ss)


def _sqrtm_psd(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(V, -1, -2)


def _predicted_moments(theta, problem, m, S, rollout, fq=tr.ou_exact):
    A, mu, sigma = unpack(theta, problem)
    F, Q = fq(A, sigma, problem.dt)

    def one(mk, Sk):
        return mu + F @ (mk - mu), F @ Sk @ F.T + Q

    if not rollout:
        return jax.vmap(one)(m[:-1], S[:-1])

    def body(carry, _):
        nxt = one(*carry)
        return nxt, nxt

    _, (mp, Sp) = jax.lax.scan(body, (m[0], S[0]), None, length=m.shape[0] - 1)
    return mp, Sp


def make_kl(problem, *, rollout=False, transition="exact", transition_kw=None,
            shrinkage=None, **_):
    fq = tr.moment_map(transition, **(transition_kw or {}))
    m_np, S_np = _empirical_moments(problem.snaps, shrinkage)
    m, S = jnp.asarray(m_np), jnp.asarray(S_np)
    m1, S1 = m[1:], S[1:]
    logdet1 = jnp.asarray(np.linalg.slogdet(S_np[1:])[1])
    G = problem.n_genes

    def kl_one(mp, Sp, mo, So, ld_o):
        L = jnp.linalg.cholesky(Sp)
        d = mo - mp
        tr_term = jnp.trace(jax.scipy.linalg.cho_solve((L, True), So))
        quad = d @ jax.scipy.linalg.cho_solve((L, True), d)
        logdet_p = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return 0.5 * (tr_term + quad - G + logdet_p - ld_o)

    def loss(theta, key=None):
        mp, Sp = _predicted_moments(theta, problem, m, S, rollout, fq)
        return jnp.mean(jax.vmap(kl_one)(mp, Sp, m1, S1, logdet1))

    return loss


def make_bures(problem, *, rollout=False, transition="exact", transition_kw=None,
               shrinkage=None, **_):
    fq = tr.moment_map(transition, **(transition_kw or {}))
    m_np, S_np = _empirical_moments(problem.snaps, shrinkage)
    m, S = jnp.asarray(m_np), jnp.asarray(S_np)
    m1, S1 = m[1:], S[1:]
    # sqrt of the DATA covariance, precomputed: the cross term
    # tr((S1^1/2 Sp S1^1/2)^1/2) then needs only eigvalsh of a symmetric matrix,
    # which differentiates cleanly in Sp.
    S1h = jnp.asarray(_sqrtm_psd(S_np[1:]))

    def loss(theta, key=None):
        mp, Sp = _predicted_moments(theta, problem, m, S, rollout, fq)
        C = S1h @ Sp @ S1h
        cross = jnp.sum(jnp.sqrt(jnp.clip(jnp.linalg.eigvalsh(C), 1e-12, None)), axis=-1)
        w2 = (jnp.sum((m1 - mp) ** 2, axis=-1)
              + jnp.trace(Sp, axis1=-2, axis2=-1) + jnp.trace(S1, axis1=-2, axis2=-1)
              - 2.0 * cross)
        return jnp.mean(w2)

    return loss


BUILDERS = {"sw": make_sw, "kl": make_kl, "bures": make_bures}


def build(problem, name, **kw):
    if name not in BUILDERS:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[name](problem, **kw)
