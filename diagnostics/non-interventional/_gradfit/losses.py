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
  oufp   the paper's Algorithm 1 terms (src/.../compute_loss.py), any subset
         of {"ou", "fp", "cons"}, summed over intervals, with gradients:
           ou    SW(Phi_OU(X_k), X_{k+1})       observed cells pushed forward
           fp    SW(Phi_OU(KDE(X_k)), X_{k+1})  KDE-resampled cells pushed
           cons  SW(Phi_OU(X_k), Phi_OU(KDE(X_k)))
         Snapshots may differ in size (quantile interpolation, as in
         compute_loss._sliced_w2).

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


def _quantile_gather(n_from, n_to):
    """(i0, i1, w) so that sorted[i0]*(1-w) + sorted[i1]*w is sorted's linear
    interpolation at n_to evenly spaced quantile levels -- compute_loss.
    _sliced_w2's scheme, precomputed because sizes are known up front."""
    pos = np.linspace(0.0, 1.0, n_to) * (n_from - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n_from - 1)
    return i0, i1, (pos - i0)[:, None]


def _at_quantiles(sorted_, g):
    if g is None:
        return sorted_
    i0, i1, w = g
    return sorted_[i0] * (1 - w) + sorted_[i1] * w


def _sw_setup(n_pred, n_other):
    n = min(n_pred, n_other)
    return (n,
            None if n_pred == n else _quantile_gather(n_pred, n),
            None if n_other == n else _quantile_gather(n_other, n))


def kde_resample(X, rng, n=None):
    """scipy.stats.gaussian_kde(X.T).resample(n): pick cells uniformly, add
    N(0, h^2 Cov) with Scott's h = n^(-1/(G+4)). Independent of theta, so it is
    drawn once and the FP term stays differentiable. Note h -> 1 as G grows
    (0.93 at G=100, n=2000): the resampled cloud then carries ~(1+h^2) x the
    data covariance -- the bias compute_loss.py's docstring describes."""
    X = np.asarray(X, dtype=float)
    N, G = X.shape
    n = n or N
    h = N ** (-1.0 / (G + 4))
    C = np.atleast_2d(np.cov(X, rowvar=False)) * h**2
    L = np.linalg.cholesky(C + 1e-9 * np.eye(G))
    return X[rng.integers(0, N, n)] + rng.standard_normal((n, G)) @ L.T


def make_oufp(problem, *, terms=("ou",), n_proj=200, floor=True, seed=0,
              transition="exact", transition_kw=None, **_):
    terms = tuple(terms)
    bad = set(terms) - {"ou", "fp", "cons"}
    if bad or not terms:
        raise ValueError(f"terms must be a non-empty subset of ou/fp/cons, got {terms}")
    trans = tr.get(transition, **(transition_kw or {}))
    snaps_np = [np.asarray(x, dtype=float) for x in problem.snaps]
    snaps = [jnp.asarray(x) for x in snaps_np]
    G = problem.n_genes
    fl = problem.floor if floor else None
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, G))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # everything random is drawn ONCE (common random numbers): KDE particles
    # for the FP arm and the OU noise for both arms
    kde = []
    for x in snaps_np[:-1]:
        r = kde_resample(x, rng)
        kde.append(jnp.asarray(np.maximum(r, fl) if fl is not None else r))
    xi_ou = [jnp.asarray(rng.standard_normal(trans.noise_shape(1, x.shape[0], G))[0])
             for x in snaps_np[:-1]]
    xi_fp = [jnp.asarray(rng.standard_normal(trans.noise_shape(1, x.shape[0], G))[0])
             for x in snaps_np[:-1]]
    # The observed side of every OU/FP comparison does not depend on theta:
    # sort and interpolate it here, in numpy. Left inside the traced loss, XLA
    # constant-folds those sorts at compile time (~1.5 min EACH at 2000 x 300).
    K = len(snaps_np) - 1
    to_next = []
    for k in range(K):
        n, g_pred, g_obs = _sw_setup(snaps_np[k].shape[0], snaps_np[k + 1].shape[0])
        obs = _at_quantiles(np.sort(snaps_np[k + 1] @ dirs.T, axis=0), g_obs)
        to_next.append((g_pred, jnp.asarray(obs)))
    dirs = jnp.asarray(dirs)

    def sw_obs(pred, k):
        g, obs = to_next[k]
        return jnp.mean((_at_quantiles(jnp.sort(pred @ dirs.T, axis=0), g) - obs) ** 2)

    def sw_pred(a, b):   # cons: both clouds depend on theta, same size
        return jnp.mean((jnp.sort(a @ dirs.T, axis=0) - jnp.sort(b @ dirs.T, axis=0)) ** 2)

    def loss(theta, key=None):
        A, mu, sigma = unpack(theta, problem)
        dt = problem.dt
        total = 0.0
        for k in range(K):
            ou = trans.push(snaps[k], A, mu, sigma, dt, xi_ou[k], fl) \
                if ("ou" in terms or "cons" in terms) else None
            fp = trans.push(kde[k], A, mu, sigma, dt, xi_fp[k], fl) \
                if ("fp" in terms or "cons" in terms) else None
            if "ou" in terms:
                total += sw_obs(ou, k)
            if "fp" in terms:
                total += sw_obs(fp, k)
            if "cons" in terms:
                total += sw_pred(ou, fp)
        return total

    return loss


BUILDERS = {"sw": make_sw, "kl": make_kl, "bures": make_bures, "oufp": make_oufp}


def build(problem, name, **kw):
    if name not in BUILDERS:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[name](problem, **kw)
