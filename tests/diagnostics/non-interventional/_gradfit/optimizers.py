"""
Optimizers with one shared signature, so they are swappable in an ablation:

    run(f, p0, *, key, **opts) -> (p_best, info)

f(params, key) -> scalar. `key` is only consumed by losses built with
resample=True; adam advances it every step (stochastic gradients), the others
pass one fixed key and so see a deterministic objective.

info always has: n_iter, n_evals (best estimate), final_value, history.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree


def _tree_norm(t):
    return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(t)))


def adam(f, p0, *, key, steps=1000, lr=1e-2, log_every=50):
    opt = optax.adam(lr)

    @jax.jit
    def step(p, s, k):
        v, g = jax.value_and_grad(f)(p, k)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, v

    p, s, hist = p0, opt.init(p0), []
    for i in range(steps):
        key, k = jax.random.split(key)
        p_new, s, v = step(p, s, k)
        if not np.isfinite(float(v)):
            break
        p = p_new
        if i % log_every == 0:
            hist.append(float(v))
    return p, {"n_iter": i + 1, "n_evals": i + 1,
               "final_value": float(f(p, key)), "history": hist}


def lbfgs(f, p0, *, key, steps=500, tol=1e-9, memory=20, log_every=10):
    fk = lambda p: f(p, key)  # noqa: E731 -- fixed key: deterministic objective
    opt = optax.lbfgs(memory_size=memory)
    vg = optax.value_and_grad_from_state(fk)

    @jax.jit
    def step(p, s):
        v, g = vg(p, state=s)
        u, s = opt.update(g, s, p, value=v, grad=g, value_fn=fk)
        return optax.apply_updates(p, u), s, v, _tree_norm(g)

    p, s, hist = p0, opt.init(p0), []
    for i in range(steps):
        p_new, s, v, gn = step(p, s)
        if not (np.isfinite(float(v)) and np.all(np.isfinite(ravel_pytree(p_new)[0]))):
            break
        p = p_new
        if i % log_every == 0:
            hist.append(float(v))
        if float(gn) < tol:
            break
    return p, {"n_iter": i + 1, "n_evals": None,
               "final_value": float(fk(p)), "history": hist}


def cma_es(f, p0, *, key, max_evals=20000, sigma0=0.3, popsize=None, seed=0,
           eval_batch=2):
    import cma

    x0, unravel = ravel_pytree(p0)
    # lax.map in small batches, not a full vmap: the SW loss sorts a
    # (K, n_cells, n_proj) array per candidate, ~70 MB at the default size
    fb = jax.jit(lambda X: jax.lax.map(lambda x: f(unravel(x), key), X,
                                       batch_size=eval_batch))
    opts = {"maxfevals": max_evals, "verbose": -9, "seed": seed + 1}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(np.asarray(x0), sigma0, opts)
    hist = []
    while not es.stop():
        X = es.ask()
        vals = np.asarray(fb(jnp.asarray(np.array(X))))
        vals = np.where(np.isfinite(vals), vals, 1e30)
        es.tell(X, vals.tolist())
        hist.append(float(es.result.fbest))
    p = unravel(jnp.asarray(es.result.xbest))
    return p, {"n_iter": es.result.iterations, "n_evals": es.result.evaluations,
               "final_value": float(f(p, key)), "history": hist[::10]}


def nelder_mead(f, p0, *, key, max_evals=20000, seed=0):
    from scipy.optimize import minimize

    x0, unravel = ravel_pytree(p0)
    fj = jax.jit(lambda x: f(unravel(x), key))

    def obj(x):
        v = float(fj(jnp.asarray(x)))
        return v if np.isfinite(v) else 1e30

    res = minimize(obj, np.asarray(x0), method="Nelder-Mead",
                   options={"maxfev": max_evals, "xatol": 1e-8, "fatol": 1e-12,
                            "adaptive": True})
    p = unravel(jnp.asarray(res.x))
    return p, {"n_iter": int(res.nit), "n_evals": int(res.nfev),
               "final_value": float(res.fun), "history": []}


OPTIMIZERS = {"adam": adam, "lbfgs": lbfgs, "cma": cma_es, "nelder_mead": nelder_mead}


def get(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name]
