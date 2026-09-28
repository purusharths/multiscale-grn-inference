"""
A fit = a Config: parameterization + initialisation + a chain of Stages.
Each Stage is (loss, optimizer, penalty) and warm-starts from the previous
stage's A, so "moment fit -> SW refine" and "SW only" differ by one Stage.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np

from . import losses, optimizers, params


@dataclass(frozen=True)
class Stage:
    loss: str = "sw"                 # sw | kl | bures
    optimizer: str = "lbfgs"         # adam | lbfgs | cma | nelder_mead
    lam: float = 0.0                 # smooth-L1 weight on off-diagonal A
    # loss options (ignored by losses that do not use them)
    transition: str = "exact"        # sw: exact | em
    n_proj: int = 200                # sw
    resample: bool = False           # sw: fresh projections + noise per eval
    floor: bool = True               # sw: apply the generator's expression floor
    rollout: bool = False            # kl/bures: multi-step moment propagation
    opt_kwargs: dict = field(default_factory=dict, hash=False)

    def loss_kwargs(self):
        if self.loss == "sw":
            return {"transition": self.transition, "n_proj": self.n_proj,
                    "resample": self.resample, "floor": self.floor}
        return {"rollout": self.rollout}


@dataclass(frozen=True)
class Config:
    name: str
    stages: tuple
    param: str = "posdiag"           # full | posdiag
    init: str = "diag"               # diag | random | truth
    n_starts: int = 1                # starts > 0 use init="random"
    seed: int = 0

    def describe(self):
        return {"name": self.name, "param": self.param, "init": self.init,
                "n_starts": self.n_starts,
                "stages": [asdict(s) for s in self.stages]}


L1_EPS = 1e-3


def smooth_l1_offdiag(A, eps=L1_EPS):
    off = A * (1.0 - jnp.eye(A.shape[0]))
    return jnp.sum(jnp.sqrt(off**2 + eps**2) - eps)


def stage_objective(problem, stage, param, seed):
    """(objective over params, raw loss over A). Loss seed fixes projections/noise."""
    loss = losses.build(problem, stage.loss, seed=seed, **stage.loss_kwargs())

    def obj(p, key):
        A = param.to_A(p)
        val = loss(A, key)
        return val + stage.lam * smooth_l1_offdiag(A) if stage.lam else val

    return obj, loss


def _loss_seed(cfg, i):
    # depends on the stage, NOT on the start: every start of a config sees the
    # same projections/noise, so their final objectives are comparable
    return cfg.seed + 7919 * i


def _run_stages(problem, cfg, A0, opt_seed):
    param = params.get(cfg.param, problem.n_genes)
    A, logs = np.asarray(A0, dtype=float), []
    for i, stage in enumerate(cfg.stages):
        obj, _ = stage_objective(problem, stage, param, seed=_loss_seed(cfg, i))
        opt = optimizers.get(stage.optimizer)
        kw = dict(stage.opt_kwargs)
        if stage.optimizer in ("cma", "nelder_mead"):
            kw.setdefault("seed", opt_seed)
        t0 = time.perf_counter()
        key = jax.random.PRNGKey(opt_seed + 104729 * (i + 1))
        p, info = opt(obj, param.init(A), key=key, **kw)
        info["seconds"] = time.perf_counter() - t0
        info["stage"] = i
        A = np.asarray(param.to_A(p))
        logs.append(info)
    return A, logs


@dataclass
class FitResult:
    A_hat: np.ndarray
    start_logs: list          # per start: list of per-stage info dicts
    best_start: int
    seconds: float


def fit(problem, cfg: Config) -> FitResult:
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    # starts are ranked on the last stage's objective with its randomness FIXED
    # (resample=False), else an adam-with-resampling stage would rank on noise
    last = len(cfg.stages) - 1
    rank_stage = replace(cfg.stages[last], resample=False)
    rank_obj, _ = stage_objective(problem, rank_stage, params.Full(), _loss_seed(cfg, last))
    best, best_val, all_logs = None, np.inf, []
    for s in range(cfg.n_starts):
        kind = cfg.init if s == 0 else "random"
        A0 = params.initial_A(kind, problem.n_genes, rng, A_true=problem.A_true)
        A, logs = _run_stages(problem, cfg, A0, opt_seed=cfg.seed + 1000 * s)
        all_logs.append(logs)
        val = float(rank_obj(jnp.asarray(A), None))
        logs[-1]["rank_value"] = val
        if np.isfinite(val) and val < best_val:
            best, best_val, best_s = A, val, s
    if best is None:
        best, best_s = A, cfg.n_starts - 1
    return FitResult(best, all_logs, best_s, time.perf_counter() - t0)
