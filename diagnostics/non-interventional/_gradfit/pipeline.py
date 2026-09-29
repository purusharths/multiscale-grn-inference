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
    transition: str = "exact"        # all losses: exact | em
    n_sub: int = 5                   # em substeps per interval
    n_proj: int = 200                # sw
    resample: bool = False           # sw: fresh projections + noise per eval
    floor: bool = True               # sw: apply the generator's expression floor
    rollout: bool = False            # kl/bures: multi-step moment propagation
    shrinkage: object = None         # kl/bures: None | "lw" | float (see losses)
    opt_kwargs: dict = field(default_factory=dict, hash=False)

    def loss_kwargs(self):
        tkw = {"transition": self.transition,
               "transition_kw": {"n_sub": self.n_sub} if self.transition == "em" else None}
        if self.loss == "sw":
            return {**tkw, "n_proj": self.n_proj,
                    "resample": self.resample, "floor": self.floor}
        return {**tkw, "rollout": self.rollout, "shrinkage": self.shrinkage}


@dataclass(frozen=True)
class Config:
    name: str
    stages: tuple
    param: str = "posdiag"           # full | posdiag
    init: str = "diag"               # diag | random | truth
    init_diag: float = 1.0           # diagonal of the diag/random inits
    n_starts: int = 1                # starts > 0 use init="random"
    learn_mu: bool = False           # also fit mu (Problem.mu is the init)
    learn_sigma: bool = False        # also fit per-gene sigma (Problem.sigma is the init)
    seed: int = 0

    def model(self, G):
        return params.Model(params.get(self.param, G), G, self.learn_mu, self.learn_sigma)

    def describe(self):
        return {"name": self.name, "param": self.param, "init": self.init,
                "init_diag": self.init_diag, "n_starts": self.n_starts,
                "learn_mu": self.learn_mu, "learn_sigma": self.learn_sigma,
                "stages": [asdict(s) for s in self.stages]}


L1_EPS = 1e-3


def smooth_l1_offdiag(A, eps=L1_EPS):
    off = A * (1.0 - jnp.eye(A.shape[0]))
    return jnp.sum(jnp.sqrt(off**2 + eps**2) - eps)


def stage_objective(problem, stage, model, seed):
    """(objective over params, raw loss over theta). Loss seed fixes projections/noise."""
    loss = losses.build(problem, stage.loss, seed=seed, **stage.loss_kwargs())

    def obj(p, key):
        th = model.to_theta(p)
        val = loss(th, key)
        return val + stage.lam * smooth_l1_offdiag(th["A"]) if stage.lam else val

    return obj, loss


def _loss_seed(cfg, i):
    # depends on the stage, NOT on the start: every start of a config sees the
    # same projections/noise, so their final objectives are comparable
    return cfg.seed + 7919 * i


def _run_stages(problem, cfg, theta0, opt_seed):
    model = cfg.model(problem.n_genes)
    theta, logs = dict(theta0), []
    for i, stage in enumerate(cfg.stages):
        obj, _ = stage_objective(problem, stage, model, seed=_loss_seed(cfg, i))
        opt = optimizers.get(stage.optimizer)
        kw = dict(stage.opt_kwargs)
        if stage.optimizer in ("cma", "nelder_mead"):
            kw.setdefault("seed", opt_seed)
        t0 = time.perf_counter()
        key = jax.random.PRNGKey(opt_seed + 104729 * (i + 1))
        p, info = opt(obj, model.init(theta), key=key, **kw)
        info["seconds"] = time.perf_counter() - t0
        info["stage"] = i
        theta.update({k: np.asarray(v) for k, v in model.to_theta(p).items()})
        logs.append(info)
    return theta, logs


@dataclass
class FitResult:
    theta_hat: dict           # {"A", "mu", "sigma"}; mu/sigma = Problem's if not fitted
    start_logs: list          # per start: list of per-stage info dicts
    best_start: int
    seconds: float

    @property
    def A_hat(self):
        return self.theta_hat["A"]


def fit(problem, cfg: Config) -> FitResult:
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    # starts are ranked on the last stage's objective with its randomness FIXED
    # (resample=False), else an adam-with-resampling stage would rank on noise
    last = len(cfg.stages) - 1
    rank_stage = replace(cfg.stages[last], resample=False)
    G = problem.n_genes
    rank_model = params.Model(params.Full(), G, cfg.learn_mu, cfg.learn_sigma)
    rank_obj, _ = stage_objective(problem, rank_stage, rank_model, _loss_seed(cfg, last))
    best, best_val, all_logs = None, np.inf, []
    for s in range(cfg.n_starts):
        kind = cfg.init if s == 0 else "random"
        A0 = params.initial_A(kind, G, rng, A_true=problem.A_true, diag=cfg.init_diag)
        theta0 = {"A": A0, "mu": np.asarray(problem.mu, dtype=float),
                  "sigma": np.asarray(problem.sigma, dtype=float)}
        theta, logs = _run_stages(problem, cfg, theta0, opt_seed=cfg.seed + 1000 * s)
        all_logs.append(logs)
        val = float(rank_obj(rank_model.init(theta), None))
        logs[-1]["rank_value"] = val
        if np.isfinite(val) and val < best_val:
            best, best_val, best_s = theta, val, s
    if best is None:
        best, best_s = theta, cfg.n_starts - 1
    return FitResult(best, all_logs, best_s, time.perf_counter() - t0)
