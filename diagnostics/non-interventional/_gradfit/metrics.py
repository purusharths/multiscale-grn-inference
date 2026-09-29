"""
Recovery metrics against A_true, and "check 0": is the optimizer or the loss
the bottleneck?

check 0 evaluates a loss at A_hat and at A_true under the SAME randomness,
on seeds the fit never saw. gap = L(A_hat) - L(A_true):
    gap > 0  (beyond seed noise)  the optimizer stopped short of a point the
             loss itself prefers -> a better optimizer can still help.
    gap <= 0                      the fit is at least as good as the truth by
             the loss's own standard -> remaining error is the loss / data
             (bias, identifiability), and no optimizer will close it.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.stats import rankdata

from . import losses


def _offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def auroc(scores, labels):
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recovery(A_hat, A_true):
    A_hat, A_true = np.asarray(A_hat), np.asarray(A_true)
    oh, ot = _offdiag(A_hat), _offdiag(A_true)
    edge = ot != 0
    return {
        "rel_fro": float(np.linalg.norm(A_hat - A_true) / np.linalg.norm(A_true)),
        "offdiag_rel_fro": float(np.linalg.norm(oh - ot) / np.linalg.norm(ot)),
        "diag_rel_err": float(np.linalg.norm(np.diag(A_hat) - np.diag(A_true))
                              / np.linalg.norm(np.diag(A_true))),
        "offdiag_corr": float(np.corrcoef(oh, ot)[0, 1]),
        "edge_auroc": auroc(np.abs(oh), edge),
        "sign_acc_on_edges": float(np.mean(np.sign(oh[edge]) == np.sign(ot[edge]))),
    }


def check0(problem, A_hat, *, loss="sw", eval_seeds=(9001, 9002, 9003, 9004, 9005),
           **loss_kw):
    """Loss gap A_hat vs A_true, averaged over held-out fixed-randomness seeds."""
    loss_kw = {**loss_kw, "resample": False} if loss == "sw" else loss_kw
    seeds = eval_seeds if loss == "sw" else eval_seeds[:1]   # moment losses are deterministic
    fh, ft = [], []
    for s in seeds:
        f = losses.build(problem, loss, seed=s, **loss_kw)
        fh.append(float(f(jnp.asarray(A_hat), None)))
        ft.append(float(f(jnp.asarray(problem.A_true), None)))
    gaps = np.array(fh) - np.array(ft)
    se = float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
    return {"L_fit": float(np.mean(fh)), "L_true": float(np.mean(ft)),
            "gap": float(gaps.mean()), "gap_se": se,
            "verdict": ("optimizer-limited" if gaps.mean() > 2 * se and gaps.mean() > 0
                        else "loss-limited")}
