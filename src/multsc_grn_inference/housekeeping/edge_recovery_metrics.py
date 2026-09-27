"""
Precision/recall-style scoring for recovered GRN edge structure (A's
off-diagonal entries), as an alternative to a raw correlation coefficient.

GRN edge recovery is fundamentally a detection problem -- is there a
regulatory link between gene i and gene j, yes or no -- not a magnitude-
matching problem. Pearson correlation over (true_off, hat_off) conflates
"ranked the right pairs as edges" with "got the numeric strength right",
and is sensitive to a few large-magnitude entries dominating the
coefficient; a method whose off-diagonal recovery is uniformly near zero
can still land anywhere on it depending on tiny numerical noise (this is
what produced sign-flipping edge_corr values across otherwise-identical
runs before this module existed -- see
tests/diagnostics/jko-testing/README.md).

AUPRC/AUROC/precision@k instead rank candidate edges by |A_hat_ij| and
score that ranking against which entries are actually nonzero in A_true,
independent of whether the recovered magnitudes are numerically close to
the true ones. This is the standard approach in the GRN-inference
literature (the DREAM4/DREAM5 network inference challenges use AUROC and
AUPRC as their primary metrics).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def edge_labels(true_off: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    """Binary ground truth: which off-diagonal entries are real edges."""
    return (np.abs(true_off) > tol).astype(int)


def auprc(true_off: np.ndarray, hat_off: np.ndarray) -> float:
    """
    Area under the precision-recall curve, ranking candidate edges by
    |A_hat_ij| against which entries are real edges in A_true.

    The primary edge-recovery metric to reach for: robust to edge/no-edge
    class imbalance in a way AUROC isn't (most real GRNs are far sparser
    than this branch's 50%-density test network), and it's what the
    DREAM network-inference challenges score on.

    Returns NaN if A_true has no edges or is fully dense (undefined).
    """
    labels = edge_labels(true_off)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, np.abs(hat_off)))


def auroc(true_off: np.ndarray, hat_off: np.ndarray) -> float:
    """
    Area under the ROC curve, same ranking as `auprc`. Report alongside
    AUPRC, not instead of it -- AUPRC is the more informative of the two
    whenever the edge/no-edge split is imbalanced.
    """
    labels = edge_labels(true_off)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, np.abs(hat_off)))


def precision_at_k(true_off: np.ndarray, hat_off: np.ndarray, k: int) -> float:
    """
    Of the k off-diagonal entries with the largest |A_hat_ij|, what
    fraction are real edges. Directly interpretable: "if you trusted our
    top-k predicted edges, how many would actually be real".
    """
    labels = edge_labels(true_off)
    top_k = np.argsort(-np.abs(hat_off))[:k]
    return float(labels[top_k].mean())


def recall_at_k(true_off: np.ndarray, hat_off: np.ndarray, k: int) -> float:
    """Of all real edges, what fraction are captured in the top-k
    |A_hat_ij| entries."""
    labels = edge_labels(true_off)
    n_true = int(labels.sum())
    if n_true == 0:
        return float("nan")
    top_k = np.argsort(-np.abs(hat_off))[:k]
    return float(labels[top_k].sum() / n_true)
