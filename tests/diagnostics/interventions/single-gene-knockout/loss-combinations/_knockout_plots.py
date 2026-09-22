"""
Knockout-scenario wrappers around the shared recovery figures.

The implementations live in
multsc_grn_inference.housekeeping.recovery_plots, which is scenario-agnostic;
this module only supplies what is specific to the single-gene knockout -- the
titles, and the knocked-out gene to ring in the network panels.

Signatures are unchanged, so compare_losses_single_gene_knockout.py and
merge_knockout_parts.py call these exactly as before.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.housekeeping.recovery_plots import (
    grn_networks as _grn_networks,
    metrics_and_heatmaps as _metrics_and_heatmaps,
)

from _knockout_shared import (
    COMBINATIONS,
    KO_GENE,
    MAXITER,
    N_GENES,
    NETWORK_DENSITY,
    OPTIMIZER_METHOD,
)


def _ordered(fitted: dict) -> dict:
    """COMBINATIONS order, skipping any whose array task failed."""
    return {n: fitted[n] for n in COMBINATIONS if n in fitted}


def metrics_and_heatmaps(df, fitted, A_true, path, *, chance=None):
    _metrics_and_heatmaps(
        df, _ordered(fitted), A_true, path, chance=chance,
        title=(f"Loss-combination comparison on single-gene knockout data  "
               f"(gene_{KO_GENE} KO, {N_GENES} genes, density={NETWORK_DENSITY}, "
               f"fitting full A + sigma, mu known;  {OPTIMIZER_METHOD}, maxiter={MAXITER})"),
    )


def grn_networks(df, fitted, A_true, true_off, k_edges, path):
    _grn_networks(
        df, _ordered(fitted), A_true, np.asarray(true_off), k_edges, path,
        highlight=KO_GENE, highlight_label=f"knocked-out gene (g{KO_GENE})",
        title=(f"Recovered GRN structure vs ground truth  (gene_{KO_GENE} KO, "
               f"{N_GENES} genes, density={NETWORK_DENSITY}; each panel thresholded "
               f"to its top {k_edges} edges; arrow j→i means gene j regulates gene i)"),
    )
