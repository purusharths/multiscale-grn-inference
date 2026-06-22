"""
Data loading for multiscale GRN inference.

CSV files are named  expression_t{k}_{cell_index}.csv  and live in csv_dir.
Each file contains multiple cells with columns:
    cell_id, population, gene_0, ..., gene_{G-1}

All files sharing the same timepoint index k are concatenated and de-duplicated
(by cell_id) to form  X_{t_k} ∈ R^{N × G}.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(csv_dir: str) -> dict[int, np.ndarray]:
    """
    Load cross-sectional snapshots from CSV files.

    Parameters
    ----------
    csv_dir : str
        Directory containing expression_t{k}_{cell_index}.csv files.

    Returns
    -------
    dict[int, np.ndarray]
        Maps timepoint index k → X_{t_k} of shape (N_k, G), float64.
        Cells are sorted by cell_id within each timepoint.
    """
    csv_path = Path(csv_dir)
    pattern = re.compile(r"^expression_t(\d+)_(\d+)\.csv$")

    timepoint_files: dict[int, list[Path]] = {}
    for f in csv_path.glob("expression_t*.csv"):
        m = pattern.match(f.name)
        if not m:
            continue
        t_idx = int(m.group(1))
        timepoint_files.setdefault(t_idx, []).append(f)

    if not timepoint_files:
        raise ValueError(f"No expression CSV files found in '{csv_dir}'")

    dataset: dict[int, np.ndarray] = {}
    for t_idx in sorted(timepoint_files.keys()):
        frames: list[pd.DataFrame] = []
        for fpath in sorted(timepoint_files[t_idx]):
            df = pd.read_csv(fpath)
            gene_cols = [c for c in df.columns if c.startswith("gene_")]
            if not gene_cols:
                continue
            frames.append(df[["cell_id"] + gene_cols])

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("cell_id")
            .sort_values("cell_id")
        )
        gene_cols = [c for c in combined.columns if c.startswith("gene_")]
        dataset[t_idx] = combined[gene_cols].values.astype(np.float64)

    _validate(dataset)
    return dataset


def load_true_grns(grn_dir: str) -> dict[str, np.ndarray]:
    """
    Load ground-truth weighted GRN matrices from  grn_weighted_pop_*.csv  files.

    Parameters
    ----------
    grn_dir : str
        Directory containing grn_weighted_pop_*.csv (and optionally
        grn_adjacency_pop_*.csv).

    Returns
    -------
    dict[str, np.ndarray]
        Maps population label (e.g. "pop_0") → (G, G) float64 matrix.
        Row i, column j = weight of regulation from gene j onto gene i
        (convention matches the CSV index labels).
    """
    grn_path = Path(grn_dir)
    grns: dict[str, np.ndarray] = {}
    for fpath in sorted(grn_path.glob("grn_weighted_pop_*.csv")):
        pop_label = fpath.stem.replace("grn_weighted_", "")   # e.g. "pop_0"
        df = pd.read_csv(fpath, index_col=0)
        grns[pop_label] = df.values.astype(np.float64)
    if not grns:
        raise ValueError(f"No grn_weighted_pop_*.csv files found in '{grn_dir}'")
    return grns


def pooled_grn(
    grns: dict[str, np.ndarray],
    cell_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Compute a cell-count-weighted mean of per-population GRN matrices.

    Parameters
    ----------
    grns        : output of load_true_grns()
    cell_counts : optional dict pop_label → number of cells.
                  Defaults to equal weights when None.

    Returns
    -------
    (G, G) float64 weighted-mean GRN matrix.
    """
    pops = sorted(grns.keys())
    if cell_counts is None:
        weights = {p: 1.0 for p in pops}
    else:
        weights = {p: float(cell_counts.get(p, 1)) for p in pops}

    total = sum(weights.values())
    result = sum(weights[p] / total * grns[p] for p in pops)
    return result


def _validate(dataset: dict[int, np.ndarray]) -> None:
    Gs = {k: X.shape[1] for k, X in dataset.items()}
    if len(set(Gs.values())) != 1:
        raise ValueError(f"Gene count (G) differs across timepoints: {Gs}")

    Ns = {k: X.shape[0] for k, X in dataset.items()}
    if len(set(Ns.values())) != 1:
        warnings.warn(
            f"Cell count (N) differs across timepoints: {Ns}. "
            "This is expected for destructive measurements.",
            stacklevel=2,
        )
