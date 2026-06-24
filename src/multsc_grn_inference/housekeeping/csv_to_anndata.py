"""
Convert per-timepoint CSV files (produced by run_non_stationary.py) to AnnData .h5ad files.

Usage:
    python -m multsc_grn_inference.housekeeping.csv_to_anndata [INPUT_DIR] [OUTPUT_DIR]

    INPUT_DIR  — directory containing expression_t*.csv files  (default: output/expression_by_timepoint)
    OUTPUT_DIR — where to write .h5ad files                    (default: same as INPUT_DIR)
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import pandas as pd


def csv_dir_to_anndata(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("expression_t*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No expression_t*.csv files found in {input_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        meta_cols = [c for c in ("cell_id", "population") if c in df.columns]
        gene_cols = [c for c in df.columns if c not in meta_cols]

        obs = df[meta_cols].copy().reset_index(drop=True)
        if "cell_id" in obs.columns:
            obs["cell_id"] = obs["cell_id"].astype(str)
            obs.index = obs["cell_id"]

        adata = ad.AnnData(
            X=df[gene_cols].to_numpy(),
            obs=obs,
            var=pd.DataFrame(index=gene_cols),
        )

        # embed the timepoint value from the filename (e.g. expression_t0_2667.csv → 0.2667)
        t_str = csv_path.stem.replace("expression_t", "").replace("_", ".", 1)
        try:
            adata.uns["time"] = float(t_str)
        except ValueError:
            pass

        out_path = output_dir / csv_path.with_suffix(".h5ad").name
        adata.write_h5ad(out_path)
        print(f"  {csv_path.name}  →  {out_path.name}")


if __name__ == "__main__":
    args = sys.argv[1:]
    inp = Path(args[0]) if len(args) > 0 else Path("output/expression_by_timepoint")
    out = Path(args[1]) if len(args) > 1 else None
    csv_dir_to_anndata(inp, out)
