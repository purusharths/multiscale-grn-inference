#!/bin/bash
# SLURM job array for the single-gene-knockout loss-combination comparison.
#
# The six loss combinations are completely independent fits, so running them
# concurrently finishes in the time of the SLOWEST one rather than their sum
# (~3h serial -> ~40min). Each task is single-threaded on purpose: the arrays
# are tiny (8x8 products, 3000x8 clouds) and BLAS threading on them is pure
# synchronisation overhead, so N cores are worth far more as N tasks than as
# N threads inside one task.
#
# Task index -> (combination, data_seed):
#     combination = COMBOS[i % 6]
#     data_seed   = SEEDS[i / 6]
# so --array=0-5 is one seed, --array=0-47 is eight seeds, etc.
#
# Submit (single seed, fastest path to a result):
#     sbatch --array=0-5 run_knockout_array.sh
#
# Submit (eight seeds -- same wall clock, and the thing this project actually
# needs, since every result so far comes from one network):
#     sbatch --array=0-47 run_knockout_array.sh
#
# Then, once the array finishes:
#     uv run python .../merge_knockout_parts.py parts/
#
# Config can be swept without editing code:
#     GRN_N_GENES=12 GRN_NETWORK_DENSITY=0.25 sbatch --array=0-5 run_knockout_array.sh

#SBATCH --job-name=grn-ko
#SBATCH --output=logs/grn-ko-%A_%a.out
#SBATCH --error=logs/grn-ko-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

# Tiny matrices -- BLAS threading is overhead here, not speedup. Leave at 1
# and spend the cores on more concurrent tasks instead.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg          # headless compute node

COMBOS=(OU FP OU+FP OU+Cons FP+Cons OU+FP+Cons)
SEEDS=(42 43 44 45 46 47 48 49)

i=${SLURM_ARRAY_TASK_ID:-0}
COMBO=${COMBOS[$(( i % 6 ))]}
SEED=${SEEDS[$(( i / 6 ))]}

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT="$REPO/tests/diagnostics/interventions/single-gene-knockout/loss-combinations"

mkdir -p "$REPO/logs" "$SCRIPT/parts"
cd "$REPO"

echo "task $i -> combination=$COMBO data_seed=$SEED  host=$(hostname)"

uv run python "$SCRIPT/compare_losses_single_gene_knockout.py" \
    --combination "$COMBO" \
    --data-seed   "$SEED" \
    --out-dir     "$SCRIPT/parts"
