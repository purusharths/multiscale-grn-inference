#!/bin/bash
#SBATCH --job-name=grn-ko
#SBATCH --output=logs/grn-ko-%A_%a.out
#SBATCH --error=logs/grn-ko-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#
# SLURM job array for the single-gene-knockout loss-combination comparison.
#
# NOTE: directives must stay immediately below the shebang -- sbatch stops
# scanning for them at the first non-comment line, and some site wrappers stop
# earlier than that. Explanation goes here, below them, not above.
#
# The six loss combinations are completely independent fits, so running them
# concurrently finishes in the time of the SLOWEST one rather than their sum
# (~3h serial -> ~40min). Each task is single-threaded on purpose: the arrays
# are tiny (8x8 products, 3000x8 clouds) and BLAS threading on them is pure
# synchronisation overhead, so N cores buy far more as N tasks than as N
# threads inside one task.
#
# Task index -> (combination, data_seed):
#     combination = COMBOS[i % 6]
#     data_seed   = SEEDS[i / 6]
# so --array=0-5 is one seed, --array=0-47 is all eight.
#
# Submit from the REPO ROOT (the script resolves paths from SLURM_SUBMIT_DIR),
# and create logs/ first -- SLURM opens the .out/.err files before this script
# runs, so a missing logs/ kills every task with no diagnostic anywhere:
#
#     mkdir -p logs
#     sbatch --array=0-47 tests/diagnostics/interventions/single-gene-knockout/loss-combinations/run_knockout_array.sh
#
# Then, once it drains:
#     uv run python .../merge_knockout_parts.py .../parts/
#
# Config sweeps need no code edit (see _knockout_shared.py):
#     GRN_N_GENES=12 GRN_NETWORK_DENSITY=0.25 sbatch --array=0-5 ...

set -euo pipefail

# Batch jobs get a non-interactive, non-login shell that does not source
# ~/.bashrc, so a uv installed to ~/.local/bin is invisible without this.
export PATH="$HOME/.local/bin:$PATH"

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
max=$(( ${#COMBOS[@]} * ${#SEEDS[@]} - 1 ))
if (( i > max )); then
    echo "ERROR: array index $i exceeds $max (${#COMBOS[@]} combinations x ${#SEEDS[@]} seeds)." >&2
    echo "Use --array=0-$max, or add seeds to SEEDS." >&2
    exit 1
fi

COMBO=${COMBOS[$(( i % ${#COMBOS[@]} ))]}
SEED=${SEEDS[$(( i / ${#COMBOS[@]} ))]}

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT="$REPO/tests/diagnostics/interventions/single-gene-knockout/loss-combinations"
mkdir -p "$SCRIPT/parts"
cd "$REPO"

# Call the venv's interpreter directly rather than going through `uv run`.
#
# `uv run` re-syncs the project into .venv on EVERY invocation. With an array,
# all tasks start at once and race to reinstall the local package into one
# shared .venv: one process removes the .dist-info directory while another is
# copying RECORD into it, and the loser dies with
#     failed to copy file ... RECORD: No such file or directory (os error 2)
# That is what killed 2 of the first 48 tasks. It is a filesystem race, not a
# problem with the job -- and it gets likelier as the array grows.
#
# Run `uv sync` ONCE on a login node before submitting; tasks then only read
# the venv, never write it, so there is nothing to contend over.
if [[ -x "$REPO/.venv/bin/python" ]]; then
    PY="$REPO/.venv/bin/python"
else
    command -v uv >/dev/null || { echo "ERROR: no .venv and uv not on PATH ($PATH)" >&2; exit 1; }
    PY="uv run --no-sync python"   # --no-sync: never write to .venv from a task
fi

echo "task $i -> combination=$COMBO data_seed=$SEED  host=$(hostname)  $(date)"
echo "interpreter: $PY"

$PY "$SCRIPT/compare_losses_single_gene_knockout.py" \
    --combination "$COMBO" \
    --data-seed   "$SEED" \
    --out-dir     "$SCRIPT/parts"

echo "task $i done  $(date)"
