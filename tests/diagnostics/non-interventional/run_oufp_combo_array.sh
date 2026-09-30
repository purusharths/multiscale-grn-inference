#!/bin/bash
#SBATCH --job-name=grn-oufp
#SBATCH --output=logs/grn-oufp-%A_%a.out
#SBATCH --error=logs/grn-oufp-%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=marmot,krikri
#
# The paper's loss terms in combination (run_oufp_combo_ablation.py), same
# grid as run_loss_combo_array.sh so the two ablations line up:
#
#   n_genes   5 10 15 20
#   n_cells   500 1000 3000 10000
#   seeds     42 43 44
#   combos    OU FP OU+FP OU+Cons FP+Cons OU+FP+Cons
#
# One fit per task. Task index -> (combo, seed, cells, genes):
#     combo = COMBOS[i % 6]
#     seed  = SEEDS[(i / 6) % 3]
#     cells = CELLS[(i / 18) % 4]
#     genes = GENES[i / 72]
# with i = 287 - SLURM_ARRAY_TASK_ID, so the array starts on the LARGEST
# networks: those tasks take longest, and starting them last would leave the
# run waiting on a tail of day-long fits.
#
# 1 cpu: the losses are numpy (sorts, KDE resampling), single-threaded.
# Timing on marmot: ~7.5 s per evaluation for OU+FP+Cons at 20 genes x 10000
# cells, so that fit (~13.6k evaluations) takes ~28 h; ~0.2-0.5 s per
# evaluation at 5 genes x 500 cells. Whole grid ~900 CPU-hours.
#
# Submit from the REPO ROOT after `uv sync` on a login node, with logs/ present:
#     mkdir -p logs
#     sbatch --array=0-287%128 tests/diagnostics/non-interventional/run_oufp_combo_array.sh
# Then:
#     .venv/bin/python tests/diagnostics/non-interventional/merge_gradfit_runs.py \
#         results/diagnostics/non-interventional/oufp/oufp-combo

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export GRN_GEN_SUBSTEPS=50
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

GENES=(5 10 15 20)
CELLS=(500 1000 3000 10000)
SEEDS=(42 43 44)
COMBOS=(OU FP OU+FP OU+Cons FP+Cons OU+FP+Cons)

nc=${#COMBOS[@]}; ns=${#SEEDS[@]}; nl=${#CELLS[@]}
max=$(( nc * ns * nl * ${#GENES[@]} - 1 ))
t=${SLURM_ARRAY_TASK_ID:-0}
if (( t > max )); then
    echo "ERROR: array index $t exceeds $max" >&2
    exit 1
fi
i=$(( max - t ))
COMBO=${COMBOS[$(( i % nc ))]}
SEED=${SEEDS[$(( (i / nc) % ns ))]}
NC=${CELLS[$(( (i / (nc * ns)) % nl ))]}
NG=${GENES[$(( i / (nc * ns * nl) ))]}

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# Venv interpreter directly: `uv run` re-syncs .venv per task and races
# (see run_knockout_array.sh for the failure this caused).
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing; run 'uv sync' on a login node first" >&2; exit 1; }

echo "task $i -> combo=$COMBO n_genes=$NG n_cells=$NC data_seed=$SEED  host=$(hostname)  $(date)"

$PY tests/diagnostics/non-interventional/run_oufp_combo_ablation.py \
    --combos "$COMBO" --n-genes "$NG" --n-cells "$NC" --seeds "$SEED" --run-name oufp-combo

echo "task $i done  $(date)"
