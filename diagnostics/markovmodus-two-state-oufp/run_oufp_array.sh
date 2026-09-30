#!/bin/bash
#SBATCH --job-name=mm-oufp
#SBATCH --output=logs/mm-oufp-%A_%a.out
#SBATCH --error=logs/mm-oufp-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=krikri,marmot
#
# OU/FP/Cons loss combinations x L1 weight on the markovmodus two-state time
# series; each task fits BOTH states for one (combination, lambda).
#
# Task index -> (combo, lam):
#     combo = COMBOS[i % 6]
#     lam   = LAMS[i / 6]
# so --array=0-23 is the full grid.
#
# The lambda grid is on the sliced-W2 scale, which is NOT the KL folder's
# scale (0.01 there). LAMS below were set from the measured ratio of the two
# losses at the initial theta -- see the commit that added this file.
#
# Submit from the REPO ROOT with logs/ present, after `uv sync` on a login node:
#     mkdir -p logs
#     sbatch --array=0-23 diagnostics/markovmodus-two-state-oufp/run_oufp_array.sh
# Then:
#     .venv/bin/python diagnostics/markovmodus-two-state-oufp/merge_oufp.py \
#         results/markovmodus/two-state-timeseries-oufp/oufp-grid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg

COMBOS=(OU FP OU+FP OU+Cons FP+Cons OU+FP+Cons)
LAMS=(0 1e-5 1e-4 1e-3)

i=${SLURM_ARRAY_TASK_ID:-0}
nc=${#COMBOS[@]}
max=$(( nc * ${#LAMS[@]} - 1 ))
if (( i > max )); then echo "ERROR: array index $i exceeds $max" >&2; exit 1; fi
COMBO=${COMBOS[$(( i % nc ))]}
LAM=${LAMS[$(( i / nc ))]}

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing; run 'uv sync' first" >&2; exit 1; }

echo "task $i -> combo=$COMBO lam=$LAM  host=$(hostname)  $(date)"
$PY diagnostics/markovmodus-two-state-oufp/infer_state_grns_oufp.py \
    --combo "$COMBO" --lam "$LAM" --run-name oufp-grid
echo "task $i done  $(date)"
