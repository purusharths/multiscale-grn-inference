#!/bin/bash
#SBATCH --job-name=grn-em
#SBATCH --output=logs/grn-em-%A_%a.out
#SBATCH --error=logs/grn-em-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=krikri
#
# Euler-Maruyama ablation for the gradfit pipeline on the non-interventional
# dataset: crosses the GENERATOR's discretisation with the FIT's.
#
#   generator  GRN_GEN_SUBSTEPS in {5, 50}   (5 = the historical, too-coarse default)
#   fit        exact OU  vs  EM with 5 substeps   (per preset)
#
# Question: once the fit's transition matches how the data were made, does
# A_true become the loss minimum and does recovery improve? Run 20260928-152916
# showed an exact-OU fit on 5-substep data scoring BETTER than A_true.
#
# Task index -> (gen_substeps, preset, data_seed):
#     preset       = PRESETS[i % 5]
#     gen_substeps = GENS[(i / 5) % 2]
#     data_seed    = SEEDS[i / 10]
# so --array=0-79 is the full grid (5 presets x 2 generators x 8 seeds).
#
# 4 cpus, not 1 (unlike run_knockout_array.sh): the SW loss sorts
# (14, 3000, 200) arrays and XLA does use the threads; an lbfgs_sw fit took
# ~30 min with the whole login node available.
#
# Submit from the REPO ROOT after `uv sync` on a login node, with logs/ present:
#     mkdir -p logs
#     sbatch --array=0-79 diagnostics/non-interventional/run_em_ablation_array.sh
# Then:
#     .venv/bin/python diagnostics/non-interventional/merge_gradfit_runs.py \
#         results/diagnostics/non-interventional/gradfit/em-ablation

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg

PRESETS=(lbfgs_kl lbfgs_kl_em lbfgs_sw lbfgs_sw_em kl_em_then_sw_em)
GENS=(5 50)
SEEDS=(42 43 44 45 46 47 48 49)

i=${SLURM_ARRAY_TASK_ID:-0}
np=${#PRESETS[@]}; ng=${#GENS[@]}
max=$(( np * ng * ${#SEEDS[@]} - 1 ))
if (( i > max )); then
    echo "ERROR: array index $i exceeds $max" >&2
    exit 1
fi
PRESET=${PRESETS[$(( i % np ))]}
GEN=${GENS[$(( (i / np) % ng ))]}
SEED=${SEEDS[$(( i / (np * ng) ))]}

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# Venv interpreter directly: `uv run` re-syncs .venv per task and races
# (see run_knockout_array.sh for the failure this caused).
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing; run 'uv sync' on a login node first" >&2; exit 1; }

echo "task $i -> preset=$PRESET gen_substeps=$GEN data_seed=$SEED  host=$(hostname)  $(date)"

GRN_GEN_SUBSTEPS=$GEN $PY diagnostics/non-interventional/run_gradfit_ablation.py \
    --presets "$PRESET" --seeds "$SEED" --run-name em-ablation

echo "task $i done  $(date)"
