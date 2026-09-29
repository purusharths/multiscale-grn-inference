#!/bin/bash
#SBATCH --job-name=grn-lc
#SBATCH --output=logs/grn-lc-%A_%a.out
#SBATCH --error=logs/grn-lc-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=marmot,krikri
#
# Loss-combination ablation (run_loss_combo_ablation.py) at the best setting
# so far, over growing GRNs and cell counts:
#
#   n_genes   5 10 15 20
#   n_cells   500 1000 3000 10000
#   seeds     42 43 44            (a different random network per seed)
#   combos    all 15 non-empty subsets of {kl, bures, kl_ro, sw}
#
# The 7 moment-only combos take seconds each, so one task runs all of them for
# a (genes, cells, seed) cell; each of the 8 combos containing sw gets its own
# task (an SW L-BFGS fit took ~30 min at 8 genes x 3000 cells).
#
# Task index -> (slot, seed, cells, genes):
#     slot  = i % 9          0 = all moment combos, 1..8 = one sw combo
#     seed  = SEEDS[(i / 9) % 3]
#     cells = CELLS[(i / 27) % 4]
#     genes = GENES[i / 108]
# so --array=0-431 is the full grid, smallest networks first.
#
# 2 cpus, not 4 (run_em_ablation_array.sh): krikri showed CPULoad ~32 over 128
# allocated cpus for 32 of those tasks, i.e. ~1 busy core each.
#
# Submit from the REPO ROOT after `uv sync` on a login node, with logs/ present:
#     mkdir -p logs
#     sbatch --array=0-431%48 tests/diagnostics/non-interventional/run_loss_combo_array.sh
# Then:
#     .venv/bin/python tests/diagnostics/non-interventional/merge_gradfit_runs.py \
#         results/diagnostics/non-interventional/gradfit/loss-combo

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export GRN_GEN_SUBSTEPS=50

GENES=(5 10 15 20)
CELLS=(500 1000 3000 10000)
SEEDS=(42 43 44)
SW_COMBOS=(sw kl+sw bures+sw kl_ro+sw kl+bures+sw kl+kl_ro+sw bures+kl_ro+sw kl+bures+kl_ro+sw)

i=${SLURM_ARRAY_TASK_ID:-0}
nslot=$(( ${#SW_COMBOS[@]} + 1 ))
max=$(( nslot * ${#SEEDS[@]} * ${#CELLS[@]} * ${#GENES[@]} - 1 ))
if (( i > max )); then
    echo "ERROR: array index $i exceeds $max" >&2
    exit 1
fi
slot=$(( i % nslot ))
SEED=${SEEDS[$(( (i / nslot) % ${#SEEDS[@]} ))]}
NC=${CELLS[$(( (i / (nslot * ${#SEEDS[@]})) % ${#CELLS[@]} ))]}
NG=${GENES[$(( i / (nslot * ${#SEEDS[@]} * ${#CELLS[@]}) ))]}
if (( slot == 0 )); then COMBOS=moments; else COMBOS=${SW_COMBOS[$(( slot - 1 ))]}; fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# Venv interpreter directly: `uv run` re-syncs .venv per task and races
# (see run_knockout_array.sh for the failure this caused).
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing; run 'uv sync' on a login node first" >&2; exit 1; }

echo "task $i -> combos=$COMBOS n_genes=$NG n_cells=$NC data_seed=$SEED  host=$(hostname)  $(date)"

$PY tests/diagnostics/non-interventional/run_loss_combo_ablation.py \
    --combos "$COMBOS" --n-genes "$NG" --n-cells "$NC" --seeds "$SEED" --run-name loss-combo

echo "task $i done  $(date)"
