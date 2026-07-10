#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 7-00:00:00
#SBATCH --job-name molgen_model_large_new
#SBATCH --output job-%J.out
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evronba@post.bgu.ac.il


set -euo pipefail

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate molgen

MOLGEN_PYTHON="/home/evronba/.conda/envs/molgen/bin/python"
if [[ ! -x "$MOLGEN_PYTHON" ]]; then
    echo "ERROR: expected Python not found at $MOLGEN_PYTHON" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="/home/evronba/MolGen:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_SCRATCH_DIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${SLURM_SCRATCH_DIR:-/tmp}/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd /home/evronba/MolGen

SAVE_PATH="/home/evronba/MolGen/data/results/model_large_new"
mkdir -p "$SAVE_PATH"

echo "Running new large goal-conditioned DT training..."
"$MOLGEN_PYTHON" -m scripts.pretrain \
    --data-path data/datasets/zinc_smiles.text \
    --tokenizer-path data/tokenizers/charTokenizer \
    --save-path "$SAVE_PATH" \
    --model-type DT \
    --dataset-type DT_SMILES \
    --config-path data/configs/dt_config_char_normalized_goal_conditioned_model_large_new.toml
