#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 7-00:00:00
#SBATCH --job-name molgen_large_ab_zinc250k
#SBATCH --output job-%J.out
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6

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

DATA_PATH="${DATA_PATH:-data/datasets/250k_rndm_zinc_drugs_clean_3.csv}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/charTokenizer}"
CONFIG_PATH="${CONFIG_PATH:-data/configs/dt_config_char_normalized_goal_conditioned_model_large_new.toml}"
SAVE_PATH="${SAVE_PATH:-/home/evronba/MolGen/data/results/model_large_ab_zinc250k}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: data file not found: $DATA_PATH" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$SAVE_PATH"

echo "Running large reward-conditioned DT training on zinc250k..."
echo "Data path:      $DATA_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Config path:    $CONFIG_PATH"
echo "Save path:      $SAVE_PATH"

"$MOLGEN_PYTHON" -m scripts.pretrain \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --save-path "$SAVE_PATH" \
    --model-type DT \
    --dataset-type DT_SMILES \
    --config-path "$CONFIG_PATH"
