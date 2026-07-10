#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 7-00:00:00
#SBATCH --job-name molgen_large_reward_ab
#SBATCH --output job-%A_%a.out
#SBATCH --array=0-2
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

CONDITIONS=(
    "reward_A"
    "reward_B"
    "reward_A_and_B"
)

DATA_PATHS=(
    "data/generated_datasets/generated_reward_A.csv"
    "data/generated_datasets/generated_reward_B.csv"
    "data/generated_datasets/generated_reward_A_and_B.csv"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONDITION="${CONDITIONS[$TASK_ID]}"
DATA_PATH="${DATA_PATHS[$TASK_ID]}"

CONFIG_PATH="${CONFIG_PATH:-data/configs/dt_config_char_normalized_goal_conditioned_model_large_new.toml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/charTokenizer}"
SAVE_PATH="${SAVE_ROOT:-/home/evronba/MolGen/data/results}/model_large_${CONDITION}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: data file not found: $DATA_PATH" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$SAVE_PATH"

# Optional fine-tuning mode:
#   sbatch --export=ALL,FINETUNE_FROM=/home/evronba/MolGen/data/results/epoch_124.pth scripts/train_large_reward_ab_array.sbatch
RUN_CONFIG="$CONFIG_PATH"
if [[ -n "${FINETUNE_FROM:-}" ]]; then
    if [[ ! -f "$FINETUNE_FROM" ]]; then
        echo "ERROR: FINETUNE_FROM checkpoint not found: $FINETUNE_FROM" >&2
        exit 1
    fi

    cp "$FINETUNE_FROM" "$SAVE_PATH/latest.pth"
    RUN_CONFIG="${SLURM_SCRATCH_DIR:-/tmp}/dt_config_${CONDITION}_load_checkpoint.toml"
    sed 's/load_checkpoint = false/load_checkpoint = true/' "$CONFIG_PATH" > "$RUN_CONFIG"
fi

echo "Running large reward-conditioned DT training"
echo "Condition:      $CONDITION"
echo "Data path:      $DATA_PATH"
echo "Config path:    $RUN_CONFIG"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Save path:      $SAVE_PATH"
echo "Fine-tune from: ${FINETUNE_FROM:-<none>}"

"$MOLGEN_PYTHON" -m scripts.pretrain \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --save-path "$SAVE_PATH" \
    --model-type DT \
    --dataset-type DT_SMILES \
    --config-path "$RUN_CONFIG"
