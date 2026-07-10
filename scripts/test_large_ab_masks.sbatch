#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 0-06:00:00
#SBATCH --job-name molgen_test_large_ab
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

export PYTHONPATH="/home/evronba/MolGen:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_SCRATCH_DIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${SLURM_SCRATCH_DIR:-/tmp}/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd /home/evronba/MolGen

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/evronba/MolGen/data/results/epoch_124.pth}"
CONFIG_PATH="${CONFIG_PATH:-/home/evronba/MolGen/data/configs/dt_config_char_normalized_goal_conditioned_model_large_new.toml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/evronba/MolGen/data/tokenizers/charTokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/evronba/MolGen/data/generated_datasets/test_large_ab_masks}"
NUM_MOLECULES="${NUM_MOLECULES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REWARD_A_TARGET="${REWARD_A_TARGET:-1.0}"
REWARD_B_TARGET="${REWARD_B_TARGET:-1.0}"
MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH:-100}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-1.0}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Testing large model with A / B / A+B masks"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Config:         $CONFIG_PATH"
echo "Tokenizer:      $TOKENIZER_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Num molecules:  $NUM_MOLECULES"
echo "Batch size:     $BATCH_SIZE"
echo "Reward targets: A=$REWARD_A_TARGET B=$REWARD_B_TARGET"

"$MOLGEN_PYTHON" -m scripts.generate_reward_datasets \
  --checkpoint "$CHECKPOINT_PATH" \
  --config-path "$CONFIG_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-molecules "$NUM_MOLECULES" \
  --batch-size "$BATCH_SIZE" \
  --model-type DT \
  --reward-a-target "$REWARD_A_TARGET" \
  --reward-b-target "$REWARD_B_TARGET" \
  --max-generation-length "$MAX_GENERATION_LENGTH" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --temperature "$TEMPERATURE"
