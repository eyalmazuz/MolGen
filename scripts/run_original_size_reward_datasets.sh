#!/bin/bash
set -euo pipefail

cd /home/evronba/MolGen
source /home/evronba/MolGen/.venv/bin/activate
export PYTHONPATH="/home/evronba/MolGen:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-/home/evronba/MolGen/data/results/epoch_124.pth}"
CONFIG_PATH="${CONFIG_PATH:-data/configs/dt_config_char_normalized_goal_conditioned_original_size.toml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/charTokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-data/generated_datasets_original_size}"
NUM_MOLECULES="${NUM_MOLECULES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

/home/evronba/.conda/envs/molgen/bin/python scripts/generate_reward_datasets.py \
  --checkpoint "$CHECKPOINT" \
  --config-path "$CONFIG_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --model-type DT \
  --output-dir "$OUTPUT_DIR" \
  --num-molecules "$NUM_MOLECULES" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE"
