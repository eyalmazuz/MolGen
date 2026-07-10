#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 0-06:00:00
#SBATCH --job-name molgen_reward_sets
#SBATCH --output job-%J.out
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=24G

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

mkdir -p /home/evronba/MolGen/data/generated_datasets

echo "Running reward dataset generation..."
"$MOLGEN_PYTHON" -m scripts.generate_reward_datasets \
  --checkpoint /home/evronba/MolGen/data/results/epoch_124.pth \
  --config-path /home/evronba/MolGen/data/configs/dt_config_char_normalized_goal_conditioned.toml \
  --tokenizer-path /home/evronba/MolGen/data/tokenizers/charTokenizer \
  --output-dir /home/evronba/MolGen/data/generated_datasets \
  --num-molecules 1000 \
  --batch-size 32 \
  --model-type DT \
  --reward-a-target 1.0 \
  --reward-b-target 1.0 \
  --max-generation-length 100 \
  --seed 42 \
  --device cuda
