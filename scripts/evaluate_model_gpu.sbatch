#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 0-06:00:00
#SBATCH --job-name molgen_eval
#SBATCH --output job-%J.out
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=24G

set -euo pipefail

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate molgen

MOLGEN_PYTHON="$(which python)"
if [[ ! -x "$MOLGEN_PYTHON" ]]; then
    echo "ERROR: expected Python not found in activated molgen env" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}:${PYTHONPATH:-}"

export MPLCONFIGDIR="${SLURM_SCRATCH_DIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${SLURM_SCRATCH_DIR:-/tmp}/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CHECKPOINT_SOURCE="${SLURM_SUBMIT_DIR:-$(pwd)}/data/results/epoch_124.pth"
CHECKPOINT_WORKDIR="${SLURM_SCRATCH_DIR:-/tmp}/molgen_eval_checkpoint"
RESULTS_PATH="${SLURM_SUBMIT_DIR:-$(pwd)}/data/results/eval_epoch_124_gpu"
mkdir -p "$CHECKPOINT_WORKDIR" "$RESULTS_PATH"
cp "$CHECKPOINT_SOURCE" "$CHECKPOINT_WORKDIR/epoch_124.pth"

echo "=== CLUSTER JOB DEBUG INFO ==="
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: ${PYTHONPATH:-<unset>}"
echo "Executable: $MOLGEN_PYTHON"
"$MOLGEN_PYTHON" - <<'PY'
import sys
import torch

print(f"Torch version: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in the current Python environment.", file=sys.stderr)
    raise SystemExit(1)
PY
echo "=============================="

echo "Running evaluation on GPU..."
"$MOLGEN_PYTHON" -m molgen.training.evaluate \
    --checkpoint "$CHECKPOINT_WORKDIR/epoch_124.pth" \
    --data_path data/datasets/zinc_smiles_toy.text \
    --results_path "$RESULTS_PATH" \
    --k 20 \
    --device cuda \
    --tokenizer_path data/tokenizers/charTokenizer \
    --model_type DT \
    --dataset_type DT_SMILES \
    --config_path data/configs/dt_config_char_normalized_goal_conditioned.toml \
    --rtg '{"QED": 0.8, "PlogP": 0.6}' \
    --stats
