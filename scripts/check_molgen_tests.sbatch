#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 0-10:30:00
#SBATCH --job-name molgen_tests
#SBATCH --output job-%J.out
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=24G

set -euo pipefail

module load anaconda

MOLGEN_PYTHON="/home/evronba/.conda/envs/molgen/bin/python"
if [[ ! -x "$MOLGEN_PYTHON" ]]; then
    echo "ERROR: expected Python not found at $MOLGEN_PYTHON" >&2
    exit 1
fi

export PATH="/home/evronba/.conda/envs/molgen/bin:${PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}:${PYTHONPATH:-}"

export MPLCONFIGDIR="${SLURM_SCRATCH_DIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${SLURM_SCRATCH_DIR:-/tmp}/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

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

echo "Running pytest suite..."
"$MOLGEN_PYTHON" -m pytest -q \
    tests/molgen/models/test_dt_gpt_forward.py \
    tests/molgen/models/test_dt_gpt_smiles_dataloader_forward.py \
    tests/molgen/tokenizers/test_char_tokenizer.py \
    tests/molgen/tokenizers/test_tokenizer_factory.py

echo "Running smoke tests..."
"$MOLGEN_PYTHON" scripts/check_forward.py
"$MOLGEN_PYTHON" scripts/check_dynamic_goals.py
