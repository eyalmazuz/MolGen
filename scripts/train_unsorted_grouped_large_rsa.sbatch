#!/bin/bash
#SBATCH --partition rtx3090
#SBATCH --time 7-00:00:00
#SBATCH --job-name molgen_rsa_grouped
#SBATCH --output job-%J.out
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
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

cd /home/evronba/MolGen

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="/home/evronba/MolGen:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_SCRATCH_DIR:-/tmp}/matplotlib_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_SCRATCH_DIR:-/tmp}/.cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

SAVE_PATH="${SAVE_PATH:-/groups/bshapira_group/molgenarbitrary/unsorted_grouped_large_rsa_checkpoint_or}"
CONFIG_PATH="${CONFIG_PATH:-data/configs/dt_config_char_normalized_goal_conditioned_model_large_rsa.toml}"
RUN_INFO_PATH="$SAVE_PATH/RUN_INFO.txt"
mkdir -p "$SAVE_PATH"

{
    echo "Run name: unsorted_grouped_large_rsa_checkpoint_or"
    echo "Purpose: full-size explicit RSA reward-token training with no reward-goal padding"
    echo "Architecture: R_1 ... R_k, S, A packed per timestep; goals sorted by goal id in model forward"
    echo "Started at: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Slurm job id: ${SLURM_JOB_ID:-unknown}"
    echo "Git HEAD: $(git rev-parse --short HEAD)"
    echo "Git status:"
    git status --short
    echo
    echo "Save path: $SAVE_PATH"
    echo "Config: $CONFIG_PATH"
    echo "Data: data/datasets/zinc_smiles.text"
    echo "Tokenizer: data/tokenizers/charTokenizer"
} | tee "$RUN_INFO_PATH"

nvidia-smi --query-gpu=name,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,memory.used,memory.total \
    --format=csv

"$MOLGEN_PYTHON" -m scripts.pretrain_dt_save_final \
    --data-path data/datasets/zinc_smiles.text \
    --tokenizer-path data/tokenizers/charTokenizer \
    --save-path "$SAVE_PATH" \
    --model-type DT \
    --dataset-type DT_SMILES \
    --config-path "$CONFIG_PATH" \
    --final-ckpt-name unsorted_grouped_large_rsa_final.pth

{
    echo
    echo "Finished at: $(date --iso-8601=seconds)"
    echo "Output files:"
    find "$SAVE_PATH" -maxdepth 2 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort
} | tee -a "$RUN_INFO_PATH"
