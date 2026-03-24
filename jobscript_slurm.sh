#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=slurm_trainmodel/slurm-%j.out

set -euo pipefail
mkdir -p slurm_trainmodel

if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
    fi
fi

# Load Python module
module load 2023 # adjust to available Python module

# Create/activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Cache Hugging Face downloads outside the repo and reuse them across jobs.
export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-main.sh}"
TIMM_PRELOAD_MODEL="${TIMM_PRELOAD_MODEL:-}"

# Download timm pretrained weights ahead of time when the selected run needs them.
if [ -n "${TIMM_PRELOAD_MODEL}" ]; then
python - <<'PY'
import os
import timm

model_name = os.environ["TIMM_PRELOAD_MODEL"]
print(f"Ensuring pretrained weights for {model_name} are cached in the Hugging Face cache...")
timm.create_model(model_name, pretrained=True, num_classes=0)
print("Pretrained weights ready.")
PY
fi

# Run training
/bin/bash "${TRAIN_SCRIPT}"
