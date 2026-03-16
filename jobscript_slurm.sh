#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=slurm_trainmodel/slurm-%j.out
#SBATCH --error=slurm_trainmodel/slurm-%j.err

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

# JOB_MODE controls what this script executes.
# Train (default):
#   sbatch jobscript_slurm.sh
# Test on checkpoint:
#   sbatch --export=ALL,JOB_MODE=test,MODEL_PATH=./checkpoints/your_model.pt jobscript_slurm.sh
# Optional test overrides:
#   IMAGES_DIR=./data/EVC_Barretts_FullSet/images
#   IMAGE_SIZE=224
#   BATCH_SIZE=32
#   BACKBONE_NAME=vit_base_patch16_dinov3
#   THRESHOLD=0.5
#   PRETRAINED_FLAG=0
JOB_MODE="${JOB_MODE:-train}"

if [ "$JOB_MODE" = "test" ]; then
    if [ -z "${MODEL_PATH:-}" ]; then
        echo "ERROR: JOB_MODE=test requires MODEL_PATH."
        echo "Submit with: sbatch --export=ALL,JOB_MODE=test,MODEL_PATH=./checkpoints/<your_model>.pt jobscript_slurm.sh"
        exit 1
    fi
    /bin/bash mainTEST.sh "$MODEL_PATH"
else
    /bin/bash main.sh
fi
