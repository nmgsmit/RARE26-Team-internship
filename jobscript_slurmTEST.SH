#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00

set -euo pipefail

# Optional if you want to keep wandb available in environment.
if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
    fi
fi

# Required: set MODEL_PATH before submission, for example:
# sbatch --export=ALL,MODEL_PATH=./checkpoints/dinov3-center1-training_best.pt jobscript_slurmTEST.SH
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is not set."
    echo "Submit with: sbatch --export=ALL,MODEL_PATH=./checkpoints/<your_model>.pt jobscript_slurmTEST.SH"
    exit 1
fi

# Optional overrides via --export=ALL,...
#   IMAGES_DIR=../data/EVC_Barretts_FullSet/images
#   BATCH_SIZE=32
#   IMAGE_SIZE=224
#   BACKBONE_NAME=vit_base_patch16_dinov3
#   SAVE_CSV=results/evc_test_predictions.csv

module load 2023

if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

/bin/bash mainTEST.sh "$MODEL_PATH"
