#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --output=slurm_trainmodel/slurm_pretrain-%j.out

# LOCO SupPro pretrain with symmetric view sampling and full scale range [0.4, 1.0].
# Saves one encoder per held-out center under:
#   ./checkpoints/best_model/pretrain/fold{i}_val_{center}/best_model_pretrain_fold{i}_val_{center}_encoder.pt
#
# Submit this first, then submit jobscript_slurm_finetune_heads.sh with
# --dependency=afterok:<this job id>.  Use submit_best_model.sh to do both at once.

set -euo pipefail
mkdir -p slurm_trainmodel

# ── W&B auth ──────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then
        export WANDB_API_KEY
    fi
fi

# ── Environment ───────────────────────────────────────────────────────────────
module load 2023

if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ── Crop scale ────────────────────────────────────────────────────────────────
# Both ROI-centered and random crops sample scale from [MIN_CROP_SCALE, 1.0].
# Set via MIN_CROP_SCALE env var from submit_best_model.sh.
MIN_CROP_SCALE="${MIN_CROP_SCALE:-0.4}"

# ── Paths ─────────────────────────────────────────────────────────────────────
RUN_TAG="best_model"
WANDB_GROUP="best_model"
WANDB_PROJECT="RARE25-Project"
PRETRAIN_SAVE_DIR="./checkpoints/${RUN_TAG}/pretrain"

mkdir -p "${PRETRAIN_SAVE_DIR}"

# ── Pretrain ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo "LOCO SupPro pretrain | RUN_TAG=${RUN_TAG}"
echo "Scale range [0.4, 1.0] | roi_focus_prob=0.5 (symmetric views)"
echo "================================================================="

python train.py \
    --stage pretrain \
    --loss-name suppro \
    --experiment-id "${RUN_TAG}_pretrain" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --backbone-preset gastronet \
    --backbone-weights-path ../Gastronet/dinov2.pth \
    --loco \
    --batch-size 32 \
    --epochs 30 \
    --pretrain-backbone-lr 1e-5 \
    --pretrain-proj-lr 3e-4 \
    --warmup-epochs 3 \
    --temperature 0.1 \
    --base-temperature 0.07 \
    --augmentation-intensity 3 \
    --roi-focus-prob 0.5 \
    --roi-negative-focus-prob 0.0 \
    --roi-warmup-epochs 5 \
    --roi-context-scale 2.0 \
    --roi-min-crop-scale "${MIN_CROP_SCALE}" \
    --roi-center-jitter 0.05 \
    --roi-max-aspect-ratio 1.5 \
    --roi-records-path "./roi_records/rois.json" \
    --num-workers 10 \
    --seed 42 \
    --save-dir "${PRETRAIN_SAVE_DIR}"

echo ""
echo "Pretrain complete. Encoders saved under ${PRETRAIN_SAVE_DIR}/"
