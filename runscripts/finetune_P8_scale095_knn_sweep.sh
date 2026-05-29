#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/finetune_P8_scale095_knn_sweep-%j.out

# Finetune stage: KNN sweep on P8_scale095 pretrain encoders
# Creates two finetune stages:
#   1. KNN with 20 neighbors
#   2. KNN with 3 neighbors
# Builds ensembles from both folds for each configuration.

set -euo pipefail
mkdir -p slurm_logs

if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then export WANDB_API_KEY; fi
fi

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

# Configuration
ENCODER_FOLD0="./checkpoints/P8_scale095/pretrain/fold0_val_center_1/P8_scale095_pretrain_fold0_val_center_1_encoder.pt"
ENCODER_FOLD1="./checkpoints/P8_scale095/pretrain/fold1_val_center_2/P8_scale095_pretrain_fold1_val_center_2_encoder.pt"
ENCODER_CKPTS="${ENCODER_FOLD0},${ENCODER_FOLD1}"

BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"
INPUT_SIZE="${INPUT_SIZE:-336}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
NUM_FOLDS="${NUM_FOLDS:-2}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"

# Verify encoder checkpoints exist
if [ ! -f "${ENCODER_FOLD0}" ]; then
    echo "ERROR: Encoder checkpoint not found: ${ENCODER_FOLD0}" >&2
    exit 1
fi
if [ ! -f "${ENCODER_FOLD1}" ]; then
    echo "ERROR: Encoder checkpoint not found: ${ENCODER_FOLD1}" >&2
    exit 1
fi

echo "=============================================================="
echo "P8_scale095 KNN Sweep Finetune"
echo "=============================================================="
echo "Fold 0 encoder: ${ENCODER_FOLD0}"
echo "Fold 1 encoder: ${ENCODER_FOLD1}"
echo "Backbone: ${BACKBONE_PRESET}"
echo "Input size: ${INPUT_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Data dir: ${DATA_DIR}"
echo "Num folds: ${NUM_FOLDS}"
echo "=============================================================="
echo ""

# Stage 1: KNN with 20 neighbors
echo "[1/2] Running finetune stage: KNN k=20"
echo "=============================================================="
EXPERIMENT_ID="P8_scale095_knn20"
SAVE_DIR="./checkpoints/${EXPERIMENT_ID}"
mkdir -p "${SAVE_DIR}"

python train.py \
    --stage finetune \
    --loco \
    --num-folds "${NUM_FOLDS}" \
    --fold-index 0 \
    --backbone-preset "${BACKBONE_PRESET}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --head-types knn \
    --knn-neighbors 20 \
    --encoder-ckpt "${ENCODER_CKPTS}" \
    --data-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --wandb-project "RARE25-Project" \
    --wandb-group "clean-baseline-knn-sweep" \
    --wandb-mode offline

echo ""
echo "Finetune KNN k=20 complete. Outputs in: ${SAVE_DIR}/"
echo ""

# Stage 2: KNN with 3 neighbors
echo "[2/2] Running finetune stage: KNN k=3"
echo "=============================================================="
EXPERIMENT_ID="P8_scale095_knn3"
SAVE_DIR="./checkpoints/${EXPERIMENT_ID}"
mkdir -p "${SAVE_DIR}"

python train.py \
    --stage finetune \
    --loco \
    --num-folds "${NUM_FOLDS}" \
    --fold-index 0 \
    --backbone-preset "${BACKBONE_PRESET}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --head-types knn \
    --knn-neighbors 3 \
    --encoder-ckpt "${ENCODER_CKPTS}" \
    --data-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --wandb-project "RARE25-Project" \
    --wandb-group "clean-baseline-knn-sweep" \
    --wandb-mode offline

echo ""
echo "Finetune KNN k=3 complete. Outputs in: ${SAVE_DIR}/"
echo ""

echo "=============================================================="
echo "All finetune stages complete!"
echo "=============================================================="
echo ""
echo "Ensemble models created:"
echo "  - checkpoints/P8_scale095_knn20/P8_scale095_knn20_ensembles/ensemble_knn20.pt"
echo "  - checkpoints/P8_scale095_knn3/P8_scale095_knn3_ensembles/ensemble_knn3.pt"
echo ""
echo "To submit an ensemble:"
echo "  cp checkpoints/P8_scale095_knn20/P8_scale095_knn20_ensembles/ensemble_knn20.pt model.pt"
echo "  docker build -t team-internship:latest -f Submission_files/Dockerfile ."
echo "=============================================================="
