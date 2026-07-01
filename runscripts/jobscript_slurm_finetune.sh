#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/finetune-%j.out

# Per-fold sklearn-head fit + ensemble bundle build.
# Required: ENCODER_CKPTS (comma-separated paths, one per LOCO fold in order).
# Optional: HEAD_TYPES, KNN_NEIGHBORS, SVM_C  (comma-separated sweeps).
#
# Example:
#   ENCODER_CKPTS=path/fold0_encoder.pt,path/fold1_encoder.pt \
#       HEAD_TYPES=knn,svm \
#       KNN_NEIGHBORS=5,25,51 \
#       SVM_C=0.5,2,10 \
#       sbatch runscripts/jobscript_slurm_finetune.sh

set -euo pipefail
mkdir -p slurm_logs

if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then export WANDB_API_KEY; fi
    # W&B routing defaults from .env (ngmtue/rare26). A value passed on the
    # sbatch line (--export) still wins via :=.
    : "${WANDB_ENTITY:=$(grep '^WANDB_ENTITY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)}"
    : "${WANDB_PROJECT:=$(grep '^WANDB_PROJECT=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)}"
    export WANDB_ENTITY WANDB_PROJECT
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

EXPERIMENT_ID="${EXPERIMENT_ID:-clean_baseline}"
WANDB_GROUP="${WANDB_GROUP:-clean-baseline}"
WANDB_PROJECT="${WANDB_PROJECT:-rare26}"
BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"
INPUT_SIZE="${INPUT_SIZE:-336}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HEAD_TYPES="${HEAD_TYPES:-knn}"
KNN_NEIGHBORS="${KNN_NEIGHBORS:-5,25,51}"
SVM_C="${SVM_C:-0.5,2,10}"
DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
NUM_FOLDS="${NUM_FOLDS:-2}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/${EXPERIMENT_ID}}"

if [ -z "${ENCODER_CKPTS:-}" ]; then
    echo "ENCODER_CKPTS is required: comma-separated encoder.pt paths, one per fold." >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "=============================================================="
echo "Clean-baseline finetune | ${EXPERIMENT_ID}"
echo "Encoders : ${ENCODER_CKPTS}"
echo "Heads    : ${HEAD_TYPES}"
echo "KNN k    : ${KNN_NEIGHBORS}"
echo "SVM C    : ${SVM_C}"
echo "=============================================================="

python train.py \
    --stage finetune \
    --loco \
    --num-folds "${NUM_FOLDS}" \
    --fold-index 0 \
    --backbone-preset "${BACKBONE_PRESET}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --head-types "${HEAD_TYPES}" \
    --knn-neighbors "${KNN_NEIGHBORS}" \
    --svm-C "${SVM_C}" \
    --encoder-ckpt "${ENCODER_CKPTS}" \
    --data-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}"

echo ""
echo "Per-fold heads + ensemble bundles in ${SAVE_DIR}/"
