#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/finetune_P8_flexible-%j.out

# Flexible finetune script for P8_scale095 encoders
#
# Required:
#   ENCODER_FOLD0 - Path to fold 0 encoder
#   ENCODER_FOLD1 - Path to fold 1 encoder
#   KNN_K_VALUES  - Comma-separated KNN k values to sweep (e.g., "3,20")
#
# Optional:
#   EXPERIMENT_ID, BACKBONE_PRESET, INPUT_SIZE, BATCH_SIZE, DATA_DIR, NUM_WORKERS, SEED
#
# Usage:
#   ENCODER_FOLD0=path/to/fold0_encoder.pt \
#   ENCODER_FOLD1=path/to/fold1_encoder.pt \
#   KNN_K_VALUES=3,20 \
#   sbatch runscripts/finetune_P8_encoders_flexible.sh
#
# Or with all options:
#   ENCODER_FOLD0=path/fold0_encoder.pt \
#   ENCODER_FOLD1=path/fold1_encoder.pt \
#   KNN_K_VALUES=5,10,20 \
#   EXPERIMENT_ID=my_experiment \
#   BACKBONE_PRESET=gastronet \
#   INPUT_SIZE=336 \
#   BATCH_SIZE=32 \
#   sbatch runscripts/finetune_P8_encoders_flexible.sh

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

# Verify required arguments
if [ -z "${ENCODER_FOLD0:-}" ] || [ -z "${ENCODER_FOLD1:-}" ] || [ -z "${KNN_K_VALUES:-}" ]; then
    echo "ERROR: Required environment variables not set:" >&2
    echo "  ENCODER_FOLD0 - Path to fold 0 encoder checkpoint" >&2
    echo "  ENCODER_FOLD1 - Path to fold 1 encoder checkpoint" >&2
    echo "  KNN_K_VALUES  - Comma-separated KNN k values (e.g., '3,20')" >&2
    exit 1
fi

# Verify encoder checkpoints exist
if [ ! -f "${ENCODER_FOLD0}" ]; then
    echo "ERROR: Fold 0 encoder not found: ${ENCODER_FOLD0}" >&2
    exit 1
fi
if [ ! -f "${ENCODER_FOLD1}" ]; then
    echo "ERROR: Fold 1 encoder not found: ${ENCODER_FOLD1}" >&2
    exit 1
fi

# Configuration with defaults
ENCODER_CKPTS="${ENCODER_FOLD0},${ENCODER_FOLD1}"
EXPERIMENT_PREFIX="${EXPERIMENT_ID:-P8_scale095}"
BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"
INPUT_SIZE="${INPUT_SIZE:-336}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
NUM_FOLDS="${NUM_FOLDS:-2}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"

echo "=============================================================="
echo "P8 Encoders - Flexible KNN Sweep Finetune"
echo "=============================================================="
echo "Fold 0 encoder: ${ENCODER_FOLD0}"
echo "Fold 1 encoder: ${ENCODER_FOLD1}"
echo "KNN k values: ${KNN_K_VALUES}"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "Backbone: ${BACKBONE_PRESET}"
echo "Input size: ${INPUT_SIZE}"
echo "=============================================================="
echo ""

# Convert comma-separated k values to array
IFS=',' read -ra K_ARRAY <<< "${KNN_K_VALUES}"
TOTAL_K=${#K_ARRAY[@]}
CURRENT_K=0

# Run finetune for each k value
for K in "${K_ARRAY[@]}"; do
    CURRENT_K=$((CURRENT_K + 1))
    EXPERIMENT_ID="${EXPERIMENT_PREFIX}_knn${K}"
    SAVE_DIR="./checkpoints/${EXPERIMENT_ID}"
    mkdir -p "${SAVE_DIR}"

    echo "[${CURRENT_K}/${TOTAL_K}] Running finetune stage: KNN k=${K}"
    echo "=============================================================="

    python train.py \
        --stage finetune \
        --loco \
        --num-folds "${NUM_FOLDS}" \
        --fold-index 0 \
        --backbone-preset "${BACKBONE_PRESET}" \
        --input-size "${INPUT_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --head-types knn \
        --knn-neighbors "${K}" \
        --encoder-ckpt "${ENCODER_CKPTS}" \
        --data-dir "${DATA_DIR}" \
        --num-workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --save-dir "${SAVE_DIR}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --wandb-project "RARE25-Project" \
        --wandb-group "clean-baseline-knn-sweep" \
        --wandb-mode offline

    echo "Finetune KNN k=${K} complete. Outputs in: ${SAVE_DIR}/"
    echo ""
done

echo "=============================================================="
echo "All KNN finetune stages complete!"
echo "=============================================================="
echo ""
echo "Ensemble models created:"
for K in "${K_ARRAY[@]}"; do
    EXPERIMENT_ID="${EXPERIMENT_PREFIX}_knn${K}"
    echo "  - checkpoints/${EXPERIMENT_ID}/${EXPERIMENT_ID}_ensembles/ensemble_knn${K}.pt"
done
echo ""
echo "To submit an ensemble, copy it to model.pt and build the Docker image:"
echo "  cp checkpoints/<experiment>/..._ensembles/ensemble_knn<k>.pt model.pt"
echo "  docker build -t team-internship:latest -f Submission_files/Dockerfile ."
echo "=============================================================="
