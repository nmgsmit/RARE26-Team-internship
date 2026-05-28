#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/pretrain-%j.out

# Two-fold LOCO SupPro pretrain on Gastronet DINOv2.
# Override knobs via env vars on the sbatch line, e.g.:
#   EXPERIMENT_ID=my_run BATCH_SIZE=64 sbatch runscripts/jobscript_slurm_pretrain.sh

set -euo pipefail
mkdir -p slurm_logs

# ── W&B auth ─────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then export WANDB_API_KEY; fi
fi

# ── Environment ──────────────────────────────────────────────────────────────
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

# ── Run configuration ────────────────────────────────────────────────────────
EXPERIMENT_ID="${EXPERIMENT_ID:-clean_baseline}"
WANDB_GROUP="${WANDB_GROUP:-clean-baseline}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-../Gastronet/dinov2.pth}"
INPUT_SIZE="${INPUT_SIZE:-336}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"
PRETRAIN_BACKBONE_LR="${PRETRAIN_BACKBONE_LR:-1e-5}"
PRETRAIN_PROJ_LR="${PRETRAIN_PROJ_LR:-3e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
TEMPERATURE="${TEMPERATURE:-0.1}"
BASE_TEMPERATURE="${BASE_TEMPERATURE:-0.07}"
POS_RATIO="${POS_RATIO:-0.2}"
AUGMENTATION_INTENSITY="${AUGMENTATION_INTENSITY:-3}"
DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
NUM_FOLDS="${NUM_FOLDS:-2}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/${EXPERIMENT_ID}}"

mkdir -p "${SAVE_DIR}"

echo "=============================================================="
echo "Clean-baseline pretrain | ${EXPERIMENT_ID}"
echo "Backbone: ${BACKBONE_PRESET} (${BACKBONE_WEIGHTS})"
echo "LOCO ${NUM_FOLDS} folds | batch ${BATCH_SIZE} | pos_ratio ${POS_RATIO}"
echo "T=${TEMPERATURE} backbone_lr=${PRETRAIN_BACKBONE_LR} proj_lr=${PRETRAIN_PROJ_LR}"
echo "=============================================================="

python train.py \
    --stage pretrain \
    --loco \
    --num-folds "${NUM_FOLDS}" \
    --fold-index 0 \
    --backbone-preset "${BACKBONE_PRESET}" \
    --backbone-weights-path "${BACKBONE_WEIGHTS}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --pretrain-backbone-lr "${PRETRAIN_BACKBONE_LR}" \
    --pretrain-proj-lr "${PRETRAIN_PROJ_LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --temperature "${TEMPERATURE}" \
    --base-temperature "${BASE_TEMPERATURE}" \
    --balanced-sampler \
    --pos-ratio "${POS_RATIO}" \
    --augmentation-intensity "${AUGMENTATION_INTENSITY}" \
    --data-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}"

echo ""
echo "Encoders saved under ${SAVE_DIR}/"
