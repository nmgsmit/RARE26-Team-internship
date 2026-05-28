#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --output=slurm_trainmodel/slurm_loco_final-%j.out

# End-to-end LOCO pipeline:
#   1. SupPro pretrain, one encoder per center (LOCO at pretrain time)
#   2. Finetune linear-norm head, one model per center, auto-load matching encoder
#   3. Ensemble + weight-averaged submission .pt
#
# The submission artifact you upload lives at:
#   ${CHECKPOINT_ROOT}/finetune/${RUN_TAG}_submission.pt
# with calibrated threshold in:
#   ${CHECKPOINT_ROOT}/finetune/${RUN_TAG}_submission.json

set -euo pipefail
mkdir -p slurm_trainmodel

# -----------------------------------------------------------------------------
# Environment setup (matches jobscript_slurm_roi.sh conventions)
# -----------------------------------------------------------------------------
if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then
        export WANDB_API_KEY
    fi
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

# Required by torch.use_deterministic_algorithms(True) when running CUDA >= 10.2.
# Without this, deterministic cuBLAS matmul (used inside the SupPro loss) crashes.
# main.sh sets this for the legacy path; we set it here for the LOCO jobscript.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# -----------------------------------------------------------------------------
# Run configuration — change RUN_TAG between experiments to keep wandb tidy
# -----------------------------------------------------------------------------
RUN_TAG="loco_v3_$(date +%Y%m%d_%H%M)"
WANDB_GROUP="loco_v3"
WANDB_PROJECT="RARE25-Project"
CHECKPOINT_ROOT="./checkpoints/${RUN_TAG}"
SEED=42

PRETRAIN_SAVE_DIR="${CHECKPOINT_ROOT}/pretrain"
FINETUNE_SAVE_DIR="${CHECKPOINT_ROOT}/finetune"

# Encoder path template — {fold_index} and {holdout_center} get substituted by
# train.py's LOCO orchestrator. The path on the right matches what LOCO pretrain
# writes out.
ENCODER_CKPT_TEMPLATE="${PRETRAIN_SAVE_DIR}/fold{fold_index}_val_{holdout_center}/${RUN_TAG}_pretrain_fold{fold_index}_val_{holdout_center}_encoder.pt"

# -----------------------------------------------------------------------------
# Stage 1 — SupPro pretrain, one encoder per held-out center
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "Stage 1/2 | LOCO SupPro pretrain | RUN_TAG=${RUN_TAG}"
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
    --roi-min-crop-scale 0.4 \
    --roi-max-crop-scale 1.0 \
    --roi-center-jitter 0.05 \
    --roi-max-aspect-ratio 1.5 \
    --num-workers 10 \
    --seed "${SEED}" \
    --save-dir "${PRETRAIN_SAVE_DIR}"

# -----------------------------------------------------------------------------
# Stage 2 — Supervised finetune of LN-linear head + auto ensemble + submission
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "Stage 2/2 | LOCO finetune + ensemble + submission .pt"
echo "Encoder template: ${ENCODER_CKPT_TEMPLATE}"
echo "================================================================="

python train.py \
    --stage finetune \
    --loss-name label-smoothed-ce \
    --label-smoothing 0.05 \
    --experiment-id "${RUN_TAG}_finetune" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --backbone-preset gastronet \
    --backbone-weights-path ../Gastronet/dinov2.pth \
    --head-type ln_linear \
    --loco \
    --batch-size 32 \
    --epochs 30 \
    --finetune-lr 2e-4 \
    --warmup-epochs 3 \
    --augmentation-intensity 3 \
    --roi-focus-prob 0.5 \
    --roi-negative-focus-prob 0.0 \
    --roi-warmup-epochs 5 \
    --roi-context-scale 2.0 \
    --roi-min-crop-scale 0.4 \
    --roi-max-crop-scale 1.0 \
    --roi-center-jitter 0.05 \
    --roi-max-aspect-ratio 1.5 \
    --num-workers 10 \
    --seed "${SEED}" \
    --save-dir "${FINETUNE_SAVE_DIR}" \
    --encoder-ckpt "${ENCODER_CKPT_TEMPLATE}" \
    --post-train-gradcam

# -----------------------------------------------------------------------------
# Summary — where to find what you'll submit
# -----------------------------------------------------------------------------
SUBMISSION_PT="${FINETUNE_SAVE_DIR}/${RUN_TAG}_finetune_submission.pt"
SUBMISSION_JSON="${FINETUNE_SAVE_DIR}/${RUN_TAG}_finetune_submission.json"

echo ""
echo "================================================================="
echo "Pipeline finished."
echo "  Submission checkpoint : ${SUBMISSION_PT}"
echo "  Submission metadata   : ${SUBMISSION_JSON}"
echo "  Ensemble predictions  : ${FINETUNE_SAVE_DIR}/${RUN_TAG}_finetune_ensemble.npz"
echo ""
echo "Inspect with:"
echo "  cat ${SUBMISSION_JSON}"
echo "================================================================="
