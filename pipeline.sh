#!/usr/bin/env bash
# pipeline.sh: Full workflow automation for pretrain → finetune → GradCAM ROI → pretrain (ROI) → finetune (ROI)
set -euo pipefail

# --- CONFIGURABLE PARAMETERS ---
BACKBONES_CSV="gastronet,dinov3"
PRETRAIN_LOSS="suppro"
FINETUNE_LOSS="ce"
HEAD_TYPE="linear"
BATCH_SIZE=32
PRETRAIN_EPOCHS=20
FINETUNE_EPOCHS=20
LR=1e-4
WARMUP_EPOCHS=3
SEED=42
SAVE_DIR="./checkpoints/pipeline"
ROI_RECORDS_PATH="./checkpoints/pipeline/gradcam_rois.json"


# Always use fixed encoder checkpoint for all main.sh calls
export ENCODER_CKPT_OVERRIDE="./checkpoints/linear_suppro_dual_backbone/gastronet_pretrain_suppro_encoder.pt"

# --- 1. INITIAL PRETRAINING ---
echo "[1/5] Pretraining (no ROI)"
export EXPERIMENT_ID_SUFFIX="pretrain_noROI"
export ENABLE_ROI_GUIDANCE=0
export RUN_COMPARISON_BASELINE=0
export SAVE_DIR="$SAVE_DIR"
export BACKBONES_CSV="$BACKBONES_CSV"
export PRETRAIN_LOSS="$PRETRAIN_LOSS"
export FINETUNE_LOSS="$FINETUNE_LOSS"
export HEAD_TYPE="$HEAD_TYPE"
export BATCH_SIZE="$BATCH_SIZE"
export PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS"
export FINETUNE_EPOCHS="$FINETUNE_EPOCHS"
export LR="$LR"
export WARMUP_EPOCHS="$WARMUP_EPOCHS"
export SEED="$SEED"
./main.sh

# --- 2. FINETUNE FOR GRADCAM ROI GENERATION ---
echo "[2/5] Finetuning for GradCAM ROI generation (no ROI)"
export EXPERIMENT_ID_SUFFIX="finetune_noROI"
export ENABLE_ROI_GUIDANCE=0
export RUN_COMPARISON_BASELINE=0
export SAVE_DIR="$SAVE_DIR"
export BACKBONES_CSV="$BACKBONES_CSV"
export PRETRAIN_LOSS="$PRETRAIN_LOSS"
export FINETUNE_LOSS="$FINETUNE_LOSS"
export HEAD_TYPE="$HEAD_TYPE"
export BATCH_SIZE="$BATCH_SIZE"
export PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS"
export FINETUNE_EPOCHS="$FINETUNE_EPOCHS"
export LR="$LR"
export WARMUP_EPOCHS="$WARMUP_EPOCHS"
export SEED="$SEED"
./main.sh

# --- 3. SAVE GRADCAM ROIS ---
echo "[3/5] Saving GradCAM ROIs"
# (Assumes main.sh saves GradCAM ROIs automatically after finetune. If not, add code here to extract and save ROIs.)

# --- 4. SECOND PRETRAINING USING GRADCAM ROIS ---
echo "[4/5] Pretraining using GradCAM ROIs"
export EXPERIMENT_ID_SUFFIX="pretrain_ROI"
export ENABLE_ROI_GUIDANCE=1
export RUN_COMPARISON_BASELINE=0
export SAVE_DIR="$SAVE_DIR"
export BACKBONES_CSV="$BACKBONES_CSV"
export PRETRAIN_LOSS="$PRETRAIN_LOSS"
export FINETUNE_LOSS="$FINETUNE_LOSS"
export HEAD_TYPE="$HEAD_TYPE"
export BATCH_SIZE="$BATCH_SIZE"
export PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS"
export FINETUNE_EPOCHS="$FINETUNE_EPOCHS"
export LR="$LR"
export WARMUP_EPOCHS="$WARMUP_EPOCHS"
export SEED="$SEED"
export PRETRAIN_ROI_RECORDS_PATH="$ROI_RECORDS_PATH"
./main.sh

# --- 5. FINAL FINETUNING WITH ROI GUIDANCE ---
echo "[5/5] Final finetuning with ROI guidance"
export EXPERIMENT_ID_SUFFIX="finetune_ROI"
export ENABLE_ROI_GUIDANCE=1
export RUN_COMPARISON_BASELINE=0
export SAVE_DIR="$SAVE_DIR"
export BACKBONES_CSV="$BACKBONES_CSV"
export PRETRAIN_LOSS="$PRETRAIN_LOSS"
export FINETUNE_LOSS="$FINETUNE_LOSS"
export HEAD_TYPE="$HEAD_TYPE"
export BATCH_SIZE="$BATCH_SIZE"
export PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS"
export FINETUNE_EPOCHS="$FINETUNE_EPOCHS"
export LR="$LR"
export WARMUP_EPOCHS="$WARMUP_EPOCHS"
export SEED="$SEED"
export PRETRAIN_ROI_RECORDS_PATH="$ROI_RECORDS_PATH"
./main.sh

echo "Pipeline complete."
