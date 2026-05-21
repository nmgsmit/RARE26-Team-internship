#!/usr/bin/env bash
# Finetune: KNN head on the encoder trained by main_pretrain.sh.
# Run main_pretrain.sh first.

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

PRETRAIN_SAVE_DIR="./checkpoints/best_model/pretrain"
ENCODER_CKPT="${PRETRAIN_SAVE_DIR}/best_model_pretrain_encoder.pt"

if [ ! -f "${ENCODER_CKPT}" ]; then
    echo "Encoder checkpoint not found: ${ENCODER_CKPT}" >&2
    echo "Run main_pretrain.sh first." >&2
    exit 1
fi

export CV_NUM_FOLDS=2
export WANDB_GROUP="best_model"
export CHECKPOINT_ROOT_DIR="./checkpoints"
export STAGES_CSV="finetune"
export FORCE_PRETRAIN=0
export PRETRAIN_CHECKPOINT="${ENCODER_CKPT}"

export HEAD_TYPE="knn"

export ROI_FOCUS_PROB=0.5
export ROI_NEGATIVE_FOCUS_PROB=0.0
export ROI_MIN_CROP_SCALE=0.4
export ROI_MAX_CROP_SCALE=1.0
export ROI_CENTER_JITTER=0.05
export ROI_MAX_ASPECT_RATIO=1.5
export ROI_CONTEXT_SCALE=2.0

export EXPERIMENT_ID="best_model_knn"
export EXPERIMENT_SAVE_SUBDIR="best_model/finetune/knn"

/bin/bash main.sh
