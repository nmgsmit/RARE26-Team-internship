#!/usr/bin/env bash
# Pretrain the encoder once with a full crop scale range [0.4, 1.0].
# Run this first, then run main_finetune_{head}.sh for each head.
#
# Scale range [0.4, 1.0] lets each view independently sample any zoom level,
# so the encoder learns scale invariance across the full spectrum rather than
# overfitting to one fixed scale.

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

export CV_NUM_FOLDS=2
export WANDB_GROUP="best_model"
export CHECKPOINT_ROOT_DIR="./checkpoints"
export STAGES_CSV="pretrain"

# Symmetric view sampling: each view independently uses ROI 50% of the time;
# negatives always get random crops.
export ROI_FOCUS_PROB=0.5
export ROI_NEGATIVE_FOCUS_PROB=0.0

# Full scale range: the encoder sees tight crops (0.4) through near-full-frame
# (1.0) during training, building genuine scale invariance.
export ROI_MIN_CROP_SCALE=0.4
export ROI_MAX_CROP_SCALE=1.0

export ROI_CENTER_JITTER=0.05
export ROI_MAX_ASPECT_RATIO=1.5
export ROI_CONTEXT_SCALE=2.0

export EXPERIMENT_ID="best_model"
export EXPERIMENT_SAVE_SUBDIR="best_model/pretrain"

/bin/bash main.sh
