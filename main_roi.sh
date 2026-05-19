#!/usr/bin/env bash
# Test ROI-sampled crops (roi_focus_prob=1.0) at different scales with 2 folds
# Compares ROI-centered crops at fixed scales: 0.4, 0.8, 1.0

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

echo "Testing ROI-sampled crops at different scales (2 folds each)"
echo "============================================================"

# Common settings for all runs
export CV_NUM_FOLDS=2
export WANDB_GROUP="roi_crops_ablation"
export CHECKPOINT_ROOT_DIR="./checkpoints"
export ROI_FOCUS_PROB=1.0  # All crops are ROI-guided (no random fallback)
export ROI_NEGATIVE_FOCUS_PROB=0.5  # 50% of negatives use hard-negative ROI crops
export ROI_CENTER_JITTER=0.05
export ROI_MAX_ASPECT_RATIO=1.5
export ROI_CONTEXT_SCALE=2.0

# Test 1: ROI crops at 0.4 scale
echo ""
echo "Test 1: ROI-sampled crops at 0.4 scale"
echo "========================================"
export EXPERIMENT_ID="roi_0.4"
export EXPERIMENT_SAVE_SUBDIR="roi_crops_ablation/0.4"
export ROI_MIN_CROP_SCALE=0.4
export ROI_MAX_CROP_SCALE=0.4
/bin/bash main.sh

# Test 2: ROI crops at 0.8 scale
echo ""
echo "Test 2: ROI-sampled crops at 0.8 scale"
echo "========================================"
export EXPERIMENT_ID="roi_0.8"
export EXPERIMENT_SAVE_SUBDIR="roi_crops_ablation/0.8"
export ROI_MIN_CROP_SCALE=0.8
export ROI_MAX_CROP_SCALE=0.8
/bin/bash main.sh

# Test 3: ROI crops at 1.0 scale (full image)
echo ""
echo "Test 3: ROI-sampled crops at 1.0 scale (full image)"
echo "==================================================="
export EXPERIMENT_ID="roi_1.0"
export EXPERIMENT_SAVE_SUBDIR="roi_crops_ablation/1.0"
export ROI_MIN_CROP_SCALE=1.0
export ROI_MAX_CROP_SCALE=1.0
/bin/bash main.sh

echo ""
echo "All ROI-sampled crop tests completed!"
