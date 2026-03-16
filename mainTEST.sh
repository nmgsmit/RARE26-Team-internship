#!/bin/bash
set -euo pipefail

# Usage:
#   /bin/bash mainTEST.sh /path/to/checkpoint.pt
# or set MODEL_PATH in environment before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"

# -------------------------------------------------------------------------------------------------
# Test settings: edit these defaults here, similar to main.sh for training.
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_base_patch16_dinov3}"
THRESHOLD="${THRESHOLD:-0.5}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:-0}"  # 1 -> --pretrained, 0 -> --no-pretrained
# -------------------------------------------------------------------------------------------------

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No model checkpoint path provided."
    echo "Pass it as first argument or set MODEL_PATH env variable."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

echo "Running evaluation with:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  IMAGES_DIR=./data/EVC_Barretts_FullSet/images"
echo "  IMAGE_SIZE=$IMAGE_SIZE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BACKBONE_NAME=$BACKBONE_NAME"
echo "  THRESHOLD=$THRESHOLD"
echo "  PRETRAINED_FLAG=$PRETRAINED_FLAG"

PRETRAINED_ARG="--no-pretrained"
if [ "$PRETRAINED_FLAG" = "1" ]; then
    PRETRAINED_ARG="--pretrained"
fi

python3 TestResult.py \
    --model-path "$MODEL_PATH" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --backbone-name "$BACKBONE_NAME" \
    --threshold "$THRESHOLD" \
    "$PRETRAINED_ARG"
