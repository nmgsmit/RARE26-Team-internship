#!/bin/bash
set -euo pipefail

# Usage:
#   /bin/bash mainTEST.sh /path/to/checkpoint.pt
# or set MODEL_PATH in environment before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
IMAGES_DIR="${IMAGES_DIR:-$SCRIPT_DIR/data/EVC_Barretts_FullSet/images}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_base_patch16_dinov3}"
THRESHOLD="${THRESHOLD:-0.5}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:-0}"

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
echo "  IMAGES_DIR=$IMAGES_DIR"
echo "  IMAGE_SIZE=$IMAGE_SIZE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BACKBONE_NAME=$BACKBONE_NAME"
echo "  THRESHOLD=$THRESHOLD"
echo "  PRETRAINED_FLAG=$PRETRAINED_FLAG"

CMD=(
    python3 TestResult.py
    --model-path "$MODEL_PATH"
    --images-dir "$IMAGES_DIR"
    --image-size "$IMAGE_SIZE"
    --batch-size "$BATCH_SIZE"
    --backbone-name "$BACKBONE_NAME"
    --threshold "$THRESHOLD"
)

if [ "$PRETRAINED_FLAG" = "1" ]; then
    CMD+=(--pretrained)
fi

"${CMD[@]}"
