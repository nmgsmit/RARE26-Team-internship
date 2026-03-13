#!/bin/bash
set -euo pipefail

# Usage:
#   /bin/bash mainTEST.sh /path/to/checkpoint.pt
# or set MODEL_PATH in environment before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
IMAGES_DIR="${IMAGES_DIR:-"$SCRIPT_DIR/../data/EVC_Barretts_FullSet/images"}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_base_patch16_dinov3}"
SAVE_CSV="${SAVE_CSV:-results/evc_test_predictions.csv}"

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No model checkpoint path provided."
    echo "Pass it as first argument or set MODEL_PATH env variable."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

echo "Running evaluation with:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  IMAGES_DIR=$IMAGES_DIR"
echo "  BACKBONE_NAME=$BACKBONE_NAME"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  IMAGE_SIZE=$IMAGE_SIZE"
echo "  SAVE_CSV=$SAVE_CSV"

python3 TestResult.py \
    --model-path "$MODEL_PATH" \
    --images-dir "$IMAGES_DIR" \
    --batch-size "$BATCH_SIZE" \
    --image-size "$IMAGE_SIZE" \
    --backbone-name "$BACKBONE_NAME" \
    --save-csv "$SAVE_CSV"
