SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../data/Challenge_train_data}"

python3 train.py \
    --data-dir "$DATA_DIR" \
    --centers center_1 \
    --debug-center1-balanced \
    --debug-class-count 61 \
    --batch-size 64 \
    --epochs 2 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov3-center1-training" \
    --backbone-name vit_base_patch16_dinov3 \
    --no-pretrained
