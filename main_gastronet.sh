GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"

python3 train.py \
    --data-dir ./data/Challenge_train_data \
    --batch-size 64 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Gastronet_DINOv2_linear_probe_224" \
    --backbone-name vit_base_patch14_reg4_dinov2 \
    --backbone-weights-path "${GASTRONET_CKPT}" \
    --input-size 224
