GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
EXPERIMENT_ID="${EXPERIMENT_ID:-Gastronet_DINOv2_linear_probe_336_gradcam_test}"

python3 train.py \
    --data-dir ./data/Challenge_train_data \
    --batch-size 64 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "${EXPERIMENT_ID}" \
    --backbone-name vit_base_patch14_reg4_dinov2 \
    --backbone-weights-path "${GASTRONET_CKPT}" \
    --input-size 336 \
    --post-train-gradcam \
    --post-train-gradcam-dataset-root ./data/EVC_Barretts_FullSet
