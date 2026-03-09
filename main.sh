wandb login

python3 train.py \
    --data-dir ./data/Challenge_train_data \
    --centers center_1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov3-center1-training" \
    --backbone-name vit_base_patch16_dinov3 \
    --pretrained
