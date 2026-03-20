python3 train.py \
    --data-dir ./data/Challenge_train_data \
    --batch-size 64 \
    --epochs 20 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Dinov3_Focal_wrs_baseline_(Linear_head)" \
    --backbone-name vit_base_patch16_dinov3.lvd1689m
