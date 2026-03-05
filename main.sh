wandb login

python3 train.py \
    --data-dir ###FILL IN FOR OUR DATA### \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training" \
