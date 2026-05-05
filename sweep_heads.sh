#!/usr/bin/env bash
# sweep_heads.sh: Sweep 3 head types on gastronet/dinov3 for supmin/suppro (finetune only)
set -euo pipefail

HEAD_TYPES=(linear mlp ln-linear)
BACKBONES=(gastronet dinov3)
PRETRAIN_LOSSES=(supmin suppro)
FINETUNE_LOSS=ce
FINETUNE_EPOCHS=20
BATCH_SIZE=32
LR=1e-4
WARMUP_EPOCHS=3
SEED=42
SAVE_DIR_BASE=./checkpoints/sweep_heads

for HEAD_TYPE in "${HEAD_TYPES[@]}"; do
  for BACKBONE in "${BACKBONES[@]}"; do
    for PRETRAIN_LOSS in "${PRETRAIN_LOSSES[@]}"; do
      PRETRAIN_CKPT="./checkpoints/${BACKBONE}_pretrain_${PRETRAIN_LOSS}_encoder.pt"
      if [ ! -f "$PRETRAIN_CKPT" ]; then
        echo "[SKIP] Pretrain checkpoint not found: $PRETRAIN_CKPT"
        continue
      fi
      EXP_ID="${BACKBONE}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}"
      SAVE_DIR="${SAVE_DIR_BASE}/${EXP_ID}"
      mkdir -p "$SAVE_DIR"
      echo "[RUN] Backbone: $BACKBONE | Pretrain: $PRETRAIN_LOSS | Head: $HEAD_TYPE"
      HEAD_TYPE="$HEAD_TYPE" BACKBONES_CSV="$BACKBONE" PRETRAIN_LOSS="$PRETRAIN_LOSS" FINETUNE_LOSS="$FINETUNE_LOSS" FINETUNE_EPOCHS="$FINETUNE_EPOCHS" BATCH_SIZE="$BATCH_SIZE" LR="$LR" WARMUP_EPOCHS="$WARMUP_EPOCHS" SEED="$SEED" SAVE_DIR="$SAVE_DIR" ./main.sh --stage finetune --encoder-ckpt "$PRETRAIN_CKPT"
    done
  done
done
