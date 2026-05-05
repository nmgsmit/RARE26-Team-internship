#!/usr/bin/env bash

set -euo pipefail

# Edit these first when launching a head sweep.
EXPERIMENT_ID="${EXPERIMENT_ID:-sweep_heads}"
WANDB_GROUP="${WANDB_GROUP:-sweep-heads}"

HEAD_TYPES=(linear ln_linear mlp_fullwidth)
BACKBONES=(gastronet dinov3)
PRETRAIN_LOSSES=(supmin suppro)

FINETUNE_LOSS="${FINETUNE_LOSS:-ce}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-./checkpoints/sweep_heads}"

for HEAD_TYPE in "${HEAD_TYPES[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for PRETRAIN_LOSS in "${PRETRAIN_LOSSES[@]}"; do
            PRETRAIN_CKPT="./checkpoints/${BACKBONE}_pretrain_${PRETRAIN_LOSS}_encoder.pt"
            if [ ! -f "${PRETRAIN_CKPT}" ]; then
                echo "[SKIP] Pretrain checkpoint not found: ${PRETRAIN_CKPT}"
                continue
            fi

            RUN_SAVE_DIR="${SAVE_DIR_BASE}/${BACKBONE}_${PRETRAIN_LOSS}_${HEAD_TYPE}"
            mkdir -p "${RUN_SAVE_DIR}"

            echo "[RUN] Backbone: ${BACKBONE} | Pretrain: ${PRETRAIN_LOSS} | Head: ${HEAD_TYPE}"
            EXPERIMENT_ID="${EXPERIMENT_ID}" \
            WANDB_GROUP="${WANDB_GROUP}" \
            HEAD_TYPE="${HEAD_TYPE}" \
            BACKBONES_CSV="${BACKBONE}" \
            PRETRAIN_LOSS="${PRETRAIN_LOSS}" \
            FINETUNE_LOSS="${FINETUNE_LOSS}" \
            PRETRAIN_CHECKPOINT="${PRETRAIN_CKPT}" \
            FORCE_PRETRAIN=0 \
            FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" \
            BATCH_SIZE="${BATCH_SIZE}" \
            LR="${LR}" \
            WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
            SEED="${SEED}" \
            SAVE_DIR="${RUN_SAVE_DIR}" \
            ./main.sh
        done
    done
done
