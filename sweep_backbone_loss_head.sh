#!/usr/bin/env bash

set -euo pipefail

# One-click sweep for:
# - backbones: gastronet, dinov3
# - pretrain losses: suppro, supmin
# - heads: linear, mlp_fullwidth
#
# For each backbone/loss pair:
# 1. reuse an existing pretrained encoder if available
# 2. otherwise run pretraining once
# 3. finetune both heads from that encoder

RUN_PREFIX="${RUN_PREFIX:-Baseline}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_PREFIX}}"

BACKBONES=(gastronet dinov3)
PRETRAIN_LOSSES=(suppro supmin)
HEAD_TYPES=(linear mlp_fullwidth)

FINETUNE_LOSS="${FINETUNE_LOSS:-class-balanced}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-./checkpoints/${RUN_PREFIX}}"

find_pretrain_ckpt() {
    local backbone="$1"
    local pretrain_loss="$2"
    local run_id="$3"
    local save_dir="$4"

    local candidates=(
        "${save_dir}/${run_id}_encoder.pt"
        "./checkpoints/${run_id}_encoder.pt"
        "./checkpoints/${backbone}_pretrain_${pretrain_loss}_encoder.pt"
    )

    for candidate in "${candidates[@]}"; do
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

for BACKBONE in "${BACKBONES[@]}"; do
    for PRETRAIN_LOSS in "${PRETRAIN_LOSSES[@]}"; do
        PRETRAIN_RUN_ID="${RUN_PREFIX}_${BACKBONE}_${PRETRAIN_LOSS}_pretrain"
        PRETRAIN_SAVE_DIR="${SAVE_DIR_BASE}/${BACKBONE}/${PRETRAIN_LOSS}"
        mkdir -p "${PRETRAIN_SAVE_DIR}"
        if PRETRAIN_CKPT="$(find_pretrain_ckpt "${BACKBONE}" "${PRETRAIN_LOSS}" "${PRETRAIN_RUN_ID}" "${PRETRAIN_SAVE_DIR}")"; then
            echo
            echo "[REUSE] Backbone: ${BACKBONE} | Loss: ${PRETRAIN_LOSS} | Checkpoint: ${PRETRAIN_CKPT}"
        else
            echo
            echo "[PRETRAIN] Backbone: ${BACKBONE} | Loss: ${PRETRAIN_LOSS}"
            EXPERIMENT_ID_EXACT="${PRETRAIN_RUN_ID}" \
            WANDB_GROUP="${WANDB_GROUP}" \
            BACKBONES_CSV="${BACKBONE}" \
            PRETRAIN_LOSS="${PRETRAIN_LOSS}" \
            FORCE_PRETRAIN=0 \
            BATCH_SIZE="${BATCH_SIZE}" \
            PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" \
            FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" \
            LR="${LR}" \
            WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
            SEED="${SEED}" \
            SAVE_DIR="${PRETRAIN_SAVE_DIR}" \
            ./main.sh

            PRETRAIN_CKPT="${PRETRAIN_SAVE_DIR}/${PRETRAIN_RUN_ID}_encoder.pt"
            if [ ! -f "${PRETRAIN_CKPT}" ]; then
                echo "Expected pretrained checkpoint not found: ${PRETRAIN_CKPT}" >&2
                exit 1
            fi
        fi

        for HEAD_TYPE in "${HEAD_TYPES[@]}"; do
            SHORT_HEAD="${HEAD_TYPE}"
            if [ "${HEAD_TYPE}" = "mlp_fullwidth" ]; then
                SHORT_HEAD="mlp"
            fi

            FINETUNE_RUN_ID="${RUN_PREFIX}_${BACKBONE}_${PRETRAIN_LOSS}_${SHORT_HEAD}"
            FINETUNE_SAVE_DIR="${SAVE_DIR_BASE}/${BACKBONE}/${PRETRAIN_LOSS}/${HEAD_TYPE}"
            mkdir -p "${FINETUNE_SAVE_DIR}"

            echo
            echo "[FINETUNE] Backbone: ${BACKBONE} | Pretrain: ${PRETRAIN_LOSS} | Head: ${HEAD_TYPE}"
            EXPERIMENT_ID_EXACT="${FINETUNE_RUN_ID}" \
            WANDB_GROUP="${WANDB_GROUP}" \
            HEAD_TYPE="${HEAD_TYPE}" \
            BACKBONES_CSV="${BACKBONE}" \
            PRETRAIN_LOSS="${PRETRAIN_LOSS}" \
            FINETUNE_LOSS="${FINETUNE_LOSS}" \
            PRETRAIN_CHECKPOINT="${PRETRAIN_CKPT}" \
            FORCE_PRETRAIN=0 \
            BATCH_SIZE="${BATCH_SIZE}" \
            PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" \
            FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" \
            LR="${LR}" \
            WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
            SEED="${SEED}" \
            SAVE_DIR="${FINETUNE_SAVE_DIR}" \
            ./main.sh
        done
    done
done
