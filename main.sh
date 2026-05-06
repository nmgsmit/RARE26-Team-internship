#!/usr/bin/env bash

set -euo pipefail

# Fill these in first for new runs.
# This tag is added in front of the auto-generated stage/backbone experiment ids.
EXPERIMENT_ID="${EXPERIMENT_ID:-${EXPERIMENT_ID_PREFIX:-}}"
# Set this when you want to use the exact experiment id without auto-appending stage details.
EXPERIMENT_ID_EXACT="${EXPERIMENT_ID_EXACT:-}"
WANDB_GROUP="${WANDB_GROUP:-supcon}"

# Crucial model choices.
BACKBONES_CSV="${BACKBONES_CSV:-gastronet,dinov3}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-class-balanced}"
HEAD_TYPE="${HEAD_TYPE:-linear}"
TEMPERATURE="${TEMPERATURE:-0.07}"
BASE_TEMPERATURE="${BASE_TEMPERATURE:-0.07}"

# Checkpoint control. Leave PRETRAIN_CHECKPOINT blank to auto-detect one.
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-}"
# Standard is 0: reuse an existing checkpoint when possible.
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"

# Training and optimization.
BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"

# Shared paths and runtime defaults: these usually stay fixed across runs.
DATA_DIR="${DATA_DIR:-./data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-./data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-./data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/linear_suppro_dual_backbone}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
NUM_WORKERS="${NUM_WORKERS:-10}"
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"

if [ -n "${PRETRAIN_CHECKPOINT}" ] && [ "${#BACKBONES[@]}" -gt 1 ]; then
    echo "PRETRAIN_CHECKPOINT expects a single backbone run. Set BACKBONES_CSV to one backbone or leave PRETRAIN_CHECKPOINT blank." >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

build_common_args() {
    local stage="$1"
    local backbone="$2"
    local loss_name="$3"
    local experiment_id="$4"
    local epochs="$5"

    local args=(
        --stage "${stage}"
        --loss-name "${loss_name}"
        --experiment-id "${experiment_id}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --backbone-preset "${backbone}"
        --head-type "${HEAD_TYPE}"
        --data-dir "${DATA_DIR}"
        --testset-images-dir "${TESTSET_IMAGES_DIR}"
        --batch-size "${BATCH_SIZE}"
        --epochs "${epochs}"
        --lr "${LR}"
        --warmup-epochs "${WARMUP_EPOCHS}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --save-dir "${SAVE_DIR}"
        --temperature "${TEMPERATURE}"
        --base-temperature "${BASE_TEMPERATURE}"
    )

    if [ "${backbone}" = "gastronet" ]; then
        args+=(--backbone-weights-path "${GASTRONET_CKPT}")
    fi

    if [ "${stage}" = "finetune" ]; then
        args+=(--post-train-gradcam --post-train-gradcam-dataset-root "${POST_TRAIN_GRADCAM_DATASET_ROOT}")
    fi

    printf '%s\n' "${args[@]}"
}

run_python_train() {
    local -a args=("$@")
    echo
    echo "Running: ${PYTHON_BIN} train.py ${args[*]}"
    "${PYTHON_BIN}" train.py "${args[@]}"
}

build_experiment_id() {
    local base_id="$1"
    if [ -n "${EXPERIMENT_ID_EXACT}" ]; then
        printf '%s\n' "${EXPERIMENT_ID_EXACT}"
    elif [ -n "${EXPERIMENT_ID}" ]; then
        printf '%s_%s\n' "${EXPERIMENT_ID}" "${base_id}"
    else
        printf '%s\n' "${base_id}"
    fi
}

resolve_encoder_checkpoint() {
    local backbone="$1"
    local pretrain_experiment_id="$2"
    local base_pretrain_experiment_id="$3"
    local primary_ckpt="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

    local candidates=(
        "${primary_ckpt}"
        "./checkpoints/${pretrain_experiment_id}_encoder.pt"
        "./checkpoints/${base_pretrain_experiment_id}_encoder.pt"
        "./checkpoints/${backbone}_pretrain_${PRETRAIN_LOSS}_encoder.pt"
    )

    for candidate in "${candidates[@]}"; do
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

for backbone in "${BACKBONES[@]}"; do
    base_pretrain_experiment_id="${backbone}_pretrain_${PRETRAIN_LOSS}"
    pretrain_experiment_id="$(build_experiment_id "${base_pretrain_experiment_id}")"
    encoder_ckpt="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

    if [ "${FORCE_PRETRAIN}" = "1" ]; then
        echo
        echo "FORCE_PRETRAIN=1, so pretraining will run even if an encoder checkpoint already exists."
        mapfile -t pretrain_args < <(
            build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
        )
        run_python_train "${pretrain_args[@]}"
    elif [ -n "${PRETRAIN_CHECKPOINT}" ]; then
        if [ ! -f "${PRETRAIN_CHECKPOINT}" ]; then
            echo "Configured PRETRAIN_CHECKPOINT does not exist: ${PRETRAIN_CHECKPOINT}" >&2
            exit 1
        fi
        encoder_ckpt="${PRETRAIN_CHECKPOINT}"
        echo
        echo "Using PRETRAIN_CHECKPOINT from main.sh: ${encoder_ckpt}"
    elif resolved_encoder_ckpt="$(resolve_encoder_checkpoint "${backbone}" "${pretrain_experiment_id}" "${base_pretrain_experiment_id}")"; then
        encoder_ckpt="${resolved_encoder_ckpt}"
        echo
        echo "Found existing pretrained encoder checkpoint: ${encoder_ckpt}"
        echo "Skipping pretraining and reusing the existing encoder. Set FORCE_PRETRAIN=1 to retrain it."
    else
        mapfile -t pretrain_args < <(
            build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
        )
        run_python_train "${pretrain_args[@]}"
    fi

    base_finetune_experiment_id="${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}"
    finetune_experiment_id="$(build_experiment_id "${base_finetune_experiment_id}")"
    mapfile -t finetune_args < <(
        build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}"
    )
    finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
    run_python_train "${finetune_args[@]}"
done
