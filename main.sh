#!/usr/bin/env bash

set -euo pipefail

# Core experiment choices: these usually matter most.
BACKBONES_CSV="${BACKBONES_CSV:-gastronet,dinov3}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-ce}"
HEAD_TYPE="${HEAD_TYPE:-linear}"

# Optimization and training length.
BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"

# Head architecture details.
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-1}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"

CLASSIFIER_INPUT="${CLASSIFIER_INPUT:-}"
FINETUNE_TRAIN_MODE="${FINETUNE_TRAIN_MODE:-}"

# Data, outputs, and logging.
DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-../data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-../data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/linear_suppro_dual_backbone}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
WANDB_GROUP="${WANDB_GROUP:-supcon}"

# Runtime and system knobs.
NUM_WORKERS="${NUM_WORKERS:-10}"
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"

mkdir -p "${SAVE_DIR}"



add_optional_arg() {
    local -n ref_args=$1
    local flag="$2"
    local value="$3"
    if [ -n "${value}" ]; then
        ref_args+=("${flag}" "${value}")
    fi
}

build_common_args() {
    local stage="$1"
    local backbone="$2"
    local loss_name="$3"
    local experiment_id="$4"
    local epochs="$5"

    local args=(
        --stage "${stage}"
        --loss-name "${loss_name}"
        --data-dir "${DATA_DIR}"
        --testset-images-dir "${TESTSET_IMAGES_DIR}"
        --batch-size "${BATCH_SIZE}"
        --epochs "${epochs}"
        --lr "${LR}"
        --warmup-epochs "${WARMUP_EPOCHS}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --experiment-id "${experiment_id}"
        --save-dir "${SAVE_DIR}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --backbone-preset "${backbone}"
        --head-type "${HEAD_TYPE}"
        --head-dropout "${HEAD_DROPOUT}"
        --mlp-hidden-layers "${MLP_HIDDEN_LAYERS}"
        --mlp-dropout "${MLP_DROPOUT}"
    )

    add_optional_arg args --head-hidden-dim "${HEAD_HIDDEN_DIM}"
    add_optional_arg args --mlp-hidden-dim "${MLP_HIDDEN_DIM}"
    add_optional_arg args --classifier-input "${CLASSIFIER_INPUT}"
    add_optional_arg args --finetune-train-mode "${FINETUNE_TRAIN_MODE}"

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

resolve_encoder_checkpoint() {
    local backbone="$1"
    local pretrain_experiment_id="$2"
    local primary_ckpt="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

    local candidates=(
        "${primary_ckpt}"
        "./checkpoints/${pretrain_experiment_id}_encoder.pt"
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
    pretrain_experiment_id="${backbone}_pretrain_${PRETRAIN_LOSS}"
    encoder_ckpt="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

    if [ "${FORCE_PRETRAIN}" != "1" ] && resolved_encoder_ckpt="$(resolve_encoder_checkpoint "${backbone}" "${pretrain_experiment_id}")"; then
        encoder_ckpt="${resolved_encoder_ckpt}"
        echo
        echo "Found existing pretrained encoder checkpoint: ${encoder_ckpt}"
        echo "Skipping pretraining and reusing the existing encoder. Set FORCE_PRETRAIN=1 to retrain it."
    else
        if [ "${FORCE_PRETRAIN}" = "1" ]; then
            echo
            echo "FORCE_PRETRAIN=1, so pretraining will run even if an encoder checkpoint already exists."
        fi
        mapfile -t pretrain_args < <(
            build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
        )
        run_python_train "${pretrain_args[@]}"
    fi

    finetune_experiment_id="${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}"
    mapfile -t finetune_args < <(
        build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}"
    )
    finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
    run_python_train "${finetune_args[@]}"
done
