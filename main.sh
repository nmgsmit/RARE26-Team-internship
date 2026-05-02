#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-../data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-../data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/linear_suppro_dual_backbone}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
WANDB_GROUP="${WANDB_GROUP:-supcon}"

BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-ce}"

HEAD_TYPE="${HEAD_TYPE:-linear}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-1}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"
ENABLE_SMOTE="${ENABLE_SMOTE:-0}"
ENABLE_SMOTE_FILTER="${ENABLE_SMOTE_FILTER:-0}"
ENABLE_SMOTE_REFINE="${ENABLE_SMOTE_REFINE:-0}"
SMOTE_NEIGHBORS="${SMOTE_NEIGHBORS:-3}"
SMOTE_SAMPLING_STRATEGY="${SMOTE_SAMPLING_STRATEGY:-minority}"
SMOTE_SYNTHETIC_RATIO="${SMOTE_SYNTHETIC_RATIO:-0.50}"
SMOTE_REFINE_STEPS="${SMOTE_REFINE_STEPS:-5}"
SMOTE_REFINE_STEP_SIZE="${SMOTE_REFINE_STEP_SIZE:-0.05}"
SMOTE_ENERGY_EPOCHS="${SMOTE_ENERGY_EPOCHS:-25}"
SMOTE_ENERGY_LR="${SMOTE_ENERGY_LR:-1e-3}"
SMOTE_ENERGY_WEIGHT_DECAY="${SMOTE_ENERGY_WEIGHT_DECAY:-1e-4}"
SMOTE_ENERGY_BATCH_SIZE="${SMOTE_ENERGY_BATCH_SIZE:-256}"
SMOTE_ENERGY_HIDDEN_DIM="${SMOTE_ENERGY_HIDDEN_DIM:-256}"
SMOTE_ENERGY_LAYERS="${SMOTE_ENERGY_LAYERS:-2}"
SMOTE_ENERGY_DROPOUT="${SMOTE_ENERGY_DROPOUT:-0.1}"
SMOTE_ENERGY_QUANTILE="${SMOTE_ENERGY_QUANTILE:-0.95}"
SMOTE_ENERGY_NOISE_STD="${SMOTE_ENERGY_NOISE_STD:-0.15}"
SMOTE_ENERGY_NOISE_COPIES="${SMOTE_ENERGY_NOISE_COPIES:-2}"
SMOTE_ENERGY_MAJORITY_RATIO="${SMOTE_ENERGY_MAJORITY_RATIO:-1.0}"

GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BACKBONES_CSV="${BACKBONES_CSV:-gastronet,dinov3}"
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

build_finetune_suffix() {
    local suffix=""
    if [ "${ENABLE_SMOTE}" = "1" ]; then
        suffix="_smote"
        if [ "${ENABLE_SMOTE_FILTER}" = "1" ]; then
            suffix="${suffix}_filter"
        fi
        if [ "${ENABLE_SMOTE_REFINE}" = "1" ]; then
            suffix="${suffix}_refine"
        fi
    fi
    printf '%s\n' "${suffix}"
}

for backbone in "${BACKBONES[@]}"; do
    pretrain_experiment_id="${backbone}_pretrain_${PRETRAIN_LOSS}"
    encoder_ckpt="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

    mapfile -t pretrain_args < <(
        build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
    )
    run_python_train "${pretrain_args[@]}"

    finetune_suffix="$(build_finetune_suffix)"
    finetune_experiment_id="${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}${finetune_suffix}"
    mapfile -t finetune_args < <(
        build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}"
    )
    finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
    if [ "${ENABLE_SMOTE}" = "1" ]; then
        finetune_args+=(
            --finetune-with-smote
            --smote-neighbors "${SMOTE_NEIGHBORS}"
            --smote-sampling-strategy "${SMOTE_SAMPLING_STRATEGY}"
            --smote-synthetic-ratio "${SMOTE_SYNTHETIC_RATIO}"
            --smote-energy-epochs "${SMOTE_ENERGY_EPOCHS}"
            --smote-energy-lr "${SMOTE_ENERGY_LR}"
            --smote-energy-weight-decay "${SMOTE_ENERGY_WEIGHT_DECAY}"
            --smote-energy-batch-size "${SMOTE_ENERGY_BATCH_SIZE}"
            --smote-energy-hidden-dim "${SMOTE_ENERGY_HIDDEN_DIM}"
            --smote-energy-layers "${SMOTE_ENERGY_LAYERS}"
            --smote-energy-dropout "${SMOTE_ENERGY_DROPOUT}"
            --smote-energy-threshold-quantile "${SMOTE_ENERGY_QUANTILE}"
            --smote-energy-noise-std "${SMOTE_ENERGY_NOISE_STD}"
            --smote-energy-noise-copies "${SMOTE_ENERGY_NOISE_COPIES}"
            --smote-energy-majority-ratio "${SMOTE_ENERGY_MAJORITY_RATIO}"
        )
        if [ "${ENABLE_SMOTE_FILTER}" = "1" ]; then
            finetune_args+=(--smote-energy-filter)
        fi
        if [ "${ENABLE_SMOTE_REFINE}" = "1" ]; then
            finetune_args+=(
                --smote-energy-refine-steps "${SMOTE_REFINE_STEPS}"
                --smote-energy-refine-step-size "${SMOTE_REFINE_STEP_SIZE}"
            )
        fi
    fi
    run_python_train "${finetune_args[@]}"
done
