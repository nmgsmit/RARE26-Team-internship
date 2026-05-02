#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-../data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-../data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/gastronet_head_smote_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
WANDB_GROUP="${WANDB_GROUP:-gastronet_suppro_head_smote_sweep}"

BATCH_SIZE="${BATCH_SIZE:-32}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"
FINETUNE_LOSS="${FINETUNE_LOSS:-ce}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PRETRAINED_ENCODER_CKPT="${PRETRAINED_ENCODER_CKPT:-./checkpoints/gastronet_pretrain_suppro_encoder.pt}"

HEADS_CSV="${HEADS_CSV:-linear,mlp_fullwidth}"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-1}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"

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

IFS=',' read -r -a HEADS <<< "${HEADS_CSV}"

mkdir -p "${SAVE_DIR}"

if [ ! -f "${PRETRAINED_ENCODER_CKPT}" ]; then
    echo "Pretrained encoder checkpoint not found: ${PRETRAINED_ENCODER_CKPT}" >&2
    echo "Set PRETRAINED_ENCODER_CKPT to the correct path before running this sweep." >&2
    exit 1
fi

add_optional_arg() {
    local -n ref_args=$1
    local flag="$2"
    local value="$3"
    if [ -n "${value}" ]; then
        ref_args+=("${flag}" "${value}")
    fi
}

build_common_args() {
    local experiment_id="$1"
    local head_type="$2"

    local args=(
        --stage "finetune"
        --loss-name "${FINETUNE_LOSS}"
        --encoder-ckpt "${PRETRAINED_ENCODER_CKPT}"
        --data-dir "${DATA_DIR}"
        --testset-images-dir "${TESTSET_IMAGES_DIR}"
        --batch-size "${BATCH_SIZE}"
        --epochs "${FINETUNE_EPOCHS}"
        --lr "${LR}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --experiment-id "${experiment_id}"
        --save-dir "${SAVE_DIR}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --backbone-preset "${BACKBONE_PRESET}"
        --backbone-weights-path "${GASTRONET_CKPT}"
        --head-type "${head_type}"
        --post-train-gradcam
        --post-train-gradcam-dataset-root "${POST_TRAIN_GRADCAM_DATASET_ROOT}"
    )

    if [ "${head_type}" = "mlp_fullwidth" ]; then
        args+=(--mlp-hidden-layers "${MLP_HIDDEN_LAYERS}" --mlp-dropout "${MLP_DROPOUT}")
        add_optional_arg args --mlp-hidden-dim "${MLP_HIDDEN_DIM}"
    fi

    printf '%s\n' "${args[@]}"
}

append_smote_args() {
    local -n ref_args=$1
    ref_args+=(
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
}

run_python_train() {
    local -a args=("$@")
    echo
    echo "Running: ${PYTHON_BIN} train.py ${args[*]}"
    "${PYTHON_BIN}" train.py "${args[@]}"
}

for head_type in "${HEADS[@]}"; do
    head_label="${head_type}"

    mapfile -t clean_args < <(
        build_common_args "gastronet_suppro_${head_label}_clean" "${head_type}"
    )
    run_python_train "${clean_args[@]}"

    mapfile -t smote_args < <(
        build_common_args "gastronet_suppro_${head_label}_smote" "${head_type}"
    )
    append_smote_args smote_args
    run_python_train "${smote_args[@]}"

    mapfile -t smote_filter_refine_args < <(
        build_common_args "gastronet_suppro_${head_label}_smote_filter_refine" "${head_type}"
    )
    append_smote_args smote_filter_refine_args
    smote_filter_refine_args+=(
        --smote-energy-filter
        --smote-energy-refine-steps "${SMOTE_REFINE_STEPS}"
        --smote-energy-refine-step-size "${SMOTE_REFINE_STEP_SIZE}"
    )
    run_python_train "${smote_filter_refine_args[@]}"
done
