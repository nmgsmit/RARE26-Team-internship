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
CLASSIFIER_INPUT="${CLASSIFIER_INPUT:-}"
FINETUNE_TRAIN_MODE="${FINETUNE_TRAIN_MODE:-}"
ENABLE_SMOTE="${ENABLE_SMOTE:-0}"
ENABLE_SMOTE_FILTER="${ENABLE_SMOTE_FILTER:-0}"
ENABLE_SMOTE_KNN_FILTER="${ENABLE_SMOTE_KNN_FILTER:-0}"
ENABLE_SMOTE_REFINE="${ENABLE_SMOTE_REFINE:-0}"
SMOTE_FEATURE_SPACE="${SMOTE_FEATURE_SPACE:-projection}"
SMOTE_NEIGHBORS="${SMOTE_NEIGHBORS:-3}"
SMOTE_SAMPLING_STRATEGY="${SMOTE_SAMPLING_STRATEGY:-minority}"
SMOTE_SYNTHETIC_RATIO="${SMOTE_SYNTHETIC_RATIO:-0.10}"
SMOTE_WARMSTART_EPOCHS="${SMOTE_WARMSTART_EPOCHS:-3}"
SMOTE_REFINE_STEPS="${SMOTE_REFINE_STEPS:-3}"
SMOTE_REFINE_STEP_SIZE="${SMOTE_REFINE_STEP_SIZE:-0.01}"
SMOTE_ENERGY_EPOCHS="${SMOTE_ENERGY_EPOCHS:-25}"
SMOTE_ENERGY_LR="${SMOTE_ENERGY_LR:-1e-3}"
SMOTE_ENERGY_WEIGHT_DECAY="${SMOTE_ENERGY_WEIGHT_DECAY:-1e-4}"
SMOTE_ENERGY_BATCH_SIZE="${SMOTE_ENERGY_BATCH_SIZE:-256}"
SMOTE_ENERGY_HIDDEN_DIM="${SMOTE_ENERGY_HIDDEN_DIM:-256}"
SMOTE_ENERGY_LAYERS="${SMOTE_ENERGY_LAYERS:-2}"
SMOTE_ENERGY_DROPOUT="${SMOTE_ENERGY_DROPOUT:-0.1}"
SMOTE_ENERGY_QUANTILE="${SMOTE_ENERGY_QUANTILE:-0.80}"
SMOTE_ENERGY_NOISE_STD="${SMOTE_ENERGY_NOISE_STD:-0.15}"
SMOTE_ENERGY_NOISE_COPIES="${SMOTE_ENERGY_NOISE_COPIES:-2}"
SMOTE_ENERGY_MAJORITY_RATIO="${SMOTE_ENERGY_MAJORITY_RATIO:-1.0}"
SMOTE_ENERGY_REFINE_ANCHOR_WEIGHT="${SMOTE_ENERGY_REFINE_ANCHOR_WEIGHT:-5.0}"
SMOTE_ENERGY_REFINE_MARGIN_WEIGHT="${SMOTE_ENERGY_REFINE_MARGIN_WEIGHT:-2.0}"
SMOTE_ENERGY_REFINE_TARGET_MARGIN="${SMOTE_ENERGY_REFINE_TARGET_MARGIN:-0.05}"
SMOTE_KNN_NEIGHBORS="${SMOTE_KNN_NEIGHBORS:-}"
SMOTE_KNN_SUPPORT_QUANTILE="${SMOTE_KNN_SUPPORT_QUANTILE:-0.50}"
SMOTE_KNN_MINORITY_PURITY="${SMOTE_KNN_MINORITY_PURITY:-1.0}"
SMOTE_KNN_MARGIN="${SMOTE_KNN_MARGIN:-0.02}"
SMOTE_KNN_CENTER_AWARE="${SMOTE_KNN_CENTER_AWARE:-0}"

GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"

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

build_finetune_suffix() {
    local suffix=""
    if [ "${ENABLE_SMOTE}" = "1" ]; then
        suffix="_${SMOTE_FEATURE_SPACE}_smote"
        if [ "${FINETUNE_TRAIN_MODE:-probe}" != "probe" ]; then
            suffix="${suffix}_warmstart_${FINETUNE_TRAIN_MODE:-last_block}"
        fi
        if [ "${ENABLE_SMOTE_FILTER}" = "1" ]; then
            suffix="${suffix}_energy"
        fi
        if [ "${ENABLE_SMOTE_KNN_FILTER}" = "1" ]; then
            suffix="${suffix}_knn"
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

    finetune_suffix="$(build_finetune_suffix)"
    finetune_experiment_id="${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}${finetune_suffix}"
    mapfile -t finetune_args < <(
        build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}"
    )
    finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
    if [ "${ENABLE_SMOTE}" = "1" ]; then
        finetune_args+=(
            --classifier-input "${CLASSIFIER_INPUT:-projection}"
            --finetune-train-mode "${FINETUNE_TRAIN_MODE:-probe}"
            --finetune-with-smote
            --smote-feature-space "${SMOTE_FEATURE_SPACE}"
            --smote-neighbors "${SMOTE_NEIGHBORS}"
            --smote-sampling-strategy "${SMOTE_SAMPLING_STRATEGY}"
            --smote-synthetic-ratio "${SMOTE_SYNTHETIC_RATIO}"
            --smote-warmstart-epochs "${SMOTE_WARMSTART_EPOCHS}"
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
            --smote-energy-refine-anchor-weight "${SMOTE_ENERGY_REFINE_ANCHOR_WEIGHT}"
            --smote-energy-refine-margin-weight "${SMOTE_ENERGY_REFINE_MARGIN_WEIGHT}"
            --smote-energy-refine-target-margin "${SMOTE_ENERGY_REFINE_TARGET_MARGIN}"
            --smote-knn-support-quantile "${SMOTE_KNN_SUPPORT_QUANTILE}"
            --smote-knn-minority-purity "${SMOTE_KNN_MINORITY_PURITY}"
            --smote-knn-margin "${SMOTE_KNN_MARGIN}"
        )
        add_optional_arg finetune_args --smote-knn-neighbors "${SMOTE_KNN_NEIGHBORS}"
        if [ "${ENABLE_SMOTE_FILTER}" = "1" ]; then
            finetune_args+=(--smote-energy-filter)
        fi
        if [ "${ENABLE_SMOTE_KNN_FILTER}" = "1" ]; then
            finetune_args+=(--smote-knn-filter)
            if [ "${SMOTE_KNN_CENTER_AWARE}" = "1" ]; then
                finetune_args+=(--smote-knn-center-aware)
            fi
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
