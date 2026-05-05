#!/usr/bin/env bash

set -euo pipefail

# Core experiment choices.
BACKBONES_CSV="${BACKBONES_CSV:-gastronet}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-ce}"
HEAD_TYPE="${HEAD_TYPE:-linear}"
ENABLE_SMOTE="${ENABLE_SMOTE:-0}"

# Training lengths.
INITIAL_PRETRAIN_EPOCHS="${INITIAL_PRETRAIN_EPOCHS:-${PRETRAIN_EPOCHS:-20}}"
ROI_BOOTSTRAP_FINETUNE_EPOCHS="${ROI_BOOTSTRAP_FINETUNE_EPOCHS:-20}"
ROI_REPRETRAIN_EPOCHS="${ROI_REPRETRAIN_EPOCHS:-${INITIAL_PRETRAIN_EPOCHS}}"
FINAL_FINETUNE_EPOCHS="${FINAL_FINETUNE_EPOCHS:-${FINETUNE_EPOCHS:-50}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"

# ROI controls.
FINAL_ENABLE_ROI_GUIDANCE="${FINAL_ENABLE_ROI_GUIDANCE:-1}"
ROI_START_EPOCH="${ROI_START_EPOCH:-20}"
ROI_FOCUS_PROB="${ROI_FOCUS_PROB:-1.0}"
ROI_CONTEXT_SCALE="${ROI_CONTEXT_SCALE:-2.0}"
ROI_MIN_CROP_SCALE="${ROI_MIN_CROP_SCALE:-0.4}"
ROI_CENTER_JITTER="${ROI_CENTER_JITTER:-0.05}"
ROI_GRADCAM_THRESHOLD="${ROI_GRADCAM_THRESHOLD:-0.6}"
ROI_GRADCAM_MIN_PROB="${ROI_GRADCAM_MIN_PROB:-0.5}"
GRADCAM_EXPORT_BATCH_SIZE="${GRADCAM_EXPORT_BATCH_SIZE:-8}"

# Head architecture details.
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-1}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"

# Optional feature-specific configs.
SMOTE_CONFIG_PATH="${SMOTE_CONFIG_PATH:-./smote_config.sh}"
CLASSIFIER_INPUT="${CLASSIFIER_INPUT:-}"
FINETUNE_TRAIN_MODE="${FINETUNE_TRAIN_MODE:-}"

# Data, outputs, and logging.
DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-../data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-../data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/roi_curriculum}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
WANDB_GROUP="${WANDB_GROUP:-roi-curriculum}"
ROI_RECORDS_DIR="${ROI_RECORDS_DIR:-${SAVE_DIR}/roi_records}"

# Runtime and system knobs.
NUM_WORKERS="${NUM_WORKERS:-10}"
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"
mkdir -p "${SAVE_DIR}" "${ROI_RECORDS_DIR}"

if [ "${ENABLE_SMOTE}" = "1" ]; then
    if [ ! -f "${SMOTE_CONFIG_PATH}" ]; then
        echo "SMOTE is enabled but the config file was not found: ${SMOTE_CONFIG_PATH}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${SMOTE_CONFIG_PATH}"
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

    printf '%s\n' "${args[@]}"
}

append_smote_args() {
    local -n ref_args=$1
    ref_args+=(
        --classifier-input "${CLASSIFIER_INPUT}"
        --finetune-train-mode "${FINETUNE_TRAIN_MODE}"
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
    if [ -n "${SMOTE_KNN_NEIGHBORS}" ]; then
        ref_args+=(--smote-knn-neighbors "${SMOTE_KNN_NEIGHBORS}")
    fi
    if [ "${ENABLE_SMOTE_FILTER}" = "1" ]; then
        ref_args+=(--smote-energy-filter)
    fi
    if [ "${ENABLE_SMOTE_KNN_FILTER}" = "1" ]; then
        ref_args+=(--smote-knn-filter)
        if [ "${SMOTE_KNN_CENTER_AWARE}" = "1" ]; then
            ref_args+=(--smote-knn-center-aware)
        fi
    fi
    if [ "${ENABLE_SMOTE_REFINE}" = "1" ]; then
        ref_args+=(
            --smote-energy-refine-steps "${SMOTE_REFINE_STEPS}"
            --smote-energy-refine-step-size "${SMOTE_REFINE_STEP_SIZE}"
        )
    fi
}

run_python() {
    echo
    echo "Running: $*"
    "$@"
}

build_finetune_experiment_id() {
    local backbone="$1"
    local variant_name="$2"
    local resolved_classifier_input="${CLASSIFIER_INPUT:-pooled}"
    local resolved_finetune_train_mode="${FINETUNE_TRAIN_MODE:-last_block}"
    local suffix=""

    if [ "${ENABLE_SMOTE}" = "1" ]; then
        suffix="_${SMOTE_FEATURE_SPACE}_smote"
        if [ -n "${SMOTE_SYNTHETIC_RATIO:-}" ]; then
            suffix="${suffix}_r${SMOTE_SYNTHETIC_RATIO}"
        fi
        if [ "${resolved_finetune_train_mode}" != "probe" ]; then
            suffix="${suffix}_warmstart_${resolved_finetune_train_mode}"
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

    printf '%s\n' "${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}_${resolved_classifier_input}_${resolved_finetune_train_mode}${suffix}_${variant_name}"
}

run_pretrain_stage() {
    local backbone="$1"
    local experiment_id="$2"
    local epochs="$3"
    local roi_records_path="${4:-}"

    mapfile -t args < <(build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${experiment_id}" "${epochs}")
    if [ -n "${roi_records_path}" ]; then
        args+=(--roi-records-path "${roi_records_path}")
    fi
    run_python "${PYTHON_BIN}" train.py "${args[@]}"
}

run_finetune_stage() {
    local backbone="$1"
    local experiment_id="$2"
    local epochs="$3"
    local encoder_ckpt="$4"
    local enable_roi="$5"

    mapfile -t args < <(build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${experiment_id}" "${epochs}")
    args+=(
        --encoder-ckpt "${encoder_ckpt}"
        --post-train-gradcam
        --post-train-gradcam-dataset-root "${POST_TRAIN_GRADCAM_DATASET_ROOT}"
    )
    if [ "${enable_roi}" = "1" ]; then
        args+=(
            --roi-guided-training
            --roi-start-epoch "${ROI_START_EPOCH}"
            --roi-focus-prob "${ROI_FOCUS_PROB}"
            --roi-context-scale "${ROI_CONTEXT_SCALE}"
            --roi-min-crop-scale "${ROI_MIN_CROP_SCALE}"
            --roi-center-jitter "${ROI_CENTER_JITTER}"
            --roi-gradcam-threshold "${ROI_GRADCAM_THRESHOLD}"
            --roi-gradcam-min-prob "${ROI_GRADCAM_MIN_PROB}"
        )
    fi
    if [ "${ENABLE_SMOTE}" = "1" ]; then
        append_smote_args args
    fi
    run_python "${PYTHON_BIN}" train.py "${args[@]}"
}

export_roi_records() {
    local checkpoint_path="$1"
    local output_path="$2"

    run_python "${PYTHON_BIN}" export_train_rois.py \
        --checkpoint "${checkpoint_path}" \
        --data-dir "${DATA_DIR}" \
        --output-path "${output_path}" \
        --batch-size "${GRADCAM_EXPORT_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --roi-gradcam-threshold "${ROI_GRADCAM_THRESHOLD}" \
        --roi-gradcam-min-prob "${ROI_GRADCAM_MIN_PROB}"
}

for backbone in "${BACKBONES[@]}"; do
    echo
    echo "==============================================="
    echo "ROI curriculum pipeline for backbone=${backbone}"
    echo "==============================================="

    initial_pretrain_id="${backbone}_pretrain_${PRETRAIN_LOSS}_roi_curriculum_initial"
    bootstrap_finetune_id="$(build_finetune_experiment_id "${backbone}" "roi_bootstrap")"
    roi_repretrain_id="${backbone}_pretrain_${PRETRAIN_LOSS}_roi_curriculum_guided"
    final_variant_name="ROIrun"
    if [ "${FINAL_ENABLE_ROI_GUIDANCE}" != "1" ]; then
        final_variant_name="final_plain"
    fi
    final_finetune_id="$(build_finetune_experiment_id "${backbone}" "${final_variant_name}")"

    initial_encoder_ckpt="${SAVE_DIR}/${initial_pretrain_id}_encoder.pt"
    bootstrap_best_ckpt="${SAVE_DIR}/${bootstrap_finetune_id}_best.pt"
    bootstrap_final_ckpt="${SAVE_DIR}/${bootstrap_finetune_id}_final.pt"
    roi_records_path="${ROI_RECORDS_DIR}/${backbone}_train_rois_roi_curriculum.json"
    roi_encoder_ckpt="${SAVE_DIR}/${roi_repretrain_id}_encoder.pt"

    echo "[1/5] Initial pretraining -> ${initial_pretrain_id}"
    run_pretrain_stage "${backbone}" "${initial_pretrain_id}" "${INITIAL_PRETRAIN_EPOCHS}"

    echo "[2/5] Bootstrap finetuning for Grad-CAM ROI generation -> ${bootstrap_finetune_id}"
    run_finetune_stage "${backbone}" "${bootstrap_finetune_id}" "${ROI_BOOTSTRAP_FINETUNE_EPOCHS}" "${initial_encoder_ckpt}" "0"

    bootstrap_source_ckpt="${bootstrap_best_ckpt}"
    if [ ! -f "${bootstrap_source_ckpt}" ]; then
        bootstrap_source_ckpt="${bootstrap_final_ckpt}"
    fi
    if [ ! -f "${bootstrap_source_ckpt}" ]; then
        echo "Could not find bootstrap finetune checkpoint for ROI export." >&2
        echo "Checked: ${bootstrap_best_ckpt} and ${bootstrap_final_ckpt}" >&2
        exit 1
    fi

    echo "[3/5] Exporting train-split Grad-CAM ROIs -> ${roi_records_path}"
    export_roi_records "${bootstrap_source_ckpt}" "${roi_records_path}"

    echo "[4/5] ROI-guided re-pretraining -> ${roi_repretrain_id}"
    run_pretrain_stage "${backbone}" "${roi_repretrain_id}" "${ROI_REPRETRAIN_EPOCHS}" "${roi_records_path}"

    echo "[5/5] Final finetuning -> ${final_finetune_id}"
    run_finetune_stage "${backbone}" "${final_finetune_id}" "${FINAL_FINETUNE_EPOCHS}" "${roi_encoder_ckpt}" "${FINAL_ENABLE_ROI_GUIDANCE}"
done

echo
echo "ROI curriculum pipeline complete."
