#!/usr/bin/env bash

set -euo pipefail

# Match the main-branch launcher conventions so existing checkpoints are found the same way.
EXPERIMENT_ID="${EXPERIMENT_ID:-${EXPERIMENT_ID_PREFIX:-}}"
EXPERIMENT_ID_EXACT="${EXPERIMENT_ID_EXACT:-}"
WANDB_GROUP="${WANDB_GROUP:-roi-curriculum}"

BACKBONES_CSV="${BACKBONES_CSV:-gastronet,dinov3}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
ROI_PRETRAIN_LOSS="${ROI_PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-class-balanced}"
HEAD_TYPE="${HEAD_TYPE:-linear}"

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-}"
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"

BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
ROI_BOOTSTRAP_FINETUNE_EPOCHS="${ROI_BOOTSTRAP_FINETUNE_EPOCHS:-20}"
ROI_REPRETRAIN_EPOCHS="${ROI_REPRETRAIN_EPOCHS:-${PRETRAIN_EPOCHS}}"
FINAL_FINETUNE_EPOCHS="${FINAL_FINETUNE_EPOCHS:-${FINETUNE_EPOCHS:-20}}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-../data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-../data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/linear_suppro_dual_backbone}"
ROI_RECORDS_DIR="${ROI_RECORDS_DIR:-${SAVE_DIR}/roi_records}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
NUM_WORKERS="${NUM_WORKERS:-10}"
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ROI_GRADCAM_THRESHOLD="${ROI_GRADCAM_THRESHOLD:-0.6}"
ROI_GRADCAM_MIN_PROB="${ROI_GRADCAM_MIN_PROB:-0.5}"
GRADCAM_EXPORT_BATCH_SIZE="${GRADCAM_EXPORT_BATCH_SIZE:-8}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"

if [ -n "${PRETRAIN_CHECKPOINT}" ] && [ "${#BACKBONES[@]}" -gt 1 ]; then
    echo "PRETRAIN_CHECKPOINT expects a single backbone run. Set BACKBONES_CSV to one backbone or leave PRETRAIN_CHECKPOINT blank." >&2
    exit 1
fi
if [ -n "${EXPERIMENT_ID_EXACT}" ]; then
    echo "EXPERIMENT_ID_EXACT is not supported by pipeline.sh because each stage needs a distinct experiment id. Use EXPERIMENT_ID or EXPERIMENT_ID_PREFIX instead." >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}" "${ROI_RECORDS_DIR}"

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
        --batch-size "${BATCH_SIZE}"
        --epochs "${epochs}"
        --lr "${LR}"
        --warmup-epochs "${WARMUP_EPOCHS}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --save-dir "${SAVE_DIR}"
    )

    if [ "${backbone}" = "gastronet" ]; then
        args+=(--backbone-weights-path "${GASTRONET_CKPT}")
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

resolve_or_create_base_pretrain() {
    local backbone="$1"
    local base_pretrain_experiment_id="$2"
    local pretrain_experiment_id="$3"

    if [ "${FORCE_PRETRAIN}" = "1" ]; then
        echo "FORCE_PRETRAIN=1, so the base pretraining will run even if a checkpoint already exists."
        mapfile -t pretrain_args < <(
            build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
        )
        run_python_train "${pretrain_args[@]}"
        RESOLVED_BASE_ENCODER_CKPT="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"
        return 0
    fi

    if [ -n "${PRETRAIN_CHECKPOINT}" ]; then
        if [ ! -f "${PRETRAIN_CHECKPOINT}" ]; then
            echo "Configured PRETRAIN_CHECKPOINT does not exist: ${PRETRAIN_CHECKPOINT}" >&2
            exit 1
        fi
        RESOLVED_BASE_ENCODER_CKPT="${PRETRAIN_CHECKPOINT}"
        return 0
    fi

    if resolved_encoder_ckpt="$(resolve_encoder_checkpoint "${backbone}" "${pretrain_experiment_id}" "${base_pretrain_experiment_id}")"; then
        RESOLVED_BASE_ENCODER_CKPT="${resolved_encoder_ckpt}"
        return 0
    fi

    echo "No existing base pretrain checkpoint found; running ${PRETRAIN_LOSS} pretraining for ${backbone}."
    mapfile -t pretrain_args < <(
        build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
    )
    run_python_train "${pretrain_args[@]}"
    RESOLVED_BASE_ENCODER_CKPT="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"
}

run_plain_finetune() {
    local backbone="$1"
    local experiment_id="$2"
    local epochs="$3"
    local encoder_ckpt="$4"
    local enable_post_train_gradcam="$5"

    mapfile -t finetune_args < <(
        build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${experiment_id}" "${epochs}"
    )
    finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
    if [ "${enable_post_train_gradcam}" = "1" ]; then
        finetune_args+=(--post-train-gradcam)
    fi
    run_python_train "${finetune_args[@]}"
}

run_roi_repretrain() {
    local backbone="$1"
    local experiment_id="$2"
    local epochs="$3"
    local init_encoder_ckpt="$4"
    local roi_records_path="$5"

    mapfile -t pretrain_args < <(
        build_common_args "pretrain" "${backbone}" "${ROI_PRETRAIN_LOSS}" "${experiment_id}" "${epochs}"
    )
    pretrain_args+=(
        --init-encoder-ckpt "${init_encoder_ckpt}"
        --roi-records-path "${roi_records_path}"
    )
    run_python_train "${pretrain_args[@]}"
}

export_roi_records() {
    local checkpoint_path="$1"
    local output_path="$2"

    echo
    echo "Running: ${PYTHON_BIN} export_train_rois.py --checkpoint ${checkpoint_path} --output-path ${output_path}"
    "${PYTHON_BIN}" export_train_rois.py \
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
    echo "ROI pipeline for backbone=${backbone}"
    echo "==============================================="

    base_pretrain_base_id="${backbone}_pretrain_${PRETRAIN_LOSS}"
    base_pretrain_id="$(build_experiment_id "${base_pretrain_base_id}")"
    bootstrap_finetune_id="$(build_experiment_id "${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}_roi_bootstrap")"
    roi_pretrained_id="$(build_experiment_id "${backbone}_pretrain_${ROI_PRETRAIN_LOSS}_ROIpretrained")"
    final_finetune_id="$(build_experiment_id "${backbone}_finetune_${ROI_PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}_ROIpretrained")"

    echo "[1/4] Resolving baseline pretrain checkpoint"
    resolve_or_create_base_pretrain "${backbone}" "${base_pretrain_base_id}" "${base_pretrain_id}"
    base_encoder_ckpt="${RESOLVED_BASE_ENCODER_CKPT}"
    echo "Using baseline encoder checkpoint: ${base_encoder_ckpt}"

    echo "[2/4] Bootstrap finetune for Grad-CAM ROI discovery -> ${bootstrap_finetune_id}"
    run_plain_finetune "${backbone}" "${bootstrap_finetune_id}" "${ROI_BOOTSTRAP_FINETUNE_EPOCHS}" "${base_encoder_ckpt}" "0"

    bootstrap_best_ckpt="${SAVE_DIR}/${bootstrap_finetune_id}_best.pt"
    bootstrap_final_ckpt="${SAVE_DIR}/${bootstrap_finetune_id}_final.pt"
    bootstrap_source_ckpt="${bootstrap_best_ckpt}"
    if [ ! -f "${bootstrap_source_ckpt}" ]; then
        bootstrap_source_ckpt="${bootstrap_final_ckpt}"
    fi
    if [ ! -f "${bootstrap_source_ckpt}" ]; then
        echo "Could not find the bootstrap finetune checkpoint needed for ROI export." >&2
        echo "Checked: ${bootstrap_best_ckpt} and ${bootstrap_final_ckpt}" >&2
        exit 1
    fi

    roi_records_path="${ROI_RECORDS_DIR}/${roi_pretrained_id}_train_rois.json"
    echo "Exporting train-split Grad-CAM ROIs -> ${roi_records_path}"
    export_roi_records "${bootstrap_source_ckpt}" "${roi_records_path}"

    echo "[3/4] ROI re-pretrain -> ${roi_pretrained_id}"
    run_roi_repretrain "${backbone}" "${roi_pretrained_id}" "${ROI_REPRETRAIN_EPOCHS}" "${base_encoder_ckpt}" "${roi_records_path}"

    roi_pretrained_encoder_ckpt="${SAVE_DIR}/${roi_pretrained_id}_encoder.pt"
    if [ ! -f "${roi_pretrained_encoder_ckpt}" ]; then
        echo "Expected ROI-pretrained encoder checkpoint was not created: ${roi_pretrained_encoder_ckpt}" >&2
        exit 1
    fi
    echo "Saved ROI-pretrained encoder checkpoint: ${roi_pretrained_encoder_ckpt}"

    echo "[4/4] Final finetune from ROI-pretrained encoder -> ${final_finetune_id}"
    run_plain_finetune "${backbone}" "${final_finetune_id}" "${FINAL_FINETUNE_EPOCHS}" "${roi_pretrained_encoder_ckpt}" "1"
done

echo
echo "ROI pipeline complete."
