#!/usr/bin/env bash

set -euo pipefail

# This sweep runs:
# 1. 4 TTC-style pretraining runs:
#    - dinov3 + supmin
#    - dinov3 + suppro
#    - gastronet + supmin
#    - gastronet + suppro
# 2. 8 finetuning runs on top of those encoders:
#    - each encoder above with CE
#    - each encoder above with class-balanced CE
#
# The finetune stage always uses the configurable mlp_fullwidth head.
# Pretrain ignores the classifier head and only trains backbone + projection head.

DATA_DIR="${DATA_DIR:-./data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-./data/EVC_Barretts_FullSet/images}"
POST_TRAIN_GRADCAM_DATASET_ROOT="${POST_TRAIN_GRADCAM_DATASET_ROOT:-./data/EVC_Barretts_FullSet}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/sweep_pretrain_finetune_mlp_fullwidth}"

BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"

HEAD_TYPE="${HEAD_TYPE:-mlp_fullwidth}"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-1}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"

GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENABLE_POST_TRAIN_GRADCAM="${ENABLE_POST_TRAIN_GRADCAM:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

BACKBONES_CSV="${BACKBONES_CSV:-dinov3,gastronet}"
PRETRAIN_LOSSES_CSV="${PRETRAIN_LOSSES_CSV:-supmin,suppro}"
FINETUNE_LOSSES_CSV="${FINETUNE_LOSSES_CSV:-ce,class-balanced}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"
IFS=',' read -r -a PRETRAIN_LOSSES <<< "${PRETRAIN_LOSSES_CSV}"
IFS=',' read -r -a FINETUNE_LOSSES <<< "${FINETUNE_LOSSES_CSV}"

mkdir -p "${SAVE_DIR}"

if [ -z "${PYTHON_BIN}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Could not find python3 or python on PATH." >&2
        exit 1
    fi
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

    if [ "${stage}" != "pretrain" ] && [ "${ENABLE_POST_TRAIN_GRADCAM}" = "1" ]; then
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

for backbone in "${BACKBONES[@]}"; do
    for pretrain_loss in "${PRETRAIN_LOSSES[@]}"; do
        pretrain_experiment_id="${backbone}_pretrain_${pretrain_loss}"
        encoder_ckpt="${SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

        if [ "${RUN_PRETRAIN}" = "1" ]; then
            if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${encoder_ckpt}" ]; then
                echo "Skipping existing pretrain encoder: ${encoder_ckpt}"
            else
                mapfile -t pretrain_args < <(
                    build_common_args "pretrain" "${backbone}" "${pretrain_loss}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
                )
                run_python_train "${pretrain_args[@]}"
            fi
        fi

        if [ "${RUN_FINETUNE}" != "1" ]; then
            continue
        fi

        if [ ! -f "${encoder_ckpt}" ]; then
            echo "Missing encoder checkpoint for finetune: ${encoder_ckpt}" >&2
            echo "Run pretraining first or set RUN_PRETRAIN=1." >&2
            exit 1
        fi

        for finetune_loss in "${FINETUNE_LOSSES[@]}"; do
            finetune_experiment_id="${backbone}_finetune_${pretrain_loss}_${finetune_loss}_${HEAD_TYPE}"
            final_ckpt="${SAVE_DIR}/${finetune_experiment_id}_final.pt"

            if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${final_ckpt}" ]; then
                echo "Skipping existing finetune checkpoint: ${final_ckpt}"
                continue
            fi

            mapfile -t finetune_args < <(
                build_common_args "finetune" "${backbone}" "${finetune_loss}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}"
            )
            finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
            run_python_train "${finetune_args[@]}"
        done
    done
done

echo
echo "Sweep finished."
