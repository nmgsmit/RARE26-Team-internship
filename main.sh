#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# Fill this in first for new runs.
# When set, this becomes the checkpoint base name for each stage-specific folder.
EXPERIMENT_ID="${EXPERIMENT_ID:-name}"
WANDB_GROUP="${WANDB_GROUP:-group}"

# Easy checkpoint save configuration.
CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-./checkpoints}"
EXPERIMENT_SAVE_SUBDIR="${EXPERIMENT_SAVE_SUBDIR:-default_experiment}"

# Crucial model choices.
# Supported entries in BACKBONES_CSV include: gastronet, dinov3, simclr, mocov2, resnet50
BACKBONES_CSV="${BACKBONES_CSV:-gastronet}" # gastronet, dinov3, simclr, mocov2, resnet50
STAGES_CSV="${STAGES_CSV:-pretrain,finetune}" # baseline, pretrain, finetune
CV_NUM_FOLDS="${CV_NUM_FOLDS:-1}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-class-balanced}"
HEAD_TYPE="${HEAD_TYPE:-linear}"

# Training and optimization.
TEMPERATURE="${TEMPERATURE:-0.07}"
BASE_TEMPERATURE="${BASE_TEMPERATURE:-0.07}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

LR="${LR:-1e-4}"
BASELINE_LR="${BASELINE_LR:-1e-4}"
PRETRAIN_BACKBONE_LR="${PRETRAIN_BACKBONE_LR:-${LR}}"
PRETRAIN_PROJ_LR="${PRETRAIN_PROJ_LR:-3e-4}"
FINETUNE_LR="${FINETUNE_LR:-3e-4}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"

# Finetune optimization.
FORCE_PRETRAIN="${FORCE_PRETRAIN:-1}"  # set to 0 to quickly optimize finetune
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-set_checkpoint.pt}" # select pretraining checkpoint

# Shared paths and runtime defaults: these usually stay fixed across runs.
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
NUM_WORKERS="${NUM_WORKERS:-10}"


# Hardcoded backbone checkpoints.
GASTRONET_CKPT="../Gastronet/dinov2.pth"
SIMCLR_CKPT="../Gastronet/RN50_GastroNet-5M_SIMCLRv2.pth"
MOCOV2_CKPT="../Gastronet/RN50_GastroNet-5M_MOCOv2.pth"
RESNET50_CKPT="../Gastronet/RN50_ImageNet_timm_resnet50.pth"
PYTHON_BIN="${PYTHON_BIN:-python3}"

IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"
IFS=',' read -r -a STAGES <<< "${STAGES_CSV}"

if [ -n "${PRETRAIN_CHECKPOINT}" ] && [ "${#BACKBONES[@]}" -gt 1 ]; then
    echo "PRETRAIN_CHECKPOINT expects a single backbone run. Set BACKBONES_CSV to one backbone or leave PRETRAIN_CHECKPOINT blank." >&2
    exit 1
fi

for stage in "${STAGES[@]}"; do
    case "${stage}" in
        baseline|pretrain|finetune) ;;
        *)
            echo "Unsupported stage in STAGES_CSV: ${stage}. Use baseline, pretrain, finetune, or a comma-separated combination." >&2
            exit 1
            ;;
    esac
done

EXPERIMENT_SAVE_DIR="${CHECKPOINT_ROOT_DIR}/${EXPERIMENT_SAVE_SUBDIR}"

build_common_args() {
    local stage="$1"
    local backbone="$2"
    local loss_name="$3"
    local experiment_id="$4"
    local epochs="$5"
    local save_dir="$6"
    local num_folds="$7"
    local fold_index="$8"

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
        --baseline-lr "${BASELINE_LR}"
        --pretrain-backbone-lr "${PRETRAIN_BACKBONE_LR}"
        --pretrain-proj-lr "${PRETRAIN_PROJ_LR}"
        --finetune-lr "${FINETUNE_LR}"
        --warmup-epochs "${WARMUP_EPOCHS}"
        --num-folds "${num_folds}"
        --fold-index "${fold_index}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --save-dir "${save_dir}"
        --temperature "${TEMPERATURE}"
        --base-temperature "${BASE_TEMPERATURE}"
    )

    if [ "${backbone}" = "gastronet" ]; then
        args+=(--backbone-weights-path "${GASTRONET_CKPT}")
    elif [ "${backbone}" = "simclr" ]; then
        args+=(--backbone-weights-path "${SIMCLR_CKPT}")
    elif [ "${backbone}" = "mocov2" ]; then
        args+=(--backbone-weights-path "${MOCOV2_CKPT}")
    elif [ "${backbone}" = "resnet50" ]; then
        args+=(--backbone-weights-path "${RESNET50_CKPT}" --no-pretrained)
    fi

    # Baseline and finetune both produce classifier checkpoints, so run the
    # post-training Grad-CAM evaluation for both and keep it in the same W&B run.
    if [ "${stage}" = "baseline" ] || [ "${stage}" = "finetune" ]; then
        args+=(--post-train-gradcam)
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
    local stage_suffix="$2"
    if [ -n "${EXPERIMENT_ID}" ]; then
        printf '%s_%s\n' "${EXPERIMENT_ID}" "${stage_suffix}"
    else
        printf '%s_%s\n' "${base_id}" "${stage_suffix}"
    fi
}

resolve_encoder_checkpoint() {
    local pretrain_experiment_id="$1"
    local base_pretrain_experiment_id="$2"
    local pretrain_save_dir="$3"
    local fold_save_dir="$4"
    local primary_ckpt="${pretrain_save_dir}/${pretrain_experiment_id}_encoder.pt"

    local candidates=(
        "${primary_ckpt}"
        "${fold_save_dir}/${pretrain_experiment_id}_encoder.pt"
        "${fold_save_dir}/${base_pretrain_experiment_id}_encoder.pt"
        "${CHECKPOINT_ROOT_DIR}/${pretrain_experiment_id}_encoder.pt"
        "${CHECKPOINT_ROOT_DIR}/${base_pretrain_experiment_id}_encoder.pt"
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
    for ((fold_index=0; fold_index<CV_NUM_FOLDS; fold_index++)); do
        fold_save_dir="${EXPERIMENT_SAVE_DIR}"
        if [ "${CV_NUM_FOLDS}" -gt 1 ]; then
            fold_save_dir="${EXPERIMENT_SAVE_DIR}/fold_$((fold_index + 1))"
        fi

        BASELINE_SAVE_DIR="${fold_save_dir}/baselines"
        PRETRAIN_SAVE_DIR="${fold_save_dir}/pretrain"
        FINETUNE_SAVE_DIR="${fold_save_dir}/finetune"
        mkdir -p "${BASELINE_SAVE_DIR}" "${PRETRAIN_SAVE_DIR}" "${FINETUNE_SAVE_DIR}"

        base_pretrain_experiment_id="${backbone}_pretrain_${PRETRAIN_LOSS}"
        pretrain_experiment_id="$(build_experiment_id "${base_pretrain_experiment_id}" "pretrain")"
        encoder_ckpt="${PRETRAIN_SAVE_DIR}/${pretrain_experiment_id}_encoder.pt"

        run_pretrain_stage=0
        run_baseline_stage=0
        run_finetune_stage=0
        for stage in "${STAGES[@]}"; do
            case "${stage}" in
                baseline) run_baseline_stage=1 ;;
                pretrain) run_pretrain_stage=1 ;;
                finetune) run_finetune_stage=1 ;;
            esac
        done

        if [ "${run_baseline_stage}" = "1" ]; then
            echo
            if [ "${CV_NUM_FOLDS}" -gt 1 ]; then
                echo "Running fold $((fold_index + 1))/${CV_NUM_FOLDS} for backbone ${backbone}"
            fi
            base_baseline_experiment_id="${backbone}_baseline_${FINETUNE_LOSS}_${HEAD_TYPE}"
            baseline_experiment_id="$(build_experiment_id "${base_baseline_experiment_id}" "baseline")"
            mapfile -t baseline_args < <(
                build_common_args "baseline" "${backbone}" "${FINETUNE_LOSS}" "${baseline_experiment_id}" "${FINETUNE_EPOCHS}" "${BASELINE_SAVE_DIR}" "${CV_NUM_FOLDS}" "${fold_index}"
            )
            run_python_train "${baseline_args[@]}"
        fi

        if [ "${run_pretrain_stage}" = "1" ] || [ "${run_finetune_stage}" = "1" ]; then
            if [ "${run_pretrain_stage}" = "1" ] && [ "${FORCE_PRETRAIN}" = "1" ]; then
                echo
                echo "FORCE_PRETRAIN=1, so pretraining will run even if an encoder checkpoint already exists."
                mapfile -t pretrain_args < <(
                    build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}" "${PRETRAIN_SAVE_DIR}" "${CV_NUM_FOLDS}" "${fold_index}"
                )
                run_python_train "${pretrain_args[@]}"
            elif [ "${run_pretrain_stage}" = "1" ] && [ -n "${PRETRAIN_CHECKPOINT}" ]; then
                if [ ! -f "${PRETRAIN_CHECKPOINT}" ]; then
                    echo "Configured PRETRAIN_CHECKPOINT does not exist: ${PRETRAIN_CHECKPOINT}" >&2
                    exit 1
                fi
                encoder_ckpt="${PRETRAIN_CHECKPOINT}"
                echo
                echo "Using PRETRAIN_CHECKPOINT from main.sh: ${encoder_ckpt}"
            elif [ "${run_pretrain_stage}" = "1" ]; then
                mapfile -t pretrain_args < <(
                    build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}" "${PRETRAIN_SAVE_DIR}" "${CV_NUM_FOLDS}" "${fold_index}"
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
            elif resolved_encoder_ckpt="$(resolve_encoder_checkpoint "${pretrain_experiment_id}" "${base_pretrain_experiment_id}" "${PRETRAIN_SAVE_DIR}" "${fold_save_dir}")"; then
                encoder_ckpt="${resolved_encoder_ckpt}"
                echo
                echo "Found existing pretrained encoder checkpoint: ${encoder_ckpt}"
                echo "Skipping pretraining and reusing the existing encoder. Set FORCE_PRETRAIN=1 to retrain it."
            else
                echo "Finetune requested, but no encoder checkpoint was found for backbone ${backbone}." >&2
                echo "Run with STAGES_CSV=pretrain,finetune or set PRETRAIN_CHECKPOINT=/path/to/checkpoint.pt" >&2
                exit 1
            fi
        fi

        if [ "${run_finetune_stage}" = "1" ]; then
            echo
            base_finetune_experiment_id="${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}"
            finetune_experiment_id="$(build_experiment_id "${base_finetune_experiment_id}" "finetune")"
            mapfile -t finetune_args < <(
                build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}" "${FINETUNE_SAVE_DIR}" "${CV_NUM_FOLDS}" "${fold_index}"
            )
            finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
            run_python_train "${finetune_args[@]}"
        fi
    done
done
