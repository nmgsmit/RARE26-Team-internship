#!/usr/bin/env bash

set -euo pipefail

# Fill this in first for new runs.
# When set, this prefix is added in front of the auto-generated stage/backbone experiment ids.
EXPERIMENT_ID="${EXPERIMENT_ID:-name}"
WANDB_GROUP="${WANDB_GROUP:-group}"

# Supported entries in BACKBONES_CSV include: gastronet, dinov3, simclr, mocov2, resnet50
BACKBONES_CSV="${BACKBONES_CSV:-gastronet,dinov3}" # gastronet, dinov3, simclr, mocov2, resnet50
STAGES_CSV="${STAGES_CSV:-pretrain,finetune}" # baseline, pretrain, finetune
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"

# Finetune optimization
FORCE_PRETRAIN="${FORCE_PRETRAIN:-1}"  # set to 0 to quickly optimize finetune
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-set_checkpoint.pt}" # select pretraining chekcpoint

# Crucial model choices.
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
FINETUNE_LOSS="${FINETUNE_LOSS:-class-balanced}"
HEAD_TYPE="${HEAD_TYPE:-linear}"

# Training and optimization.
TEMPERATURE="${TEMPERATURE:-0.07}"
BASE_TEMPERATURE="${BASE_TEMPERATURE:-0.07}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SEED="${SEED:-42}"

# Shared paths and runtime defaults: these usually stay fixed across runs.
SAVE_DIR="${SAVE_DIR:-./checkpoints/linear_suppro_dual_backbone}"
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
    elif [ "${backbone}" = "simclr" ]; then
        args+=(--backbone-weights-path "${SIMCLR_CKPT}")
    elif [ "${backbone}" = "mocov2" ]; then
        args+=(--backbone-weights-path "${MOCOV2_CKPT}")
    elif [ "${backbone}" = "resnet50" ]; then
        args+=(--backbone-weights-path "${RESNET50_CKPT}" --no-pretrained)
    fi

    if [ "${stage}" = "finetune" ]; then
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
    if [ -n "${EXPERIMENT_ID}" ]; then
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
        base_baseline_experiment_id="${backbone}_baseline_${FINETUNE_LOSS}_${HEAD_TYPE}"
        baseline_experiment_id="$(build_experiment_id "${base_baseline_experiment_id}")"
        mapfile -t baseline_args < <(
            build_common_args "baseline" "${backbone}" "${FINETUNE_LOSS}" "${baseline_experiment_id}" "${FINETUNE_EPOCHS}"
        )
        run_python_train "${baseline_args[@]}"
    fi

    if [ "${run_pretrain_stage}" = "1" ] || [ "${run_finetune_stage}" = "1" ]; then
        if [ "${run_pretrain_stage}" = "1" ] && [ "${FORCE_PRETRAIN}" = "1" ]; then
            echo
            echo "FORCE_PRETRAIN=1, so pretraining will run even if an encoder checkpoint already exists."
            mapfile -t pretrain_args < <(
                build_common_args "pretrain" "${backbone}" "${PRETRAIN_LOSS}" "${pretrain_experiment_id}" "${PRETRAIN_EPOCHS}"
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
            echo "Finetune requested, but no encoder checkpoint was found for backbone ${backbone}." >&2
            echo "Run with STAGES_CSV=pretrain,finetune or set PRETRAIN_CHECKPOINT=/path/to/checkpoint.pt" >&2
            exit 1
        fi
    fi

    if [ "${run_finetune_stage}" = "1" ]; then
        base_finetune_experiment_id="${backbone}_finetune_${PRETRAIN_LOSS}_${FINETUNE_LOSS}_${HEAD_TYPE}"
        finetune_experiment_id="$(build_experiment_id "${base_finetune_experiment_id}")"
        mapfile -t finetune_args < <(
            build_common_args "finetune" "${backbone}" "${FINETUNE_LOSS}" "${finetune_experiment_id}" "${FINETUNE_EPOCHS}"
        )
        finetune_args+=(--encoder-ckpt "${encoder_ckpt}")
        run_python_train "${finetune_args[@]}"
    fi
done
