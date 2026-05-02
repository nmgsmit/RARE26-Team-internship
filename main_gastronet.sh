#!/usr/bin/env bash

BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"   # change to: gastronet | dinov3
TRAIN_STAGE="${TRAIN_STAGE:-baseline}"            # change to: baseline | pretrain | finetune
LOSS_NAME="${LOSS_NAME:-}"                        # blank = stage default
HEAD_TYPE="${HEAD_TYPE:-mlp_fullwidth}"           # linear | ln_linear | mlp_fullwidth | mlp_bottleneck | residual_bottleneck | cosine_linear
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-}"            # used by bottleneck heads
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"               # used by bottleneck heads
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-1}"       # used by mlp_fullwidth
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-}"              # blank = backbone feature width
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"                 # used by mlp_fullwidth
EXPERIMENT_ID="${EXPERIMENT_ID:-}"

# Pretrain flow:
# 1. TRAIN_STAGE=pretrain with LOSS_NAME=supmin or suppro.
#    This creates ./checkpoints/<EXPERIMENT_ID>_encoder.pt
# 2. TRAIN_STAGE=finetune with LOSS_NAME=ce or class-balanced and ENCODER_CKPT set to that encoder checkpoint.
# 3. If you want to skip TTC pretraining entirely, use TRAIN_STAGE=baseline.
ENCODER_CKPT="${ENCODER_CKPT:-}"

# Only needed when BACKBONE_PRESET=gastronet and you want a custom checkpoint path.
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"

if [ -z "${LOSS_NAME}" ]; then
    case "${TRAIN_STAGE}" in
        pretrain) LOSS_NAME="supmin" ;;
        baseline) LOSS_NAME="class-balanced" ;;
        finetune) LOSS_NAME="ce" ;;
        *)
            echo "Unsupported TRAIN_STAGE '${TRAIN_STAGE}'." >&2
            exit 1
            ;;
    esac
fi

if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="${BACKBONE_PRESET}_${TRAIN_STAGE}_${LOSS_NAME}_${HEAD_TYPE}"
fi

COMMON_ARGS=(
    --stage "${TRAIN_STAGE}"
    --loss-name "${LOSS_NAME}"
    --data-dir ./data/Challenge_train_data
    --testset-images-dir ./data/EVC_Barretts_FullSet/images
    --batch-size 32
    --epochs 20
    --lr 1e-4
    --warmup-epochs 3
    --num-workers 10
    --seed 42
    --experiment-id "${EXPERIMENT_ID}"
    --backbone-preset "${BACKBONE_PRESET}"
    --head-type "${HEAD_TYPE}"
    --head-dropout "${HEAD_DROPOUT}"
    --mlp-hidden-layers "${MLP_HIDDEN_LAYERS}"
    --mlp-dropout "${MLP_DROPOUT}"
    --post-train-gradcam
    --post-train-gradcam-dataset-root ./data/EVC_Barretts_FullSet
)

if [ -n "${HEAD_HIDDEN_DIM}" ]; then
    COMMON_ARGS+=(--head-hidden-dim "${HEAD_HIDDEN_DIM}")
fi

if [ -n "${MLP_HIDDEN_DIM}" ]; then
    COMMON_ARGS+=(--mlp-hidden-dim "${MLP_HIDDEN_DIM}")
fi

if [ "${BACKBONE_PRESET}" = "gastronet" ]; then
    COMMON_ARGS+=(--backbone-weights-path "${GASTRONET_CKPT}")
fi

if [ "${TRAIN_STAGE}" = "finetune" ]; then
    if [ -z "${ENCODER_CKPT}" ]; then
        echo "ENCODER_CKPT must be set when TRAIN_STAGE=finetune." >&2
        exit 1
    fi
    COMMON_ARGS+=(--encoder-ckpt "${ENCODER_CKPT}")
fi

python3 train.py "${COMMON_ARGS[@]}"
