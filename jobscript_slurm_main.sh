#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=slurm_trainmodel/slurm-main-%j.out

set -euo pipefail
mkdir -p slurm_trainmodel

if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
    fi
fi

module load 2023

if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-RARE25-Project}"
WANDB_GROUP="${WANDB_GROUP:-branch_compare_pretraining}"
UMAP_WANDB_GROUP="${UMAP_WANDB_GROUP:-branch_compare_pretraining_umap}"

STAGE="${STAGE:-pretrain}"
PRETRAIN_LOSS="${PRETRAIN_LOSS:-suppro}"
BACKBONE_PRESET="${BACKBONE_PRESET:-gastronet}"
HEAD_TYPE="${HEAD_TYPE:-linear}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-branch_compare_main}"

DATA_DIR="${DATA_DIR:-../data/Challenge_train_data}"
TESTSET_IMAGES_DIR="${TESTSET_IMAGES_DIR:-../data/EVC_Barretts_FullSet/images}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/branch_compare/main}"
GASTRONET_CKPT="${GASTRONET_CKPT:-../Gastronet/dinov2.pth}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-42}"

PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
PRETRAIN_BACKBONE_LR_SCALE="${PRETRAIN_BACKBONE_LR_SCALE:-0.5}"
PRETRAIN_LAYER_DECAY="${PRETRAIN_LAYER_DECAY:-0.8}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.05}"
PRETRAIN_FREEZE_BLOCKS="${PRETRAIN_FREEZE_BLOCKS:-}"
PRETRAIN_UNFREEZE_EPOCH="${PRETRAIN_UNFREEZE_EPOCH:-10}"
PRETRAIN_BATCH_MODE="${PRETRAIN_BATCH_MODE:-balanced}"
PRETRAIN_PROBE="${PRETRAIN_PROBE:-knn}"
PRETRAIN_PROBE_K="${PRETRAIN_PROBE_K:-5}"
PRETRAIN_PROBE_EVERY="${PRETRAIN_PROBE_EVERY:-1}"

LAMBDA_SUPPRO="${LAMBDA_SUPPRO:-1.0}"
LAMBDA_SUPMIN="${LAMBDA_SUPMIN:-0.25}"
SUPMIN_MARGIN="${SUPMIN_MARGIN:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"

UMAP_EMBEDDING_TYPES_CSV="${UMAP_EMBEDDING_TYPES_CSV:-projection}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"
UMAP_METRIC="${UMAP_METRIC:-cosine}"
DISABLE_POST_PRETRAIN_UMAP="${DISABLE_POST_PRETRAIN_UMAP:-0}"

mkdir -p "${SAVE_DIR}"

IFS=',' read -r -a UMAP_EMBEDDING_TYPES <<< "${UMAP_EMBEDDING_TYPES_CSV}"

for UMAP_EMBEDDING_TYPE in "${UMAP_EMBEDDING_TYPES[@]}"; do
    RUN_ID="${EXPERIMENT_PREFIX}_${BACKBONE_PRESET}_${PRETRAIN_LOSS}_${HEAD_TYPE}_umap_${UMAP_EMBEDDING_TYPE}"

    CMD=(
        "${PYTHON_BIN}" train.py
        --stage "${STAGE}"
        --loss-name "${PRETRAIN_LOSS}"
        --experiment-id "${RUN_ID}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --backbone-preset "${BACKBONE_PRESET}"
        --head-type "${HEAD_TYPE}"
        --data-dir "${DATA_DIR}"
        --testset-images-dir "${TESTSET_IMAGES_DIR}"
        --save-dir "${SAVE_DIR}"
        --epochs "${PRETRAIN_EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --pretrain-lr "${PRETRAIN_LR}"
        --pretrain-backbone-lr-scale "${PRETRAIN_BACKBONE_LR_SCALE}"
        --pretrain-layer-decay "${PRETRAIN_LAYER_DECAY}"
        --pretrain-weight-decay "${PRETRAIN_WEIGHT_DECAY}"
        --pretrain-unfreeze-epoch "${PRETRAIN_UNFREEZE_EPOCH}"
        --pretrain-batch-mode "${PRETRAIN_BATCH_MODE}"
        --pretrain-probe "${PRETRAIN_PROBE}"
        --pretrain-probe-k "${PRETRAIN_PROBE_K}"
        --pretrain-probe-every "${PRETRAIN_PROBE_EVERY}"
        --lambda-suppro "${LAMBDA_SUPPRO}"
        --lambda-supmin "${LAMBDA_SUPMIN}"
        --supmin-margin "${SUPMIN_MARGIN}"
        --warmup-epochs "${WARMUP_EPOCHS}"
        --umap-embedding-type "${UMAP_EMBEDDING_TYPE}"
        --umap-n-neighbors "${UMAP_N_NEIGHBORS}"
        --umap-min-dist "${UMAP_MIN_DIST}"
        --umap-metric "${UMAP_METRIC}"
        --umap-wandb-group "${UMAP_WANDB_GROUP}"
    )

    if [ "${BACKBONE_PRESET}" = "gastronet" ]; then
        CMD+=(--backbone-weights-path "${GASTRONET_CKPT}")
    fi

    if [ -n "${PRETRAIN_FREEZE_BLOCKS}" ]; then
        CMD+=(--pretrain-freeze-blocks "${PRETRAIN_FREEZE_BLOCKS}")
    fi

    if [ "${DISABLE_POST_PRETRAIN_UMAP}" = "1" ]; then
        CMD+=(--disable-post-pretrain-umap)
    fi

    echo
    echo "Launching run ${RUN_ID}"
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"
done
