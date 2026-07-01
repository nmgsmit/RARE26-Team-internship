#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/eval_heads-%j.out

# Continuous-score head ablation: logistic / linear-SVM / distance-weighted kNN,
# on the 3 backbones + triple fusion. Ranks by PPV@90RECALL. Logs to W&B group head-ablation.
set -uo pipefail
mkdir -p slurm_logs

if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then export WANDB_API_KEY; fi
    : "${WANDB_ENTITY:=$(grep '^WANDB_ENTITY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)}"
    : "${WANDB_PROJECT:=$(grep '^WANDB_PROJECT=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)}"
    export WANDB_ENTITY WANDB_PROJECT
fi

module load 2023
source venv/bin/activate
export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

python eval_heads.py \
    --backbones "${BACKBONES:-dino=clean_baseline_crop095_knn5,simclr=simclr_crop095_knn5,moco=moco_crop095_knn5}" \
    --heads "${HEADS:-logistic,linsvm,dwknn}" \
    --knn "${KNN:-20}"
echo "EVAL_HEADS_DONE"
