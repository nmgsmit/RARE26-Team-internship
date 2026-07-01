#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/eval_fusion-%j.out

# Multi-backbone fusion eval: late (combine_backbones) + early (concat_fusion) for
# all subsets of {dino,simclr,moco} at k in {5,20}. Logs each to W&B group multibackbone.
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

D=clean_baseline_crop095_knn5
S=simclr_crop095_knn5
M=moco_crop095_knn5
ens() { echo "checkpoints/$1/$1_ensembles/ensemble_knn$2.pt"; }

for K in 5 20; do
    echo "############################ LATE FUSION (avg probs) k=$K ############################"
    python combine_backbones.py dino="$(ens $D $K)" simclr="$(ens $S $K)" || true
    python combine_backbones.py dino="$(ens $D $K)" moco="$(ens $M $K)"   || true
    python combine_backbones.py simclr="$(ens $S $K)" moco="$(ens $M $K)" || true
    python combine_backbones.py dino="$(ens $D $K)" simclr="$(ens $S $K)" moco="$(ens $M $K)" || true

    echo "############################ EARLY FUSION (concat) k=$K ############################"
    for CFG in "$D" "$S" "$M" "$D,$S" "$D,$M" "$S,$M" "$D,$S,$M"; do
        python concat_fusion.py --eids "$CFG" --knn "$K" || echo "concat FAILED: $CFG k$K"
    done
done
echo "EVAL_FUSION_DONE"
