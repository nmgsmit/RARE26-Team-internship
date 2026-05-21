#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=16:00:00
#SBATCH --output=slurm_trainmodel/slurm_finetune_heads-%j.out

# Finetune four classifier heads (KNN, linear probe, SVM, MLP) on the LOCO
# encoders produced by jobscript_slurm_pretrain.sh.  All heads share the same
# encoder; each runs in sequence within this single job.
#
# Outputs per head:
#   ./checkpoints/best_model/finetune/{head}/best_model_finetune_{head}_submission.pt
#   ./checkpoints/best_model/finetune/{head}/best_model_finetune_{head}_submission.json
#
# Submit after pretrain completes (submit_best_model.sh handles this automatically).

set -euo pipefail
mkdir -p slurm_trainmodel

# ── W&B auth ──────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
    if [ -n "${WANDB_API_KEY}" ]; then
        export WANDB_API_KEY
    fi
fi

# ── Environment ───────────────────────────────────────────────────────────────
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

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ── Crop scale (must match jobscript_slurm_pretrain.sh) ──────────────────────
MIN_CROP_SCALE="${MIN_CROP_SCALE:-0.4}"

# ── Paths (must match jobscript_slurm_pretrain.sh) ────────────────────────────
RUN_TAG="best_model"
WANDB_GROUP="best_model"
WANDB_PROJECT="RARE25-Project"
PRETRAIN_SAVE_DIR="./checkpoints/${RUN_TAG}/pretrain"

# Encoder path template: train.py substitutes {fold_index} and {holdout_center}
# per LOCO fold automatically, matching the filenames written by pretrain.
ENCODER_CKPT_TEMPLATE="${PRETRAIN_SAVE_DIR}/fold{fold_index}_val_{holdout_center}/${RUN_TAG}_pretrain_fold{fold_index}_val_{holdout_center}_encoder.pt"

# ── Shared finetune args ───────────────────────────────────────────────────────
# Build the args that are identical across all four heads into an array so we
# don't repeat ourselves and can't accidentally drift between heads.
COMMON_FINETUNE_ARGS=(
    --stage finetune
    --loss-name label-smoothed-ce
    --label-smoothing 0.05
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --backbone-preset gastronet
    --backbone-weights-path ../Gastronet/dinov2.pth
    --loco
    --batch-size 32
    --epochs 30
    --finetune-lr 2e-4
    --warmup-epochs 3
    --augmentation-intensity 3
    --roi-focus-prob 0.5
    --roi-negative-focus-prob 0.0
    --roi-warmup-epochs 5
    --roi-context-scale 2.0
    --roi-min-crop-scale "${MIN_CROP_SCALE}"
    --roi-center-jitter 0.05
    --roi-max-aspect-ratio 1.5
    --num-workers 10
    --seed 42
    --encoder-ckpt "${ENCODER_CKPT_TEMPLATE}"
    --post-train-gradcam
)

run_finetune() {
    local head="$1"
    local save_dir="./checkpoints/${RUN_TAG}/finetune/${head}"
    mkdir -p "${save_dir}"

    echo ""
    echo "================================================================="
    echo "Finetune | head=${head} | save_dir=${save_dir}"
    echo "================================================================="

    python train.py \
        "${COMMON_FINETUNE_ARGS[@]}" \
        --head-type "${head}" \
        --experiment-id "${RUN_TAG}_finetune_${head}" \
        --save-dir "${save_dir}"

    echo "Head ${head} done."
    echo "  Submission : ${save_dir}/${RUN_TAG}_finetune_${head}_submission.pt"
    echo "  Metadata   : ${save_dir}/${RUN_TAG}_finetune_${head}_submission.json"
}

# ── Run all four heads ─────────────────────────────────────────────────────────
run_finetune "knn"
run_finetune "linear"
run_finetune "svm"
run_finetune "mlp_fullwidth"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo "All heads complete. Submission artifacts:"
for head in knn linear svm mlp_fullwidth; do
    echo "  [${head}] ./checkpoints/${RUN_TAG}/finetune/${head}/${RUN_TAG}_finetune_${head}_submission.pt"
done
echo "================================================================="
