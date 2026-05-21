#!/usr/bin/env bash
# Run this on the login node to submit the full pipeline.
# Pretrain runs first; the four finetune heads start automatically once it succeeds.
#
# Usage:
#   bash submit_best_model.sh
#
# Optional overrides:
#   PARTITION=gpu_h100 bash submit_best_model.sh
#   PRETRAIN_TIME=48:00:00 bash submit_best_model.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu_a100}"
PRETRAIN_TIME="${PRETRAIN_TIME:-24:00:00}"
FINETUNE_TIME="${FINETUNE_TIME:-16:00:00}"

echo "Submitting pretrain job..."
PRETRAIN_JOB_ID=$(sbatch \
    --parsable \
    --partition="${PARTITION}" \
    --time="${PRETRAIN_TIME}" \
    jobscript_slurm_pretrain.sh)

echo "  Pretrain job ID : ${PRETRAIN_JOB_ID}"

echo "Submitting finetune job (depends on pretrain succeeding)..."
FINETUNE_JOB_ID=$(sbatch \
    --parsable \
    --partition="${PARTITION}" \
    --time="${FINETUNE_TIME}" \
    --dependency="afterok:${PRETRAIN_JOB_ID}" \
    jobscript_slurm_finetune_heads.sh)

echo "  Finetune job ID : ${FINETUNE_JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -u \${USER}"
echo "  tail -f slurm_trainmodel/slurm_pretrain-${PRETRAIN_JOB_ID}.out"
echo "  tail -f slurm_trainmodel/slurm_finetune_heads-${FINETUNE_JOB_ID}.out"
echo ""
echo "If pretrain fails the finetune job is automatically cancelled."
echo "Resubmit individually with:"
echo "  sbatch jobscript_slurm_pretrain.sh"
echo "  sbatch --dependency=afterok:<new_pretrain_id> jobscript_slurm_finetune_heads.sh"
