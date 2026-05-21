#!/usr/bin/env bash
# Submit the full scale ablation: 4 crop-scale pretrains × 4 heads = 16 finetune runs.
#
# All 4 pretrains are submitted in parallel.  For each scale, the finetune-heads
# job is queued with --dependency=afterok so it only starts once its pretrain
# succeeds.  If a pretrain fails, the matching finetune is cancelled automatically.
#
# Total SLURM jobs: 4 pretrain + 4 finetune-heads = 8 jobs.
# Total model runs: 4 × 4 heads = 16.
#
# Usage:
#   bash submit_scale_ablation.sh
#
# Optional overrides:
#   PARTITION=gpu_h100 bash submit_scale_ablation.sh
#   PRETRAIN_TIME=36:00:00 bash submit_scale_ablation.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu_a100}"
PRETRAIN_TIME="${PRETRAIN_TIME:-24:00:00}"
FINETUNE_TIME="${FINETUNE_TIME:-16:00:00}"
WANDB_PROJECT="RARE25-Project"
WANDB_GROUP="P8_scale_ablation"

SCALES=("0.4" "0.6" "0.8" "0.95")

echo "Submitting scale ablation | group=${WANDB_GROUP}"
echo "Scales: ${SCALES[*]}"
echo ""

for scale in "${SCALES[@]}"; do
    # Build a compact tag: 0.4→scale04  0.95→scale095
    scale_tag="scale${scale//./}"
    run_tag="P8_${scale_tag}"

    # Use --export to bake the variables directly into each sbatch call.
    # This is more reliable than relying on the cluster to propagate
    # exported shell variables, which varies between SLURM configurations.
    SBATCH_EXPORT="ALL,RUN_TAG=${run_tag},WANDB_GROUP=${WANDB_GROUP},MIN_CROP_SCALE=${scale}"

    echo "── Scale ${scale} (RUN_TAG=${run_tag}) ──────────────────────────────"

    pretrain_job_id=$(sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --time="${PRETRAIN_TIME}" \
        --job-name="pretrain_${run_tag}" \
        --output="slurm_trainmodel/slurm_pretrain_${run_tag}-%j.out" \
        --export="${SBATCH_EXPORT}" \
        jobscript_slurm_pretrain.sh)

    echo "  Pretrain job ID : ${pretrain_job_id}"

    finetune_job_id=$(sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --time="${FINETUNE_TIME}" \
        --job-name="finetune_${run_tag}" \
        --output="slurm_trainmodel/slurm_finetune_${run_tag}-%j.out" \
        --dependency="afterok:${pretrain_job_id}" \
        --export="${SBATCH_EXPORT}" \
        jobscript_slurm_finetune_heads.sh)

    echo "  Finetune job ID : ${finetune_job_id} (depends on ${pretrain_job_id})"
    echo ""
done

echo "All jobs submitted. Monitor with:"
echo "  squeue -u \${USER}"
echo ""
echo "Expected submission artifacts (one per head per scale):"
for scale in "${SCALES[@]}"; do
    scale_tag="scale${scale//./}"
    run_tag="P8_${scale_tag}"
    for head in knn linear svm mlp_fullwidth; do
        echo "  ./checkpoints/${run_tag}/finetune/${head}/${run_tag}_finetune_${head}_submission.pt"
    done
done
