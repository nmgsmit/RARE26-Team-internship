#!/usr/bin/env bash
# TEST version of submit_scale_ablation.sh
# Runs all 4 scales with 1 epoch each and all 4 heads (KNN, linear, SVM, MLP).
# Use this to verify the full pipeline is working before the full run.
#
# Usage:
#   bash submit_scale_ablation_test.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu_a100}"
WANDB_GROUP="P8_scale_ablation_test"

SCALES=("0.4" "0.6" "0.8" "0.95")

echo "TEST RUN — 1 epoch, all 4 heads | group=${WANDB_GROUP}"
echo "Scales: ${SCALES[*]}"
echo ""

for scale in "${SCALES[@]}"; do
    scale_tag="scale${scale//./}"
    run_tag="P8_${scale_tag}_test"

    SBATCH_EXPORT="ALL,RUN_TAG=${run_tag},WANDB_GROUP=${WANDB_GROUP},MIN_CROP_SCALE=${scale},PRETRAIN_EPOCHS=1,FINETUNE_EPOCHS=1,HEADS=knn linear svm mlp_fullwidth"

    echo "── Scale ${scale} (RUN_TAG=${run_tag}) ──────────────────────────────"

    pretrain_job_id=$(sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --time="00:30:00" \
        --job-name="test_pretrain_${run_tag}" \
        --output="slurm_trainmodel/slurm_test_pretrain_${run_tag}-%j.out" \
        --export="${SBATCH_EXPORT}" \
        jobscript_slurm_pretrain.sh)

    echo "  Pretrain job ID : ${pretrain_job_id}"

    finetune_job_id=$(sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --time="01:30:00" \
        --job-name="test_finetune_${run_tag}" \
        --output="slurm_trainmodel/slurm_test_finetune_${run_tag}-%j.out" \
        --dependency="afterok:${pretrain_job_id}" \
        --export="${SBATCH_EXPORT}" \
        jobscript_slurm_finetune_heads.sh)

    echo "  Finetune job ID : ${finetune_job_id} (depends on ${pretrain_job_id})"
    echo ""
done

echo "All test jobs submitted. Monitor with:"
echo "  squeue -u \${USER}"
echo ""
echo "Check logs at:"
for scale in "${SCALES[@]}"; do
    scale_tag="scale${scale//./}"
    echo "  slurm_trainmodel/slurm_test_pretrain_P8_${scale_tag}_test-*.out"
    echo "  slurm_trainmodel/slurm_test_finetune_P8_${scale_tag}_test-*.out"
done
