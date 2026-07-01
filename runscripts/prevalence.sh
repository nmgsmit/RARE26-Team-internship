#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/prevalence-%j.out

# Per-center vs pooled threshold + prevalence-shift stress test of logistic+triple.
set -uo pipefail
mkdir -p slurm_logs
module load 2023
source venv/bin/activate
export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
python prevalence_analysis.py
echo "PREVALENCE_DONE"
