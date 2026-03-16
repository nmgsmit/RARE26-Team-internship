#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --output=slurm_testmodel/slurm-%j.out

set -euo pipefail
mkdir -p slurm_testmodel

# Load Python module
module load 2023 # adjust to available Python module

# Create/activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi


# Usage:
#   sbatch localtest.sh /path/to/checkpoint.pt
# or set MODEL_PATH in environment before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No model checkpoint path provided."
    echo "Pass it as first argument or set MODEL_PATH env variable."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# All test config variables are defined in mainTEST.sh (similar to main.sh for training).
export MODEL_PATH
/bin/bash "$SCRIPT_DIR/mainTEST.sh" "$MODEL_PATH"
