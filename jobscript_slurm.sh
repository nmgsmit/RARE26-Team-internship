#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | head -n 1 | cut -d= -f2- | tr -d '\r' | xargs)
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not found in .env or empty."
    exit 1
fi
export WANDB_API_KEY

# Load Python module
module load Python/3.11.3-GCCcore-12.3.0  # adjust to available Python module

# Create/activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run training
/bin/bash main.sh