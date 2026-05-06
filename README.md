# Team Internship - AIES Master Program

This repository contains code and resources for the Team Internship project as part of the Master's program in Artificial Intelligence and Engineering Systems (AIES) at TU Eindhoven.

Example run command:

EXPERIMENT_ID=mlpfw_lr1em6_temp05 \
BACKBONES_CSV=gastronet \
HEAD_TYPE=mlp_fullwidth \
PRETRAIN_LOSS=suppro \
LR=1e-6 \
TEMPERATURE=0.5 \
BASE_TEMPERATURE=0.5 \
PRETRAIN_EPOCHS=50 \
BATCH_SIZE=32 \
sbatch jobscript_slurm.sh
