# Team Internship - AIES Master Program

This repository contains code and resources for the Team Internship project as part of the Master's program in Artificial Intelligence and Engineering Systems (AIES) at TU Eindhoven.

Example run command:

EXPERIMENT_ID=lin_lr1em6_temp01 \
BACKBONES_CSV=gastronet \
HEAD_TYPE=linear \
PRETRAIN_LOSS=suppro \
LR=1e-6 \
TEMPERATURE=0.1 \
BASE_TEMPERATURE=0.1 \
PRETRAIN_EPOCHS=50 \
BATCH_SIZE=32 \
sbatch --export=EXPERIMENT_ID,BACKBONES_CSV,HEAD_TYPE,PRETRAIN_LOSS,LR,TEMPERATURE,LAMBDA_SUPPRO,LAMBDA_SUPMIN,PRETRAIN_EPOCHS,BATCH_SIZE jobscript_slurm.sh
