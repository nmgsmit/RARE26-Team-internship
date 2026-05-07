# Team Internship - AIES Master Program

This repository contains code and resources for the Team Internship project as part of the Master's program in Artificial Intelligence and Engineering Systems (AIES) at TU Eindhoven.

Example run command:
 STAGES_CSV='pretrain,finetune' \
 sbatch --export=ALL,\
 EXPERIMENT_ID=P1_BB_GastronetDinoV2_t1,\
 BACKBONES_CSV=gastronet,\
 TEMPERATURE=0.07,\
 PRETRAIN_BACKBONE_LR=1e-5,\
 PRETRAIN_PROJ_LR=3e-4,\
 FINETUNE_LR=3e-4,\
 BATCH_SIZE=32,\
 PRETRAIN_LOSS=suppro,\
 WANDB_GROUP=backbonsuppo,\
 EXPERIMENT_SAVE_SUBDIR=report_t1 \
 jobscript_slurm.sh
