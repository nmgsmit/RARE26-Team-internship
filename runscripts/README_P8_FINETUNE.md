# P8_scale095 Finetune Scripts

Two SLURM scripts for running KNN finetune stages on the P8_scale095 pretrain encoders.

## Scripts

### 1. `finetune_P8_scale095_knn_sweep.sh` (Hardcoded)

**Best for:** Running the exact two KNN configurations (k=20 and k=3) on P8_scale095.

**Usage:**
```bash
sbatch runscripts/finetune_P8_scale095_knn_sweep.sh
```

**What it does:**
- Automatically uses the P8_scale095 fold0 and fold1 encoders
- Runs KNN finetune with k=20
- Runs KNN finetune with k=3
- Builds ensembles for both configurations
- Logs to W&B (offline mode)

**Output:**
```
checkpoints/P8_scale095_knn20/
  ├── P8_scale095_knn20_finetune/
  │   ├── knn20/
  │   │   ├── P8_scale095_knn20_finetune_knn20_fold0.pt
  │   │   └── P8_scale095_knn20_finetune_knn20_fold1.pt
  │   └── ...
  └── P8_scale095_knn20_ensembles/
      └── ensemble_knn20.pt  ← Submit this

checkpoints/P8_scale095_knn3/
  ├── P8_scale095_knn3_finetune/
  │   ├── knn3/
  │   │   ├── P8_scale095_knn3_finetune_knn3_fold0.pt
  │   │   └── P8_scale095_knn3_finetune_knn3_fold1.pt
  │   └── ...
  └── P8_scale095_knn3_ensembles/
      └── ensemble_knn3.pt  ← Submit this
```

### 2. `finetune_P8_encoders_flexible.sh` (Customizable)

**Best for:** Running different KNN configurations or with different encoders.

**Required Environment Variables:**
- `ENCODER_FOLD0` - Path to fold 0 encoder checkpoint
- `ENCODER_FOLD1` - Path to fold 1 encoder checkpoint
- `KNN_K_VALUES` - Comma-separated k values (e.g., "3,5,10,20")

**Optional Environment Variables:**
- `EXPERIMENT_ID` - Prefix for output directories (default: "P8_scale095")
- `BACKBONE_PRESET` - Backbone preset (default: "gastronet")
- `INPUT_SIZE` - Input image size (default: 336)
- `BATCH_SIZE` - Batch size (default: 32)
- `DATA_DIR` - Path to training data (default: "../data/Challenge_train_data")
- `NUM_WORKERS` - Number of workers (default: 10)
- `SEED` - Random seed (default: 42)

**Usage Examples:**

*Basic - P8_scale095 with k=3 and k=20:*
```bash
ENCODER_FOLD0="./checkpoints/P8_scale095/pretrain/fold0_val_center_1/P8_scale095_pretrain_fold0_val_center_1_encoder.pt" \
ENCODER_FOLD1="./checkpoints/P8_scale095/pretrain/fold1_val_center_2/P8_scale095_pretrain_fold1_val_center_2_encoder.pt" \
KNN_K_VALUES="3,20" \
sbatch runscripts/finetune_P8_encoders_flexible.sh
```

*Advanced - Different experiment with more k values:*
```bash
ENCODER_FOLD0="./checkpoints/P8_scale095/pretrain/fold0_val_center_1/P8_scale095_pretrain_fold0_val_center_1_encoder.pt" \
ENCODER_FOLD1="./checkpoints/P8_scale095/pretrain/fold1_val_center_2/P8_scale095_pretrain_fold1_val_center_2_encoder.pt" \
KNN_K_VALUES="3,5,10,20,25,51" \
EXPERIMENT_ID="P8_knn_comprehensive" \
BATCH_SIZE=32 \
NUM_WORKERS=20 \
sbatch runscripts/finetune_P8_encoders_flexible.sh
```

*Different encoders with custom experiment ID:*
```bash
ENCODER_FOLD0="./checkpoints/other_experiment/fold0_encoder.pt" \
ENCODER_FOLD1="./checkpoints/other_experiment/fold1_encoder.pt" \
KNN_K_VALUES="5,20" \
EXPERIMENT_ID="my_custom_sweep" \
sbatch runscripts/finetune_P8_encoders_flexible.sh
```

## SLURM Configuration

Both scripts use:
- **GPU**: 1x A100
- **CPU**: 10 cores
- **Time limit**: 4 hours
- **Partition**: gpu_a100
- **Output**: `slurm_logs/finetune_*.out`

Adjust `#SBATCH` directives if your cluster differs.

## Monitoring

Check job status:
```bash
squeue -u $USER
```

Check output in real-time:
```bash
tail -f slurm_logs/finetune_P8_flexible-<jobid>.out
```

## After Completion

1. Check ensemble performance metrics in W&B (if online mode is enabled)
2. Copy desired ensemble to `model.pt`:
   ```bash
   cp checkpoints/P8_scale095_knn20/P8_scale095_knn20_ensembles/ensemble_knn20.pt model.pt
   ```
3. Build Docker image for submission:
   ```bash
   docker build -t team-internship:latest -f Submission_files/Dockerfile .
   docker save -o RARE26-submission.tar team-internship:latest
   ```

## Troubleshooting

**Error: "Encoder checkpoint not found"**
- Verify encoder paths are correct and absolute or relative to repo root
- Check file exists: `ls -lh <path>`

**Error: "No module named 'timm'"**
- The script should automatically install requirements
- If not, run manually: `pip install -r requirements.txt`

**SLURM job pending for long time**
- Check cluster load: `sinfo -a`
- Try different partition or reduce GPU request
- Contact your HPC administrator

**Out of memory errors**
- Reduce `BATCH_SIZE` (e.g., 16 or 8)
- Reduce `NUM_WORKERS`
- Monitor with: `nvidia-smi`

## W&B Logging

Scripts run in **offline mode** by default. To enable online logging:
1. Set up `.env` file with `WANDB_API_KEY=your_key`
2. Change `--wandb-mode offline` to `--wandb-mode online` in the script

## Performance Notes

- Each KNN finetune stage typically takes 5-30 minutes (depends on data size)
- Feature extraction is deterministic (same results each run)
- Ensemble building adds minimal overhead
- You can submit multiple `sbatch` commands to run sweeps in parallel (if resources allow)
