# Quick Run Commands

## Standard Run Template (Copy & Paste)

```bash
export EXPERIMENT_ID="experiment_name"
export AUGMENTATION_INTENSITY=1
export ROI_FOCUS_PROB=1.0
export ROI_NEGATIVE_FOCUS_PROB=0.0
export ROI_WARMUP_EPOCHS=0
export ROI_MIN_CROP_SCALE=0.4
export STAGES_CSV="pretrain,finetune"
export WANDB_GROUP="group_name"
sbatch jobscript_slurm.sh
```

---

## Quick Examples (Modify the Template Above)

### With Different ROI Min Crop Scale
Change: `export ROI_MIN_CROP_SCALE=0.6`

### With Warmup
Change: `export ROI_WARMUP_EPOCHS=5`

### With Negative ROI
Change: `export ROI_NEGATIVE_FOCUS_PROB=0.5` and add:
`export HARD_NEG_ROI_RECORDS_PATH="./path/to/negative_rois.json"`

### Augmentation Ablation

```bash
for intensity in 1 2 3; do
    export EXPERIMENT_ID="aug_level_${intensity}"
    export AUGMENTATION_INTENSITY=$intensity
    export WANDB_GROUP="augmentation_ablation"
    sbatch jobscript_slurm.sh
    sleep 5
done
```

## ROI Zoom Range Ablation

```bash
for scale in 0.2 0.4 0.6 0.8 1.0; do
    export EXPERIMENT_ID="roi_scale_${scale}"
    export ROI_MIN_CROP_SCALE=$scale
    export WANDB_GROUP="roi_zoom_ablation"
    sbatch jobscript_slurm.sh
    sleep 5
done
```

---

## Key Parameters

| Parameter | Env Variable | Values | Example |
|-----------|--------------|--------|---------|
| Name | `EXPERIMENT_ID` | string | `"my_exp"` |
| Augmentation | `AUGMENTATION_INTENSITY` | 1, 2, 3 | `1` |
| Stages | `STAGES_CSV` | `baseline`, `pretrain`, `finetune` | `"pretrain,finetune"` |
| ROI Positive Prob | `ROI_FOCUS_PROB` | 0.0-1.0 | `1.0` |
| ROI Negative Prob | `ROI_NEGATIVE_FOCUS_PROB` | 0.0-1.0 | `0.5` |
| ROI Warmup | `ROI_WARMUP_EPOCHS` | integer | `5` |
| ROI Min Zoom | `ROI_MIN_CROP_SCALE` | 0.0-1.0 | `0.4` |
| W&B Group | `WANDB_GROUP` | string | `"my_group"` |
