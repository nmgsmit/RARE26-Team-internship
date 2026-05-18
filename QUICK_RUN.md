# Quick Run Commands

## Basic Run

```bash
export EXPERIMENT_ID="my_experiment"
export AUGMENTATION_INTENSITY=1
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

## With ROI Min Crop Scale

```bash
export EXPERIMENT_ID="roi_zoom_test"
export ROI_MIN_CROP_SCALE=0.4
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

## With Warmup

```bash
export EXPERIMENT_ID="with_warmup"
export ROI_WARMUP_EPOCHS=5
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

## With Negative ROI

```bash
export EXPERIMENT_ID="with_neg_roi"
export ROI_NEGATIVE_FOCUS_PROB=0.5
export HARD_NEG_ROI_RECORDS_PATH="./path/to/negative_rois.json"
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

## Augmentation Ablation

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
