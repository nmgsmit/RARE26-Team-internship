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
export HEAD_TYPE='linear'
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

---

## Classifier Head Selection

The `HEAD_TYPE` variable controls which classifier head is attached to the frozen backbone during `finetune` (and `baseline`) stages. Set it before calling `sbatch`.

### Available heads

| `HEAD_TYPE` | Description | Extra variables |
|-------------|-------------|-----------------|
| `linear` | Single linear layer (simplest probe) | — |
| `ln_linear` | LayerNorm → Linear | — |
| `mlp_fullwidth` | Full-width MLP with configurable hidden layers **(default)** | — |
| `mlp_bottleneck` | Bottleneck MLP (compress → expand) | — |
| `residual_bottleneck` | Bottleneck MLP with residual connection | — |
| `cosine_linear` | Cosine similarity head with learned temperature | — |
| `knn` | k-Nearest Neighbours (no gradient training, fitted once) | `KNN_NEIGHBORS` (default `5`) |
| `svm` | SVM with RBF kernel, soft margin C (no gradient training, fitted once) | `SVM_C` (default `2.0`) |

> **How sklearn heads work**: `knn` and `svm` skip the SGD training loop entirely. After loading the pretrained encoder, all training images are passed through the frozen backbone once, and the classifier is fitted on those features. Every subsequent epoch only runs evaluation — training is already done.

### Quick examples

```bash
# Linear probe
export HEAD_TYPE="linear"
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh

# Full-width MLP (default)
export HEAD_TYPE="mlp_fullwidth"
sbatch jobscript_slurm.sh

# KNN with k=11
export HEAD_TYPE="knn"
export KNN_NEIGHBORS=11
sbatch jobscript_slurm.sh

# SVM — margin C=2 (default)
export HEAD_TYPE="svm"
sbatch jobscript_slurm.sh

# SVM — tighter margin
export HEAD_TYPE="svm"
export SVM_C=0.5
sbatch jobscript_slurm.sh
```

### Head ablation sweep

```bash
for head in linear mlp_fullwidth knn svm; do
    export EXPERIMENT_ID="head_ablation_${head}"
    export HEAD_TYPE="${head}"
    export WANDB_GROUP="head_ablation"
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
| Head type | `HEAD_TYPE` | see table above | `"svm"` |
| KNN neighbours | `KNN_NEIGHBORS` | integer | `5` |
| SVM margin C | `SVM_C` | float | `2.0` |
| ROI Positive Prob | `ROI_FOCUS_PROB` | 0.0-1.0 | `1.0` |
| ROI Negative Prob | `ROI_NEGATIVE_FOCUS_PROB` | 0.0-1.0 | `0.5` |
| ROI Warmup | `ROI_WARMUP_EPOCHS` | integer | `5` |
| ROI Min Zoom | `ROI_MIN_CROP_SCALE` | 0.0-1.0 | `0.4` |
| W&B Group | `WANDB_GROUP` | string | `"my_group"` |
