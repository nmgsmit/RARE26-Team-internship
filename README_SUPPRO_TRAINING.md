# SupPro Training Configuration

## 2-View Sampling

**View 1:** Random crop at scale uniform(0.9, 1.0)

**View 2:** 
- Positive with ROI: ROI crop at scale uniform(roi_min_crop_scale, 1.0) + jitter
- Positive without ROI: Random crop at scale uniform(0.9, 1.0)
- Negative with ROI: Negative ROI crop at scale uniform(roi_min_crop_scale, 1.0) + jitter
- Negative without ROI: Random crop at scale uniform(0.9, 1.0)

---

## Augmentation Intensity Levels

| Level | H-Flip | V-Flip | Rotation | Color Jitter |
|-------|--------|--------|----------|--------------|
| 1 (Low) | 10% | 0% | ±2° | 0.05 |
| 2 (Med) | 30% | 10% | ±5° | 0.08 |
| 3 (Strong) | 50% | 20% | ±10° | 0.1 |

Set via: `export AUGMENTATION_INTENSITY=1` (default: 3)

---

## Control Parameters

| Parameter | Environment Variable | Argument | Default | Range |
|-----------|---------------------|----------|---------|-------|
| Augmentation Intensity | `AUGMENTATION_INTENSITY` | `--augmentation-intensity` | 3 | 1, 2, 3 |
| Positive ROI Probability | `ROI_FOCUS_PROB` | `--roi-focus-prob` | 1.0 | [0, 1] |
| Negative ROI Probability | `ROI_NEGATIVE_FOCUS_PROB` | `--roi-negative-focus-prob` | 0.0 | [0, 1] |
| ROI Warmup Epochs | `ROI_WARMUP_EPOCHS` | `--roi-warmup-epochs` | 0 | >= 0 |
| ROI Min Crop Scale | `ROI_MIN_CROP_SCALE` | `--roi-min-crop-scale` | 0.4 | (0, 1] |
| ROI Context Scale | `ROI_CONTEXT_SCALE` | `--roi-context-scale` | 2.0 | > 0 |
| ROI Center Jitter | `ROI_CENTER_JITTER` | `--roi-center-jitter` | 0.05 | >= 0 |
| ROI Max Aspect Ratio | `ROI_MAX_ASPECT_RATIO` | `--roi-max-aspect-ratio` | 1.5 | >= 1.0 |

---

## Usage

### Set parameters via environment variables:
```bash
export AUGMENTATION_INTENSITY=1
export ROI_FOCUS_PROB=1.0
export ROI_NEGATIVE_FOCUS_PROB=0.5
export ROI_WARMUP_EPOCHS=5
export ROI_MIN_CROP_SCALE=0.4
sbatch jobscript_slurm.sh
```

### ROI Warmup Behavior

During first N epochs (0 to roi_warmup_epochs-1): ROI crops disabled, both views use random crops.

Starting from epoch roi_warmup_epochs: ROI crops activated if available.

### Implementation Details

**data.py:**
- `TwoViewDataset` accepts `roi_negative_focus_prob` and `roi_warmup_epochs` parameters
- `set_epoch(epoch)` method tracks current epoch for warmup control
- View 2 checks `current_epoch < roi_warmup_epochs` before using ROI crops

**train.py:**
- Added `--roi-negative-focus-prob` and `--roi-warmup-epochs` arguments
- Calls `train_ds.set_epoch(epoch)` at start of each epoch

**main.sh:**
- Environment variables passed to train.py via `build_common_args()`
