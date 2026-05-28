# SupPro Training Configuration

## 2-View Sampling

**View 1:** Random crop at scale uniform(0.9, 1.0)

**View 2:** 
- **With ROI available** (Positive or Negative): ROI crop at scale uniform(roi_min_crop_scale, 1.0) + jitter
- **Without ROI, ROI guidance ACTIVE**: Random crop at scale uniform(roi_min_crop_scale, roi_max_crop_scale) = uniform(0.4, 1.0)
  - Both positive and negative samples use same scale range as ROI crops (for fair comparison)
- **Without ROI, ROI guidance INACTIVE**: Random crop at scale uniform(0.9, 1.0) (full-context)
  - Used when no ROI records available at all

---

## Augmentation Intensity Levels

| Level | Name | H-Flip | V-Flip | Rotation | Color Jitter | Use Case |
|-------|------|--------|--------|----------|--------------|----------|
| 1 | Low (Conservative) | 10% | 0% | ±2° | 0.05 | Light augmentation, minimal distortion |
| 2 | Medium (Balanced) | 30% | 10% | ±5° | 0.08 | Balanced between robustness and realism |
| 3 | Strong (Aggressive) | 50% | 20% | ±10° | 0.1 | **Default** - Good regularization |
| 4 | Extreme (Very Aggressive) | 50% | 50% | ±45° | 0.2 | Hard augmentation studies, stress testing |

**ROI-Specific Augmentation:** ROI crops use lighter augmentation than full-frame crops (especially rotation and color jitter).

Set via: `export AUGMENTATION_INTENSITY=3` (default: 3)

### Detailed Augmentation Parameters

Full parameter sets for each intensity level (from `data.py` AUGMENTATION_PRESETS):

| Parameter | Level 1 | Level 2 | Level 3 | Level 4 |
|-----------|---------|---------|---------|---------|
| **Full-Frame Augmentation** | | | | |
| Horizontal Flip | 0.1 | 0.3 | 0.5 | 0.5 |
| Vertical Flip | 0.0 | 0.1 | 0.2 | 0.5 |
| Rotation Range | ±2° | ±5° | ±10° | ±45° |
| Brightness Jitter | 0.05 | 0.08 | 0.1 | 0.2 |
| Contrast Jitter | 0.05 | 0.08 | 0.1 | 0.2 |
| Saturation Jitter | 0.05 | 0.08 | 0.1 | 0.2 |
| Hue Jitter | 0.01 | 0.015 | 0.02 | 0.03 |
| **ROI-Specific Augmentation** | | | | |
| Rotation Range | ±2° | ±3° | ±5° | ±15° |
| Brightness Jitter | 0.02 | 0.03 | 0.03 | 0.05 |
| Contrast Jitter | 0.02 | 0.03 | 0.03 | 0.05 |
| Saturation Jitter | 0.02 | 0.03 | 0.03 | 0.05 |
| Hue Jitter | 0.003 | 0.005 | 0.005 | 0.007 |

**Note:** ROI crops intentionally use lighter augmentation than full-frame crops to preserve the ROI region details while still providing some regularization.

---

## Control Parameters

| Parameter | Environment Variable | Argument | Default | Range |
|-----------|---------------------|----------|---------|-------|
| Augmentation Intensity | `AUGMENTATION_INTENSITY` | `--augmentation-intensity` | 3 | 1, 2, 3, 4 |
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
