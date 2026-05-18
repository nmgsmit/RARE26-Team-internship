# SupPro 2-View Contrastive Learning: Complete Training Guide

This guide explains how to configure and run SupPro pretraining with the 2-view sampling strategy, augmentation intensity levels, and ROI crop controls.

---

## Overview

**SupPro (Supervised Contrastive Learning)** trains an encoder using contrastive loss that pulls same-class embeddings together and pushes different classes apart. The encoder sees two different augmented views of each image (called View1 and View2).

**Key Innovation:** The 2-view sampling strategy combines:
- **View 1:** Always a random crop with 90-100% scale → provides global context
- **View 2:** ROI-focused crop (when available) or random 90-100% crop (fallback) → provides local focus or diversity

This design balances global understanding with local detail when ROI guidance is available.

---

## 2-View Sampling Strategy

### View 1: Always Global Context

```
View 1 = Random crop at scale uniform(0.9, 1.0)
```

View 1 always provides full-context information by randomly cropping 90-100% of the image. This ensures:
- Stable, consistent representations of the full image
- Spatial context is preserved
- No extreme zooming that could create unrealistic synthetic views

### View 2: ROI-Focused or Fallback

**For Positive Images WITH ROI Record:**
```
View 2 = ROI crop at scale uniform(roi_min_crop_scale, 1.0)
         + jitter at ±5% of crop dimensions
```
When a detected lesion region (ROI) is available, View 2 zooms into that region with variable zoom level. The scale is sampled uniformly between `roi_min_crop_scale` (e.g., 0.4) and 1.0, so the model learns at multiple zoom levels.

**For Positive Images WITHOUT ROI Record:**
```
View 2 = Random crop at scale uniform(0.9, 1.0) [same as View 1]
```
When no ROI record exists, both views provide similar global context but from different random positions. This teaches spatial invariance.

**For Negative Images WITH Negative ROI Record:**
```
View 2 = Negative ROI crop at scale uniform(roi_min_crop_scale, 1.0)
         + jitter at ±5% of crop dimensions
```
Hard-negative regions (non-lesion regions that might be confusing) can also be focused on for negative samples.

**For Negative Images WITHOUT Negative ROI Record:**
```
View 2 = Random crop at scale uniform(0.9, 1.0) [same as View 1]
```
Both views provide global context for negative samples.

---

## Augmentation Intensity Levels

The **augmentation intensity** controls the strength of geometric transformations (flips, rotations) and color jitter applied to each view. Three levels are available:

### Level 1: LOW (Conservative) - Recommended for Endoscopy

**Best for:** Endoscopic images where anatomical orientation matters (vascular patterns, mucosal folds have fixed directions)

```
Horizontal Flip:      10% probability
Vertical Flip:        0% (disabled - preserves mucosal fold directionality)
Rotation:             ±2 degrees
Color Jitter:         brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01
```

**Philosophy:** Minimal geometric distortion preserves anatomical signals while still introducing color/lighting variability for robustness.

**When to use:**
- You have directional anatomical features
- You want to prevent learning on unrealistic synthetic augmentations
- You prefer conservative, data-preserving training

### Level 2: MEDIUM (Balanced)

**Best for:** General-purpose training with moderate augmentation

```
Horizontal Flip:      30% probability
Vertical Flip:        10% probability
Rotation:             ±5 degrees
Color Jitter:         brightness=0.08, contrast=0.08, saturation=0.08, hue=0.015
```

**Philosophy:** Moderate geometric distortion balanced with color variability. Good compromise between robustness and realism.

**When to use:**
- You're unsure about the best augmentation strategy
- You want a middle ground between robustness and realism
- You're running exploratory experiments

### Level 3: STRONG (Aggressive) - Previous Default

**Best for:** Maximum robustness when data is very limited

```
Horizontal Flip:      50% probability
Vertical Flip:        20% probability
Rotation:             ±10 degrees
Color Jitter:         brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
```

**Philosophy:** Aggressive geometric and color augmentation teaches maximum invariance.

**When to use:**
- You have very limited training data and need strong regularization
- Rotation invariance is critical for your application
- You need to match the original paper's settings

**Caution for endoscopy:** Level 3 might create too many unrealistic synthetic images due to extreme flips and rotations.

---

## Control Parameters

### ROI Focus Probability (Positive Images)

```bash
export ROI_FOCUS_PROB=1.0  # Default
```

Controls how often View 2 uses an ROI crop for **positive samples** (when available):
- `1.0` → Always use ROI crop if available (no random fallback)
- `0.5` → 50% chance to use ROI crop, 50% chance to use random full-context crop
- `0.0` → Never use ROI crops, always use random full-context crops (essentially disables ROI guidance)

### ROI Negative Focus Probability (Negative Images)

```bash
export ROI_NEGATIVE_FOCUS_PROB=0.0  # Default (disabled)
```

Controls how often View 2 uses a negative ROI crop for **negative samples** (hard negatives, when available):
- `1.0` → Always use negative ROI crop if available
- `0.5` → 50% chance to use negative ROI crop
- `0.0` → Disabled (default) - negative samples always get random full-context crops

**Use case:** When you have detected hard-negative regions (e.g., areas that might be confused with lesions), enable this to make the model more discriminative.

### ROI Warmup Epochs

```bash
export ROI_WARMUP_EPOCHS=0  # Default (no warmup)
```

Number of initial epochs to skip ROI cropping and use **only random full-context crops for both views**:
- `0` → No warmup, ROI crops start from epoch 0
- `5` → First 5 epochs use no crops (both views are random 90-100% crops), ROI crops start at epoch 5

**Use case:** Train the model on general features first (warmup), then refine with ROI guidance. This can help prevent overfitting to specific ROI regions early in training.

### Augmentation Intensity

```bash
export AUGMENTATION_INTENSITY=1  # 1=low, 2=medium, 3=strong
```

Controls the strength of flips, rotations, and color jitter (see Augmentation Intensity Levels above).

### ROI Min Crop Scale

```bash
export ROI_MIN_CROP_SCALE=0.4  # Default (40% zoom minimum)
```

Minimum normalized crop size for ROI crops. Controls the minimum zoom level:
- `0.2` → ROI crops zoom from 20% to 100% (aggressive zoom range)
- `0.4` → ROI crops zoom from 40% to 100% (moderate zoom range)
- `0.8` → ROI crops zoom from 80% to 100% (minimal zoom range)
- `1.0` → ROI crops are disabled (no zoom, equivalent to full image)

---

## Quick Start Examples

### Recommended Baseline: Endoscopy with Low Augmentation

```bash
export EXPERIMENT_ID="endoscopy_baseline_lowAug"
export AUGMENTATION_INTENSITY=1
export ROI_FOCUS_PROB=1.0
export ROI_NEGATIVE_FOCUS_PROB=0.0
export ROI_WARMUP_EPOCHS=0
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

This uses conservative augmentation suitable for endoscopic images with anatomical directionality.

### With Warmup: Pretrain General Features First

```bash
export EXPERIMENT_ID="endoscopy_with_warmup"
export AUGMENTATION_INTENSITY=1
export ROI_FOCUS_PROB=1.0
export ROI_WARMUP_EPOCHS=5  # First 5 epochs: no ROI crops
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

First 5 epochs learn general features without focusing on ROIs, then switch to ROI guidance.

### With Hard Negatives: Add Negative ROI Guidance

```bash
export EXPERIMENT_ID="endoscopy_with_hard_negatives"
export AUGMENTATION_INTENSITY=1
export ROI_FOCUS_PROB=1.0
export ROI_NEGATIVE_FOCUS_PROB=0.5  # 50% chance to use hard-negative ROIs
export HARD_NEG_ROI_RECORDS_PATH="./path/to/negative_rois.json"
export STAGES_CSV="pretrain,finetune"
sbatch jobscript_slurm.sh
```

Leverages detected hard-negative regions to make the model more discriminative.

### Augmentation Ablation: Test All Three Levels

```bash
#!/bin/bash

for intensity in 1 2 3; do
    export EXPERIMENT_ID="aug_level_${intensity}"
    export AUGMENTATION_INTENSITY=$intensity
    export WANDB_GROUP="augmentation_ablation"
    export STAGES_CSV="pretrain,finetune"
    
    echo "Running with augmentation intensity: $intensity"
    sbatch jobscript_slurm.sh
    sleep 5  # Space out submissions
done
```

Run all three augmentation levels and compare in W&B's augmentation_ablation group.

### ROI Min Crop Scale Ablation: Test Different Zoom Levels

```bash
#!/bin/bash

for min_crop_scale in 0.2 0.4 0.6 0.8 1.0; do
    export EXPERIMENT_ID="roi_zoom_${min_crop_scale}"
    export ROI_MIN_CROP_SCALE=$min_crop_scale
    export WANDB_GROUP="roi_zoom_ablation"
    export STAGES_CSV="pretrain,finetune"
    
    echo "Running with ROI_MIN_CROP_SCALE: $min_crop_scale"
    sbatch jobscript_slurm.sh
    sleep 5
done
```

Test different ROI zoom ranges to find the optimal trade-off between focus and context.

---

## How to Use main.sh

### Setting Environment Variables

All parameters are controlled via environment variables. Set them before submitting:

```bash
# Option 1: Direct export (one-off run)
export AUGMENTATION_INTENSITY=1
export ROI_FOCUS_PROB=1.0
sbatch jobscript_slurm.sh

# Option 2: In a bash script for ablations
#!/bin/bash
for level in 1 2 3; do
    export AUGMENTATION_INTENSITY=$level
    sbatch jobscript_slurm.sh
    sleep 5
done
```

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EXPERIMENT_ID` | `name` | Checkpoint base name |
| `AUGMENTATION_INTENSITY` | `3` | 1=low, 2=medium, 3=strong |
| `ROI_FOCUS_PROB` | `1.0` | Probability of using positive ROI crops |
| `ROI_NEGATIVE_FOCUS_PROB` | `0.0` | Probability of using negative ROI crops |
| `ROI_WARMUP_EPOCHS` | `0` | Epochs to skip ROI crops (warmup) |
| `ROI_MIN_CROP_SCALE` | `0.4` | Minimum zoom level for ROI crops |
| `ROI_CONTEXT_SCALE` | `2.0` | Context multiplier around ROI |
| `ROI_CENTER_JITTER` | `0.05` | ±5% random offset for ROI centers |
| `ROI_RECORDS_PATH` | `./checkpoints/roi_records/rois.json` | Path to positive ROI records |
| `HARD_NEG_ROI_RECORDS_PATH` | `` | Path to negative ROI records |
| `STAGES_CSV` | `pretrain,finetune` | Which training stages to run |

---

## Understanding View Creation: Step-by-Step Example

### Scenario 1: Positive Image WITH ROI Record

```
Input: Endoscopic image with detected lesion ROI

Step 1: Load original image
  image = RGB image (e.g., 800x600 pixels)
  label = 1 (positive/lesion)

Step 2: Create View 1 (Global Context)
  scale1 = uniform(0.9, 1.0) = 0.95  [Random sample each iteration]
  image1 = random crop at 95% of image
  → Mostly full image with slight border clipping
  → Apply transform1 (augmentation level 1, 2, or 3)
  → Result: 336x336 tensor with global context

Step 3: Check ROI Availability
  roi_record = FOUND (positive ROI guidance active, label matches)
  
Step 4: Create View 2 (ROI-Focused)
  roi_scale = uniform(roi_min_crop_scale, 1.0) = uniform(0.4, 1.0) = 0.68
  jitter = random ±5% offset = (-0.03, +0.027)
  image2 = crop_image_to_roi(lesion_center, zoom=0.68, jitter=jitter)
  → Crops to detected lesion region at 68% zoom + random offset
  → Apply roi_transform2 (same augmentation level)
  → Result: 336x336 tensor zoomed into lesion

Step 5: Return Pair
  return (view1, view2, label=1)
  
The model learns: "This lesion in full context (view1) == Same lesion zoomed (view2)"
→ Teaches multi-scale robustness
```

### Scenario 2: Positive Image WITHOUT ROI Record

```
Input: Endoscopic image (lesion present, but no ROI record available)

Step 1-2: View 1 creation same as above

Step 3-4: Check ROI Availability
  roi_record = NOT FOUND (no ROI guidance available)
  use_roi = False

Step 5: Create View 2 (Fallback Global Context)
  scale2 = uniform(0.9, 1.0) = 0.92
  image2 = random crop at 92% of image
  → Different random position than view1
  → Apply transform2 (same augmentation level)
  → Result: 336x336 tensor with full context

Step 6: Return Pair
  return (view1, view2, label=1)
  
The model learns: "Same image content at different positions == same class (lesion)"
→ Teaches spatial invariance without relying on detected ROI
```

### Scenario 3: Negative Image

```
Input: Endoscopic image without lesion (negative/NDBE)

Step 1-2: View 1 creation same as above

Step 3-4: Check Negative ROI Availability
  neg_roi_record = NOT FOUND (or roi_negative_focus_prob=0 so disabled)
  use_neg_roi = False

Step 5: Create View 2 (Random Global Context)
  scale2 = uniform(0.9, 1.0) = 0.91
  image2 = random crop at 91% of image
  → Apply transform2 (same augmentation level)
  → Result: 336x336 tensor with full context

Step 6: Return Pair
  return (view1, view2, label=0)
  
The model learns: "Different parts of negative image == same class (not a lesion)"
→ Teaches negative class coherence
```

---

## Warmup Phase Example

### Without Warmup (Default)

```
Epoch 0: ROI crops active (if available)
Epoch 1: ROI crops active
Epoch 2: ROI crops active
...
```

The model starts with ROI-focused training immediately.

### With ROI_WARMUP_EPOCHS=5

```
Epoch 0: NO ROI crops → both views are random 90-100% crops
Epoch 1: NO ROI crops → both views are random 90-100% crops
Epoch 2: NO ROI crops → both views are random 90-100% crops
Epoch 3: NO ROI crops → both views are random 90-100% crops
Epoch 4: NO ROI crops → both views are random 90-100% crops
Epoch 5: ROI crops activate → positive/negative ROI guidance starts
Epoch 6: ROI crops active
...
```

During warmup, the model learns general features. After warmup, it refines with ROI guidance.

---

## Interpreting W&B Results

### What to Look For

1. **Loss Curve:**
   - Should decrease smoothly
   - Low augmentation (level 1) might have slower decay but smoother curve
   - High augmentation (level 3) might decay faster but with more noise

2. **Validation AUC:**
   - Compare across augmentation levels
   - Pick the level with best generalization (test set performance)

3. **Augmentation Intensity Effect:**
   | Level | Expected Behavior |
   |-------|-------------------|
   | 1 (Low) | Stable training, smooth curves, conservative generalization |
   | 2 (Med) | Balanced, moderate noise, good generalization |
   | 3 (Strong) | Fast decay, noisy curves, risk of overfitting |

4. **ROI Warmup Effect:**
   - With warmup: Loss might increase slightly after epoch when ROI kicks in (model adapting to new signal)
   - Then continue decreasing as it learns ROI patterns
   - Compare validation AUC with/without warmup to see if it helps

---

## Common Configurations

### For Endoscopy (Recommended Starting Point)

```bash
export AUGMENTATION_INTENSITY=1          # Conservative (no vertical flips)
export ROI_FOCUS_PROB=1.0                # Always use positive ROI when available
export ROI_NEGATIVE_FOCUS_PROB=0.0       # Disabled (no hard negatives yet)
export ROI_WARMUP_EPOCHS=0               # No warmup (start with ROI immediately)
export ROI_MIN_CROP_SCALE=0.4            # Standard zoom range
```

### If Validation is Poor: Try More Augmentation

```bash
export AUGMENTATION_INTENSITY=2          # Try medium (more flips/rotations)
export ROI_FOCUS_PROB=0.8                # Reduce ROI focus probability
export ROI_WARMUP_EPOCHS=3               # Add warmup to learn general features first
```

### If Negative Discrimination is Poor: Add Hard Negatives

```bash
export ROI_NEGATIVE_FOCUS_PROB=0.5       # Enable hard-negative ROI guidance
export HARD_NEG_ROI_RECORDS_PATH="./path/to/hard_neg_rois.json"
```

### For Maximum Robustness with Limited Data

```bash
export AUGMENTATION_INTENSITY=3          # Aggressive augmentation
export ROI_FOCUS_PROB=1.0
export ROI_MIN_CROP_SCALE=0.2            # Wider zoom range
```

---

## Troubleshooting

**Q: How do I check if augmentation intensity is actually changing?**
A: Look at the W&B config section. You should see `"augmentation_intensity": 1` (or 2, or 3).

**Q: Should I use ROI warmup?**
A: Start without it (default 0). Only add if:
1. Your model seems to overfit to ROI regions early
2. Your validation accuracy plateaus quickly
3. Your test performance is significantly worse than validation

Try `ROI_WARMUP_EPOCHS=5` and compare validation metrics.

**Q: What if all three augmentation levels give similar results?**
A: This suggests augmentation isn't your bottleneck. Focus on:
- Data quality and quantity
- ROI record accuracy
- Model architecture
- Training duration

**Q: How do I ablate ROI min crop scale?**
A: See "ROI Min Crop Scale Ablation" example above. Test values like 0.2, 0.4, 0.6, 0.8, 1.0.

**Q: Can I change augmentation intensity mid-training?**
A: No. It's set at dataset initialization. Submit a new job with a different level.

---

## Summary

| Aspect | Default | Recommended | Purpose |
|--------|---------|-------------|---------|
| **Augmentation Intensity** | 3 (strong) | 1 (low) | Controls flip/rotation/color strength |
| **ROI Focus Prob** | 1.0 | 1.0 | Probability of using positive ROI crops |
| **Negative ROI Focus Prob** | 0.0 | 0.0 (unless hard negs available) | Probability of using negative ROI crops |
| **ROI Warmup Epochs** | 0 | 0 (try 5 if overfitting) | Initial epochs without ROI crops |
| **ROI Min Crop Scale** | 0.4 | 0.4 | Minimum zoom level for ROI crops |

**For endoscopy, start with:** `AUGMENTATION_INTENSITY=1`, use warmup only if needed, keep other defaults.

---

## References

- **Paper:** [Insert relevant contrastive learning papers]
- **Dataset:** RARE Challenge dataset (endoscopic images)
- **Loss:** SupPro (Supervised Prototypical) contrastive learning
- **Architecture:** Vision Transformer (ViT) or ResNet encoders
