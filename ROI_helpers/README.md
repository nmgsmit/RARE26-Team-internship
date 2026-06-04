# ROI Guidance System

## Overview

The ROI (Region of Interest) guidance system enables the model to focus on pathologically relevant regions during training. ROIs are detected via Grad-CAM heatmaps and stored as bounding boxes, which are then sampled as cropped views during training.

---

## Quick Start: ROI Calibration Workflow

### Step 1: Build Grad-CAM Cache

Generate Grad-CAM heatmaps for all training and validation images using SLURM batch job submission.

#### Option A: Submit as SLURM Job (Recommended for Clusters)

Create `build_gradcam_cache.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=build_gradcam
#SBATCH --output=logs/build_gradcam_%j.log
#SBATCH --error=logs/build_gradcam_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

# Activate virtual environment
source venv/bin/activate

# Create output directories
mkdir -p ./checkpoints/roi_records
mkdir -p ./logs

# Run the build script
python ROI_helpers/build_gradcam_cache.py \
  --checkpoint ./gradcam_model.pt \
  --output-cache-path ./checkpoints/roi_records/rois.gradcam_cache.npz \
  --batch-size 8 \
  --num-workers 4

echo "Grad-CAM cache build completed!"
```

Submit the job:
```bash
sbatch build_gradcam_cache.sh
```

Monitor progress:
```bash
squeue -u $USER              # Check job status
tail -f logs/build_gradcam_*.log  # View live logs
```

**Adjust SLURM parameters in the script:**
- `--time=04:00:00` — Maximum runtime (4 hours)
- `--mem=64G` — Memory allocation
- `--cpus-per-task=8` — CPU cores
- `--gpus=1` — GPU count
- `--partition=gpu` — Queue/partition name (check with `sinfo`)

#### Option B: Run Directly (Local/Interactive)

```bash
source venv/bin/activate

python ROI_helpers/build_gradcam_cache.py \
  --checkpoint ./gradcam_model.pt \
  --output-cache-path ./checkpoints/roi_records/rois.gradcam_cache.npz
```

**Arguments:**
- `--checkpoint`: Path to your trained model checkpoint (required, e.g., `./gradcam_model.pt`)
- `--output-cache-path`: Where to save the Grad-CAM cache `.npz` file (required)
- `--data-dir`: Training data directory (default: `../data/Challenge_train_data`)
- `--target-class`: Class index for Grad-CAM (default: 1 for neo/positive class)
- `--batch-size`: Batch size for processing (default: 8)
- `--num-workers`: DataLoader worker count (default: 4)
- `--cache-dtype`: Storage precision `float16` (default) or `float32`

**Output:**
- `./checkpoints/roi_records/rois.gradcam_cache.npz` — Compressed cache containing raw Grad-CAM maps for all images

### Step 2: Calibrate ROI Thresholds

Open and run the Jupyter notebook to visualize and calibrate ROI detection:

```bash
jupyter notebook ROI_helpers/ROI_calibrate.ipynb
```

**What the notebook does:**
1. **Loads the Grad-CAM cache** from `./checkpoints/roi_records/rois.gradcam_cache.npz`
2. **Provides interactive sliders** to adjust:
   - `Threshold`: Grad-CAM activation threshold (0.0–1.0, default 0.60)
   - `Min ROI size`: Minimum island coverage (0.0–0.10, default 0.01)
3. **Displays real-time previews:**
   - Original images with Grad-CAM overlays
   - Thresholded masks and detected ROI bounding boxes
   - Cropped ROI regions that will be used during training
4. **Exports ROI records** to `./checkpoints/roi_records/rois.json`

**Workflow:**
- Adjust sliders to find optimal threshold and minimum size
- View positive and negative class previews separately
- Check ROI statistics table to understand coverage rates
- Run export cell to save configuration to JSON

The exported `rois.json` is then used during training to sample ROI-focused crops alongside full-frame views.

---

## Pipeline

### 1. ROI Generation (JSON)

**Source:** `roi_gradcam_cache_visualization.ipynb`

- Loads pre-computed Grad-CAM heatmaps (normalized to [0,1])
- Applies threshold to extract signal regions (default: 0.60)
- Identifies connected components (islands) above minimum coverage (default: 0.01)
- Stores bounding boxes at their **detected size** (tight fit around signal)

**JSON Format** (`checkpoints/roi_records/rois.json`):
```json
{
  "bbox": (0.152, 0.143, 0.393, 0.506),  // normalized [x0, y0, x1, y1]
  "center_x": 0.272,
  "center_y": 0.324,
  "roi_width": 0.241,    // fraction of image width (~24%)
  "roi_height": 0.363,   // fraction of image height (~36%)
  "coverage": 0.0346,    // signal occupies 3.46% of image
  "source": "gradcam",
  "peak_activation": 1.0,
  "mean_activation": 0.715
}
```

**Key point:** JSON stores the tight bounding box without context expansion.

---

## ROI Sampling Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ROI_CONTEXT_SCALE` | 2.0 | Expand ROI by 2.0× in each direction |
| `ROI_MIN_CROP_SCALE` | 0.40 | Minimum 40% of image as final crop size |
| `ROI_MAX_ASPECT_RATIO` | 1.5 | Limit aspect ratio to prevent extreme stretching |
| `ROI_CENTER_JITTER` | 0.05 | **Random center offset ±5% of crop size** (prevents overfitting to exact lesion center) |

### Center Jitter Explanation

`ROI_CENTER_JITTER=0.05` applies a **very small random shift** to the crop center each epoch:
- Shift amount: ±5% of the crop window size
- **Example:** For a 200px crop, shift is ±10px randomly
- **Purpose:** Prevents the model from memorizing the exact lesion center location
- **Effect:** Encourages learning robust features across the lesion region
- **Training:** Jitter is **always applied** (randomized each epoch)
- **Visualization:** Set `--center-jitter 0.05` to see jitter in previews (default 0.0 = deterministic)

### Crop Size Computation

The final crop size is computed at **sampling time** (not stored in JSON):

```
1. Regularize aspect ratio using max_aspect_ratio=1.5
2. Expand: crop_width = max(0.30, roi_width * 1.8)
3. Apply jitter: center += random_jitter * crop_width
4. Clamp to image bounds [0, 1]
```

**Example:** A signal with roi_width=0.24, roi_height=0.36 becomes:
- Expanded: crop_width ≈ 0.43, crop_height ≈ 0.65
- Final crop: ~43% × 65% of the image (approximately 1.8× larger than stored)

---

## Training Integration

**File:** `data.py` → `TwoViewDataset.__getitem__()`

For positive (neo) images with ROI records:

```python
use_roi = roi_guidance_active and label == neo and random() <= roi_focus_prob

if use_roi:
    # Generate random jitter for this epoch
    jitter_xy = random_shift * roi_center_jitter
    
    # Crop with context_scale, min_crop_scale, jitter
    image2 = crop_image_to_roi(
        roi_record=roi_record,
        context_scale=1.8,
        min_crop_scale=0.30,
        jitter_xy=jitter_xy,  # ← Random each epoch
        max_aspect_ratio=1.5
    )
    # Apply lighter augmentation to ROI crop
    transform2 = roi_transform2
else:
    # Negative samples: full-frame view
    image2 = image
    transform2 = standard_transform

return view1, transform2(image2), label
```

**Key behaviors:**
- Only applied to positive (neo) images with ROI records
- View 1: Full image with standard augmentation
- View 2: ROI crop (if available) OR full image
- Jitter applied **per sample per epoch** → different crops each time
- **Multiple islands:** If an image has multiple lesion regions (islands), **one is randomly selected each epoch**
  - Prevents overfitting to a specific lesion location
  - Encourages learning features that work across all detected regions

---

## Visualization Tools

### Static Previews (Deterministic)

**`roi_gradcam_cache_visualization.ipynb`**
- 4-column visualization: Original | Grad-CAM | Threshold mask | ROI crops
- Uses jitter_xy=(0.0, 0.0) → **no random shifts**
- Shows idealized crop appearance

**`visualize_roi_sampling.py`** (Updated to support jitter)
- Grid of crops from training images
- Default: **no jitter** (deterministic preview)
- **New:** Add `--center-jitter 0.05` to visualize with training jitter

```bash
# Deterministic preview (default)
python visualize_roi_sampling.py

# Preview with training jitter (±5%)
python visualize_roi_sampling.py --center-jitter 0.05
```

### Training Simulation (With Jitter)

**`visualize_suppro_batch.py`**
- Simulates an actual training batch (8-16 samples, 50/50 neo/ndbe)
- Shows: Original | View 1 (full aug) | View 2 (ROI crop or full)
- **Includes random jitter** → crops shift slightly (±5% by default)
- Optional: displays cosine similarity matrix between views
- **Most realistic representation of what model sees during training**

```bash
# Default: with training jitter (0.05 = ±5%)
python visualize_suppro_batch.py

# Deterministic (no jitter)
python visualize_suppro_batch.py --roi-center-jitter 0.0

# Custom jitter
python visualize_suppro_batch.py --roi-center-jitter 0.10
```

---

## Summary

| Stage | Jitter | Crops Shift? | Purpose |
|-------|--------|-------------|---------|
| JSON generation | - | N/A | Store signal locations |
| visualize_roi_sampling.py | Optional* | Configurable | Preview crops (default deterministic, add `--center-jitter 0.05` to match training) |
| visualize_suppro_batch.py | 0.05 default | ✅ Dynamic | Simulate actual training batch |
| Training (data.py) | 0.05 default | ✅ Dynamic | Actual model input |

*Default: `--center-jitter 0.0` (no jitter). Set to `0.05` to match training jitter.

**Key insight:** Jitter acts as **additional augmentation**—crops shift randomly each epoch (±5% of crop size) within the region around the detected signal, preventing the model from memorizing exact crop locations.
