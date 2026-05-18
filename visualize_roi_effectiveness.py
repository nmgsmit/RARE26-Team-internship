"""
Visualize ROI selection effectiveness for positive and negative samples.

Shows:
- Original image with Grad-CAM ROI bbox (if available)
- View 1: Random crop at 0.9-1.0 scale
- View 2a: ROI crop at 0.4-0.8 scale + jitter (if available)
- View 2b: Fallback random crop at 0.9-1.0 scale (if no ROI or as alternative)

Each visualization shows the extracted crop windows on the original image.

Usage:
    # Visualize positive samples (requires rois.json)
    python visualize_roi_effectiveness.py --label 1 --output-dir outputs/positive_roi

    # Visualize negative samples (requires hard_neg_rois.json)
    python visualize_roi_effectiveness.py --label 0 --output-dir outputs/negative_roi

    # Specify custom ROI path
    python visualize_roi_effectiveness.py --label 1 --roi-json custom_rois.json
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import torch

from data import DEFAULT_DATA_DIR, build_dataset_dataframe
from roi_guidance import (
    canonicalize_image_path,
    crop_image_to_roi,
    compute_crop_window_from_roi,
    load_roi_records_from_json,
    DEFAULT_ROI_MAX_ASPECT_RATIO,
)


DEFAULT_ROI_JSON = "./checkpoints/roi_records/rois.json"
# For negatives, use the same file but filter by path containing \ndbe\ or /ndbe/
DEFAULT_OUTPUT_DIR = "outputs/roi_effectiveness"

ROI_BOX_COLOR = "#ffdd00"  # Yellow for ROI bbox
VIEW1_CROP_COLOR = "#00ff00"  # Green for View 1 crop
VIEW2_CROP_COLOR = "#ff00ff"  # Magenta for View 2 crop (ROI-based)
FALLBACK_CROP_COLOR = "#00ffff"  # Cyan for View 2 fallback (random)

POSITIVE_COLOR = "#e05555"  # Red
NEGATIVE_COLOR = "#5577cc"  # Blue


def sample_random_crop_scale(min_scale=0.9, max_scale=1.0):
    """Sample random crop scale."""
    return float(torch.empty(1).uniform_(min_scale, max_scale).item())


def sample_roi_crop_scale(min_scale=0.4, max_scale=0.8):
    """Sample ROI crop scale."""
    return float(torch.empty(1).uniform_(min_scale, max_scale).item())


def sample_jitter(center_jitter=0.05):
    """Sample random center jitter."""
    return (
        (2.0 * float(torch.rand(1).item()) - 1.0) * center_jitter,
        (2.0 * float(torch.rand(1).item()) - 1.0) * center_jitter,
    )


def get_crop_window(image, scale, jitter_xy=(0, 0)):
    """Compute crop window for a given scale and jitter."""
    w, h = image.size
    crop_w = max(1, int(scale * w))
    crop_h = max(1, int(scale * h))

    # Random position
    left = int(torch.randint(0, max(1, w - crop_w + 1), (1,)).item())
    top = int(torch.randint(0, max(1, h - crop_h + 1), (1,)).item())

    return (left, top, left + crop_w, top + crop_h)


def draw_rect_on_image(image_path, rects_with_colors, output_path, title=""):
    """Draw rectangles on image and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw rectangles
    for rect, color, label, line_width in rects_with_colors:
        # Parse color (hex to RGB)
        color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if "fallback" in label or "Fallback" in label:
            # Dashed line effect: draw multiple small segments
            x0, y0, x1, y1 = rect
            dash_length = 5
            for x in range(int(x0), int(x1), dash_length * 2):
                draw.rectangle(
                    [(x, y0), (min(x + dash_length, x1), y0 + line_width)],
                    fill=color_rgb + (200,)
                )
                draw.rectangle(
                    [(x, y1 - line_width), (min(x + dash_length, x1), y1)],
                    fill=color_rgb + (200,)
                )
                draw.rectangle(
                    [(x0, y), (x0 + line_width, min(y + dash_length, y1))],
                    fill=color_rgb + (200,)
                )
                draw.rectangle(
                    [(x1 - line_width, y), (x1, min(y + dash_length, y1))],
                    fill=color_rgb + (200,)
                )
        else:
            # Solid line
            draw.rectangle(rect, outline=color_rgb + (255,), width=line_width)

    # Add title
    if title:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), title, fill=(255, 255, 255))

    img.save(output_path)
    return img


def visualize_sample(img_path, label, roi_record, output_path, seed=None):
    """Visualize a single sample with View 1 and View 2 sampling."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # View 1: Random crop at 0.9-1.0
    view1_scale = sample_random_crop_scale(0.9, 1.0)
    view1_rect = get_crop_window(img, view1_scale)

    # View 2a: ROI crop (if available)
    view2a_rect = None
    if roi_record and "bbox" in roi_record:
        view2a_scale = sample_roi_crop_scale(0.4, 0.8)
        jitter_xy = sample_jitter(0.05)
        try:
            # Get the crop window using compute_crop_window_from_roi
            left, top, right, bottom = compute_crop_window_from_roi(
                image_size=(w, h),
                roi_record=roi_record,
                context_scale=2.0,
                min_crop_scale=view2a_scale,
                jitter_xy=jitter_xy,
                max_aspect_ratio=DEFAULT_ROI_MAX_ASPECT_RATIO,
            )
            view2a_rect = (left, top, right, bottom)
        except:
            view2a_rect = None

    # View 2b: Fallback random crop at 0.9-1.0
    view2b_scale = sample_random_crop_scale(0.9, 1.0)
    view2b_rect = get_crop_window(img, view2b_scale)

    # Create visualization with matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{'POSITIVE' if label == 1 else 'NEGATIVE'} | {Path(img_path).name}",
        fontsize=14,
        fontweight='bold',
        color=POSITIVE_COLOR if label == 1 else NEGATIVE_COLOR
    )

    # Original image with ROI bbox
    ax = axes[0]
    ax.imshow(img)
    ax.set_title("Original + ROI BBox")

    if roi_record:
        # Draw Grad-CAM ROI bbox (normalized coordinates from JSON)
        bbox_norm = roi_record.get("bbox")  # [x0_norm, y0_norm, x1_norm, y1_norm]

        if bbox_norm and len(bbox_norm) == 4:
            # Convert normalized coordinates to pixel coordinates
            x0_norm, y0_norm, x1_norm, y1_norm = bbox_norm
            x0_px = int(x0_norm * w)
            y0_px = int(y0_norm * h)
            x1_px = int(x1_norm * w)
            y1_px = int(y1_norm * h)

            rect = patches.Rectangle(
                (x0_px, y0_px), x1_px - x0_px, y1_px - y0_px,
                linewidth=2, edgecolor=ROI_BOX_COLOR, facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            ax.text(x0_px, max(0, y0_px - 10), "Grad-CAM ROI", color=ROI_BOX_COLOR, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        else:
            ax.text(10, 30, "No ROI bbox in record", color='white', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    else:
        ax.text(10, 30, "No ROI available", color='white', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.axis('off')

    # View 1
    ax = axes[1]
    ax.imshow(img)
    ax.set_title(f"View 1: Random Crop (0.9-1.0)\nScale: {view1_scale:.2f}")

    x0, y0, x1, y1 = view1_rect
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor=VIEW1_CROP_COLOR, facecolor='none', linestyle='-'
    )
    ax.add_patch(rect)

    ax.axis('off')

    # View 2
    ax = axes[2]
    ax.imshow(img)

    if view2a_rect and roi_record:
        ax.set_title(f"View 2: ROI Crop (0.4-0.8)\nOr Fallback Random (0.9-1.0)")

        # Draw both options
        x0, y0, x1, y1 = view2a_rect
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=VIEW2_CROP_COLOR, facecolor='none', linestyle='-',
            label='View 2a (ROI)'
        )
        ax.add_patch(rect)

        x0, y0, x1, y1 = view2b_rect
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=FALLBACK_CROP_COLOR, facecolor='none', linestyle='--',
            label='View 2b (Fallback)'
        )
        ax.add_patch(rect)

        ax.legend(loc='upper right')
    else:
        ax.set_title(f"View 2: Random Crop (0.9-1.0)\nNo ROI available")
        x0, y0, x1, y1 = view2b_rect
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=FALLBACK_CROP_COLOR, facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        type=int,
        required=True,
        choices=[0, 1],
        help="0 for negative samples, 1 for positive samples"
    )
    parser.add_argument(
        "--roi-json",
        type=str,
        default=DEFAULT_ROI_JSON,
        help="Path to ROI JSON file (contains both positive [neo] and negative [ndbe] ROI records)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load dataset
    df, _ = build_dataset_dataframe(DEFAULT_DATA_DIR)

    # Filter by label from dataset
    df_filtered = df[df["label"] == args.label].reset_index(drop=True)

    if len(df_filtered) == 0:
        print(f"No samples found with label {args.label}")
        return

    # Load ROI records (both positive and negative are in the same file)
    roi_records = {}
    roi_json_path = args.roi_json
    label_name = "POSITIVE" if args.label == 1 else "NEGATIVE"

    if Path(roi_json_path).exists():
        all_roi_records, _ = load_roi_records_from_json(roi_json_path)

        # Filter by label based on path: neo = positive (1), ndbe = negative (0)
        path_filter = ("\\neo\\" if args.label == 1 else "\\ndbe\\", "/neo/" if args.label == 1 else "/ndbe/")

        for image_path, record in all_roi_records.items():
            if path_filter[0] in image_path or path_filter[1] in image_path:
                roi_records[image_path] = record

        print(f"Loaded {len(roi_records)} ROI records for {label_name} samples from {roi_json_path}")

        # Filter df to only include samples that exist in the filtered ROI records
        # (This ensures the paths match what's in the JSON)
        canonical_paths_in_roi = set(roi_records.keys())
        df_filtered["canonical_img"] = df_filtered["img"].astype(str).map(canonicalize_image_path)
        df_filtered = df_filtered[df_filtered["canonical_img"].isin(canonical_paths_in_roi)].reset_index(drop=True)

        if len(df_filtered) == 0:
            print(f"No {label_name} samples found with ROI records in {roi_json_path}")
            return

        print(f"Found {len(df_filtered)} {label_name} samples with ROI guidance available")
    else:
        print(f"ROI JSON not found at {roi_json_path}, will show all {label_name} samples without ROI guidance")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize samples
    num_to_viz = min(args.num_samples, len(df_filtered))
    indices = random.sample(range(len(df_filtered)), num_to_viz)

    print(f"\nVisualizing {num_to_viz} {label_name} samples...")

    for i, idx in enumerate(indices):
        row = df_filtered.iloc[idx]
        img_path = row["img"]

        canonical_path = canonicalize_image_path(img_path)
        roi_record = roi_records.get(canonical_path)

        sample_name = Path(img_path).stem
        output_path = output_dir / f"{label_name.lower()}_{i:02d}_{sample_name}.png"

        try:
            visualize_sample(img_path, args.label, roi_record, output_path, seed=args.seed + i)
            print(f"  [{i+1}/{num_to_viz}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{num_to_viz}] Error visualizing {img_path}: {e}")

    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
