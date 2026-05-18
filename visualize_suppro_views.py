"""
Visualize exact SupPro 2-view pairs with real augmentations.

Shows what the SupPro loss actually sees during training in a grid:
- Rows: samples (default 5)
- Columns: Original+ROI | View1 | View2
- View1: random crop 0.9-1.0
- View2: ROI crop (if available) or random fallback
- Uses exact training augmentations at selectable intensity level

Usage:
    # Positive samples with low augmentation
    python visualize_suppro_views.py --label 1 --augmentation-intensity 1

    # Negative samples with strong augmentation
    python visualize_suppro_views.py --label 0 --augmentation-intensity 3

    # Custom number of samples
    python visualize_suppro_views.py --label 1 --num-samples 10 --augmentation-intensity 2
"""

import argparse
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

from data import (
    DEFAULT_DATA_DIR,
    build_dataset_dataframe,
    build_roi_focus_transform,
)
from roi_guidance import (
    canonicalize_image_path,
    crop_image_to_roi,
    load_roi_records_from_json,
    DEFAULT_ROI_MAX_ASPECT_RATIO,
)


DEFAULT_ROI_JSON = "./checkpoints/roi_records/rois.json"
DEFAULT_OUTPUT_DIR = "outputs/suppro_views"

ROI_BOX_COLOR = (255, 221, 0)  # Yellow (RGB)
TILE_SIZE = 336


def sample_crop_scale(min_scale, max_scale):
    """Sample random crop scale."""
    return float(torch.empty(1).uniform_(min_scale, max_scale).item())


def sample_jitter(center_jitter=0.05):
    """Sample random center jitter."""
    return (
        (2.0 * float(torch.rand(1).item()) - 1.0) * center_jitter,
        (2.0 * float(torch.rand(1).item()) - 1.0) * center_jitter,
    )


def get_random_crop(image, min_scale, max_scale):
    """Get random crop at specified scale."""
    scale = sample_crop_scale(min_scale, max_scale)
    w, h = image.size
    crop_w = max(1, int(scale * w))
    crop_h = max(1, int(scale * h))

    left = int(torch.randint(0, max(1, w - crop_w + 1), (1,)).item())
    top = int(torch.randint(0, max(1, h - crop_h + 1), (1,)).item())

    return image.crop((left, top, left + crop_w, top + crop_h))


def draw_roi_bboxes(image, roi_records, canonical_path):
    """Draw all ROI bboxes on image for this sample."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    roi_record = roi_records.get(canonical_path)
    if not roi_record:
        return img_copy

    # Get bbox and convert from normalized to pixel coordinates
    bbox_norm = roi_record.get("bbox")
    if bbox_norm and len(bbox_norm) == 4:
        w, h = image.size
        x0_norm, y0_norm, x1_norm, y1_norm = bbox_norm
        x0 = int(x0_norm * w)
        y0 = int(y0_norm * h)
        x1 = int(x1_norm * w)
        y1 = int(y1_norm * h)

        # Draw yellow rectangle
        draw.rectangle(
            [(x0, y0), (x1, y1)],
            outline=ROI_BOX_COLOR,
            width=3
        )

        # Draw label
        draw.text(
            (x0 + 5, y0 + 5),
            "Grad-CAM",
            fill=ROI_BOX_COLOR
        )

    return img_copy


def resize_to_tile(image, tile_size=336):
    """Resize image to tile size."""
    return image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    if isinstance(tensor, torch.Tensor):
        # Assume tensor is (C, H, W) in [0, 1] range after normalization
        # Need to denormalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        np_arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_arr)
    return tensor


def create_sample_row(img_path, label, roi_records, transform1, transform2, augmentation_intensity):
    """Create a 1x3 row for one sample: Original+ROI | View1 | View2."""
    img = Image.open(img_path).convert("RGB")
    canonical_path = canonicalize_image_path(img_path)
    roi_record = roi_records.get(canonical_path)

    # Column 1: Original + ROI bbox
    img_with_roi = draw_roi_bboxes(img, roi_records, canonical_path)
    col1 = resize_to_tile(img_with_roi)

    # Column 2: View 1 (random crop 0.9-1.0 + augmentation)
    view1_crop = get_random_crop(img, 0.9, 1.0)
    view1 = transform1(view1_crop)
    col2 = tensor_to_pil(view1)

    # Column 3: View 2 (ROI crop or random fallback, augmented)
    if roi_record:
        # ROI crop at 0.4-0.8
        roi_scale = sample_crop_scale(0.4, 0.8)
        jitter_xy = sample_jitter(0.05)
        try:
            view2_crop = crop_image_to_roi(
                image=img,
                roi_record=roi_record,
                context_scale=2.0,
                min_crop_scale=roi_scale,
                jitter_xy=jitter_xy,
                max_aspect_ratio=DEFAULT_ROI_MAX_ASPECT_RATIO,
            )
        except:
            view2_crop = get_random_crop(img, 0.9, 1.0)
    else:
        # Random crop 0.9-1.0 (fallback when no ROI)
        view2_crop = get_random_crop(img, 0.9, 1.0)

    view2 = transform2(view2_crop)
    col3 = tensor_to_pil(view2)

    # Combine columns horizontally
    row = Image.new("RGB", (TILE_SIZE * 3, TILE_SIZE), (0, 0, 0))
    row.paste(col1, (0, 0))
    row.paste(col2, (TILE_SIZE, 0))
    row.paste(col3, (TILE_SIZE * 2, 0))

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        choices=[0, 1],
        help="0=negative (ndbe), 1=positive (neo)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples (rows in grid)"
    )
    parser.add_argument(
        "--roi-json",
        type=str,
        default=DEFAULT_ROI_JSON,
        help="Path to rois.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--augmentation-intensity",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Augmentation intensity level (1=low, 2=medium, 3=strong)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load dataset
    df, _ = build_dataset_dataframe(DEFAULT_DATA_DIR)
    df_filtered = df[df["label"] == args.label].reset_index(drop=True)

    if len(df_filtered) == 0:
        print(f"No samples found with label {args.label}")
        return

    # Load ROI records
    roi_records = {}
    if Path(args.roi_json).exists():
        all_roi_records, _ = load_roi_records_from_json(args.roi_json)

        # Filter by path: neo = positive, ndbe = negative
        path_filter = ("\\neo\\" if args.label == 1 else "\\ndbe\\", "/neo/" if args.label == 1 else "/ndbe/")
        for image_path, record in all_roi_records.items():
            if path_filter[0] in image_path or path_filter[1] in image_path:
                roi_records[image_path] = record

    # Build augmentation transforms
    transform1 = build_roi_focus_transform(336, augmentation_intensity=args.augmentation_intensity)
    transform2 = build_roi_focus_transform(336, augmentation_intensity=args.augmentation_intensity)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample images
    num_samples = min(args.num_samples, len(df_filtered))
    indices = random.sample(range(len(df_filtered)), num_samples)

    label_name = "POSITIVE" if args.label == 1 else "NEGATIVE"
    print(f"Creating SupPro view visualization for {num_samples} {label_name} samples...")
    print(f"Augmentation intensity: {args.augmentation_intensity}")
    print(f"Grid layout: {num_samples} rows x 3 columns")
    print(f"  Column 1: Original + Grad-CAM ROI bbox")
    print(f"  Column 2: View 1 (random crop 0.9-1.0 + augmentation)")
    print(f"  Column 3: View 2 (ROI crop 0.4-0.8 + augmentation, or random fallback)")

    # Create grid of all samples
    total_height = TILE_SIZE * num_samples
    total_width = TILE_SIZE * 3

    grid = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    for i, idx in enumerate(indices):
        row_data = df_filtered.iloc[idx]
        img_path = row_data["img"]

        try:
            sample_row = create_sample_row(
                img_path, args.label, roi_records, transform1, transform2, args.augmentation_intensity
            )
            grid.paste(sample_row, (0, i * TILE_SIZE))
            print(f"  [{i+1}/{num_samples}] {Path(img_path).name}")
        except Exception as e:
            print(f"  [{i+1}/{num_samples}] Error: {e}")
            import traceback
            traceback.print_exc()

    # Save grid
    output_path = output_dir / f"{label_name.lower()}_suppro_{num_samples}rows_aug{args.augmentation_intensity}.png"
    grid.save(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Grid size: {grid.size}")


if __name__ == "__main__":
    main()
