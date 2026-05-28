"""visualize_suppro_batch.py

Shows what a SupPro training batch actually looks like:
  - One figure per sampled batch
  - Rows = samples (interleaved neo / ndbe to reflect BalancedBatchSampler)
  - Columns: original image | view 1 (full-frame light aug) | view 2 (ROI crop or full-frame)
  - ROI bbox overlaid on the original for neo samples that have a record (yellow solid box)
  - Actual crop window overlaid on the original (cyan dashed box) — shows context expansion
  - Similarity matrix of all (view1, view2) embeddings in the batch — shows
    which pairs the SupPro loss wants to pull together vs push apart
  - Colour-coded row/column labels: neo = red, ndbe = blue

Usage examples:
    # 8 samples, ROI-guided (requires rois.json and the gastronet checkpoint)
    python visualize_suppro_batch.py

    # Larger batch, no ROI guidance (control run)
    python visualize_suppro_batch.py --batch-size 16 --no-roi

    # Use a specific seed for reproducibility
    python visualize_suppro_batch.py --seed 7

    # Save instead of displaying
    python visualize_suppro_batch.py --save-path batch_vis.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; will switch to interactive if display available
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "../data/Challenge_train_data"
DEFAULT_ROI_JSON = REPO_ROOT / "checkpoints/roi_records/rois.json"
DEFAULT_ENCODER_CKPT = None   # set via --encoder-ckpt if you want the similarity matrix

NEO_COLOR   = "#e05555"
NDBE_COLOR  = "#5577cc"
ROI_BOX_COLOR = "#ffdd00"
CROP_WINDOW_COLOR = "#00ddff"  # Cyan

# ── roi helper imports ─────────────────────────────────────────────────────
from roi_guidance import compute_crop_window_from_roi

# ── denormalise helper ─────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_uint8(t):
    """(C,H,W) normalised tensor → (H,W,3) uint8 numpy array."""
    img = t.permute(1, 2, 0).cpu().numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def draw_roi_box(ax, roi_record, img_w, img_h, color=ROI_BOX_COLOR, lw=2):
    """Overlay a normalised bbox on a matplotlib axes that shows an image."""
    bbox = roi_record.get("bbox")
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    rect = mpatches.Rectangle(
        (x0 * img_w, y0 * img_h),
        (x1 - x0) * img_w,
        (y1 - y0) * img_h,
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)


def draw_crop_window(ax, roi_record, img_w, img_h, context_scale=2.0, min_crop_scale=0.4,
                     max_aspect_ratio=1.5, color=CROP_WINDOW_COLOR, lw=2, debug=False):
    """Overlay the actual crop window (after context expansion) on matplotlib axes."""
    if roi_record is None:
        return
    try:
        left, top, right, bottom = compute_crop_window_from_roi(
            roi_record=roi_record,
            context_scale=context_scale,
            min_crop_scale=min_crop_scale,
            jitter_xy=(0.0, 0.0),  # No jitter for visualization
            max_aspect_ratio=max_aspect_ratio,
        )
        crop_width = right - left
        crop_height = bottom - top

        if debug:
            roi_width = roi_record.get("roi_width", 0)
            roi_height = roi_record.get("roi_height", 0)
            print(f"  ROI size: {roi_width:.3f}×{roi_height:.3f} | "
                  f"Crop size: {crop_width:.3f}×{crop_height:.3f} | "
                  f"min_crop_scale={min_crop_scale}")

        rect = mpatches.Rectangle(
            (left * img_w, top * img_h),
            crop_width * img_w,
            crop_height * img_h,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            linestyle="--",  # Dashed line to distinguish from ROI box
        )
        ax.add_patch(rect)
    except Exception as e:
        if debug:
            print(f"  Error computing crop window: {e}")
        pass  # Silently skip if crop window computation fails


def build_args():
    parser = argparse.ArgumentParser(description="Visualise a SupPro training batch.")
    parser.add_argument("--batch-size",    type=int,   default=12,
                        help="Number of samples to show (will be 50/50 neo/ndbe).")
    parser.add_argument("--roi-json",      type=str,   default=str(DEFAULT_ROI_JSON),
                        help="Path to rois.json. Pass empty string to disable ROI guidance.")
    parser.add_argument("--no-roi",        action="store_true",
                        help="Disable ROI guidance regardless of --roi-json.")
    parser.add_argument("--roi-focus-prob",type=float, default=1.0,
                        help="Probability of using an ROI crop for view 2 (neo samples only).")
    parser.add_argument("--roi-context-scale", type=float, default=2.0)
    parser.add_argument("--roi-min-crop-scale", type=float, default=.2)
    parser.add_argument("--roi-center-jitter",  type=float, default=0.05)
    parser.add_argument("--roi-max-aspect-ratio", type=float, default=1.5)
    parser.add_argument("--input-size",    type=int,   default=336)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--encoder-ckpt",  type=str,   default=None,
                        help="Optional path to an encoder .pt checkpoint. "
                             "When provided, renders a cosine-similarity matrix for the batch.")
    parser.add_argument("--save-path",     type=str,   default=None,
                        help="Save figure to this path instead of displaying it.")
    parser.add_argument("--num-batches",   type=int,   default=1,
                        help="How many independent batches to visualise (one figure each).")
    parser.add_argument("--debug",         action="store_true",
                        help="Print crop window computation details (ROI size, crop size, min_crop_scale).")
    return parser.parse_args()


def load_datasets(args):
    sys.path.insert(0, str(REPO_ROOT))
    from data import (
        TwoViewDataset,
        BalancedBatchSampler,
        build_dataset_dataframe,
        build_roi_focus_transform,
        build_seeded_generator,
    )
    from roi_guidance import load_roi_records_from_json, canonicalize_image_path
    from sklearn.model_selection import train_test_split

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df, class_names = build_dataset_dataframe(str(DATA_DIR))
    df["stratify_col"] = df["center"].astype(str) + "_" + df["label"].astype(str)
    train_df, _ = train_test_split(
        df, test_size=0.2, stratify=df["stratify_col"], random_state=args.seed
    )
    train_df = train_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)

    t1 = build_roi_focus_transform(args.input_size)
    t2 = build_roi_focus_transform(args.input_size)
    t_roi = build_roi_focus_transform(args.input_size)

    ds = TwoViewDataset(
        train_df, t1, t2,
        roi_transform2=t_roi,
        roi_target_label=1,
        roi_focus_prob=args.roi_focus_prob,
        roi_context_scale=args.roi_context_scale,
        roi_min_crop_scale=args.roi_min_crop_scale,
        roi_center_jitter=args.roi_center_jitter,
        roi_max_aspect_ratio=args.roi_max_aspect_ratio,
    )

    roi_records = {}
    roi_json = args.roi_json if not args.no_roi else ""
    if roi_json and Path(roi_json).exists():
        raw_records, meta = load_roi_records_from_json(roi_json)

        # rois.json keys are RELATIVE paths (../data/center_1/neo/UUID.png).
        # ImageFolder (used inside build_dataset_dataframe) produces ABSOLUTE paths.
        # String comparison will always fail. Match on filename stem (UUID) instead —
        # these are unique across the entire dataset.
        stem_to_dataset_path = {
            Path(p).stem: canonicalize_image_path(p)
            for p in train_df["img"].astype(str).tolist()
        }
        roi_records = {}
        for roi_key, record in raw_records.items():
            stem = Path(roi_key).stem
            dataset_path = stem_to_dataset_path.get(stem)
            if dataset_path is not None:
                roi_records[dataset_path] = record

        ds.set_roi_records(roi_records, active=True)
        print(f"Loaded {len(roi_records)} ROI records (out of {len(raw_records)} total) from {roi_json}")
        if len(roi_records) == 0:
            print("  WARNING: 0 records matched — rois.json stems not found in train_df.")
    else:
        print("ROI guidance disabled — both views are full-frame light aug.")

    sampler = BalancedBatchSampler(
        labels=train_df["label"].tolist(),
        batch_size=args.batch_size,
        generator=build_seeded_generator(args.seed),
    )
    return ds, train_df, roi_records, sampler


def try_load_model(encoder_ckpt, input_size):
    """Return model on CPU (or None if no ckpt supplied or import fails)."""
    if not encoder_ckpt:
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from model import Model
        ckpt = torch.load(encoder_ckpt, map_location="cpu")
        model_config = ckpt.get("model_config", {})
        model = Model(
            in_channels=3,
            n_classes=model_config.get("n_classes", 2),
            backbone_name=model_config.get("backbone_name", "vit_base_patch14_reg4_dinov2"),
            backbone_weights_path=None,
            input_size=input_size,
            freeze_backbone=False,
            pretrained=False,
            proj_dim=128,
            head_type=model_config.get("head_type", "mlp_fullwidth"),
        )
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"Loaded encoder from {encoder_ckpt} for similarity matrix.")
        return model
    except Exception as exc:
        print(f"Warning: could not load encoder ({exc}). Skipping similarity matrix.")
        return None


def compute_similarity_matrix(model, views1, views2):
    """Returns (2N, 2N) cosine similarity matrix for the stacked view pairs."""
    with torch.no_grad():
        e1 = model(views1, return_embedding=True)["embedding"]
        e2 = model(views2, return_embedding=True)["embedding"]
    embs = torch.cat([e1, e2], dim=0)               # (2N, D)
    embs = F.normalize(embs, dim=-1)
    return (embs @ embs.T).cpu().numpy()             # (2N, 2N)


def visualize_batch(batch_indices, ds, train_df, roi_records, args, model, batch_num=0):
    from roi_guidance import canonicalize_image_path

    n = len(batch_indices)
    has_model = model is not None

    # ── layout: n rows × 3 cols + optional similarity matrix ──────────────────
    n_img_cols = 3   # original | view1 | view2
    fig_width = n_img_cols * 2.6 + (4.0 if has_model else 0)
    fig_height = n * 2.4 + 0.8
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#1a1a1a")

    if has_model:
        gs = fig.add_gridspec(
            n, n_img_cols + 1,
            width_ratios=[1, 1, 1, 1.5],
            hspace=0.06, wspace=0.06,
            left=0.01, right=0.99, top=0.94, bottom=0.03,
        )
    else:
        gs = fig.add_gridspec(
            n, n_img_cols,
            hspace=0.06, wspace=0.06,
            left=0.01, right=0.99, top=0.94, bottom=0.03,
        )

    # Column titles via fig.text so they're not overwritten by later add_subplot calls
    col_x_positions = [1/6, 3/6, 5/6]  # approximate centres of the 3 image columns
    col_title_strs = [
        "Original + ROI box",
        "View 1 — full frame, light aug",
        "View 2 — ROI crop (neo) / full frame (ndbe)",
    ]
    title_y = 0.965
    for cx, ct in zip(col_x_positions, col_title_strs):
        fig.text(cx, title_y, ct, ha="center", va="bottom",
                 fontsize=7.5, color="#bbbbbb",
                 transform=fig.transFigure)

    collected_v1 = []
    collected_v2 = []
    collected_labels = []

    for row_idx, sample_idx in enumerate(batch_indices):
        img_path = train_df.loc[sample_idx, "img"]
        label    = int(train_df.loc[sample_idx, "label"])
        class_name = "neo" if label == 1 else "ndbe"
        row_color  = NEO_COLOR if label == 1 else NDBE_COLOR
        canon_path = canonicalize_image_path(img_path)
        roi_record = roi_records.get(canon_path)

        # get the two augmented views from the dataset
        v1, v2, _ = ds[sample_idx]
        collected_v1.append(v1)
        collected_v2.append(v2)
        collected_labels.append(label)

        orig_img = Image.open(img_path).convert("RGB")
        orig_arr = np.array(orig_img)
        v1_arr   = tensor_to_uint8(v1)
        v2_arr   = tensor_to_uint8(v2)

        def make_ax(col):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(row_color)
                spine.set_linewidth(2.0)
            return ax

        def label_badge(ax, text, color):
            """Prominent class badge in the top-left corner of an image axes."""
            ax.text(
                0.03, 0.97, text,
                transform=ax.transAxes,
                fontsize=8, fontweight="bold",
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85, linewidth=0),
            )

        # ── col 0: original + ROI box + crop window ───────────────────────────
        ax0 = make_ax(0)
        ax0.imshow(orig_arr)
        h, w = orig_arr.shape[:2]
        if roi_record is not None:
            draw_roi_box(ax0, roi_record, w, h)
            draw_crop_window(
                ax0, roi_record, w, h,
                context_scale=args.roi_context_scale,
                min_crop_scale=args.roi_min_crop_scale,
                max_aspect_ratio=args.roi_max_aspect_ratio,
                debug=args.debug,
            )
        label_badge(ax0, class_name, row_color)

        # ── col 1: view 1 ─────────────────────────────────────────────────────
        ax1 = make_ax(1)
        ax1.imshow(v1_arr)
        label_badge(ax1, class_name, row_color)
        ax1.set_xlabel("full frame · light aug", fontsize=6, color="#888888", labelpad=2)

        # ── col 2: view 2 ─────────────────────────────────────────────────────
        ax2 = make_ax(2)
        ax2.imshow(v2_arr)
        label_badge(ax2, class_name, row_color)
        if roi_record is not None and label == 1 and ds.roi_guidance_active:
            v2_label = f"ROI crop · ctx={args.roi_context_scale}×"
            ax2.set_xlabel(v2_label, fontsize=6, color=ROI_BOX_COLOR, labelpad=2)
        else:
            ax2.set_xlabel("full frame · light aug", fontsize=6, color="#888888", labelpad=2)

    # ── similarity matrix ──────────────────────────────────────────────────────
    if has_model:
        v1_batch = torch.stack(collected_v1)
        v2_batch = torch.stack(collected_v2)
        sim = compute_similarity_matrix(model, v1_batch, v2_batch)

        ax_sim = fig.add_subplot(gs[:, n_img_cols])
        im = ax_sim.imshow(sim, vmin=-0.2, vmax=1.0, cmap="RdBu_r", aspect="auto")

        # tick labels: [v1_0, v1_1, …, v2_0, v2_1, …]
        tick_labels = (
            [f"v1·{'neo' if l==1 else 'ndbe'}" for l in collected_labels] +
            [f"v2·{'neo' if l==1 else 'ndbe'}" for l in collected_labels]
        )
        tick_colors = (
            [NEO_COLOR if l==1 else NDBE_COLOR for l in collected_labels] * 2
        )
        ax_sim.set_xticks(range(2 * n))
        ax_sim.set_xticklabels(tick_labels, rotation=90, fontsize=5.5)
        ax_sim.set_yticks(range(2 * n))
        ax_sim.set_yticklabels(tick_labels, fontsize=5.5)
        for xtick, color in zip(ax_sim.get_xticklabels(), tick_colors):
            xtick.set_color(color)
        for ytick, color in zip(ax_sim.get_yticklabels(), tick_colors):
            ytick.set_color(color)

        # dividing line between view1 block and view2 block
        ax_sim.axhline(n - 0.5, color="#888888", lw=0.8, linestyle="--")
        ax_sim.axvline(n - 0.5, color="#888888", lw=0.8, linestyle="--")
        ax_sim.set_title("Cosine sim\n(view1 | view2)", fontsize=7, color="#bbbbbb", pad=3)
        ax_sim.tick_params(colors="#888888")
        for spine in ax_sim.spines.values():
            spine.set_edgecolor("#444444")
        plt.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#888888", labelcolor="#888888")

    # ── figure title ──────────────────────────────────────────────────────────
    roi_status = f"ROI-guided (prob={args.roi_focus_prob})" if ds.roi_guidance_active else "No ROI guidance"
    fig.suptitle(
        f"SupPro batch #{batch_num + 1}  |  {n} samples (50% neo / 50% ndbe)  |  {roi_status}  |  seed={args.seed}",
        fontsize=9, color="#cccccc", y=0.98,
    )
    return fig


def main():
    args = build_args()

    ds, train_df, roi_records, sampler = load_datasets(args)
    model = try_load_model(args.encoder_ckpt, args.input_size)

    # draw args.num_batches independent batches
    batch_iter = iter(sampler)
    for batch_num in range(args.num_batches):
        try:
            batch_indices = next(batch_iter)
        except StopIteration:
            print(f"Sampler exhausted after {batch_num} batches.")
            break

        print(f"\nBatch {batch_num + 1}: {len(batch_indices)} samples")
        labels_in_batch = [train_df.loc[i, "label"] for i in batch_indices]
        neo_count  = sum(1 for l in labels_in_batch if l == 1)
        ndbe_count = sum(1 for l in labels_in_batch if l == 0)
        from roi_guidance import canonicalize_image_path as _canon
        roi_count  = sum(
            1 for i in batch_indices
            if train_df.loc[i, "label"] == 1
            and _canon(train_df.loc[i, "img"]) in roi_records
        )
        print(f"  neo={neo_count}  ndbe={ndbe_count}  neo_with_roi={roi_count}")

        fig = visualize_batch(batch_indices, ds, train_df, roi_records, args, model, batch_num)

        if args.save_path:
            save_path = args.save_path
            if args.num_batches > 1:
                stem = Path(save_path).stem
                suffix = Path(save_path).suffix or ".png"
                save_path = str(Path(save_path).parent / f"{stem}_batch{batch_num + 1}{suffix}")
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"  Saved to {save_path}")
            plt.close(fig)
        else:
            # try interactive display; fall back to saving beside the script
            try:
                matplotlib.use("TkAgg")
                plt.show()
            except Exception:
                fallback = str(REPO_ROOT / f"suppro_batch_vis_{batch_num + 1}.png")
                fig.savefig(fallback, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
                print(f"  No display available — saved to {fallback}")
                plt.close(fig)


if __name__ == "__main__":
    main()
