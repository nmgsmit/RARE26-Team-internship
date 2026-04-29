from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import average_precision_score

from metrics import compute_batch_binary_dice_scores, compute_binary_dice_score


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
FLAT_CAM_STD_EPS = 1e-6
NEAR_ZERO_CAM_MAX_EPS = 1e-6


def _forward_tokens_and_logits(model, images):
    if hasattr(model, "forward_tokens") and hasattr(model, "forward_from_tokens"):
        tokens = model.forward_tokens(images)
        logits = model.forward_from_tokens(tokens)
        return tokens, logits

    tokens = model.backbone.forward_features(images)
    pooled = model.backbone.forward_head(tokens, pre_logits=True)
    logits = model.head(pooled)
    return tokens, logits


def _get_num_prefix_tokens(model):
    if hasattr(model.backbone, "num_prefix_tokens"):
        return int(model.backbone.num_prefix_tokens)
    return 1


def _get_patch_grid_size(model):
    grid_size = getattr(model.backbone.patch_embed, "grid_size", None)
    if grid_size is None:
        raise ValueError("Backbone patch grid size is unavailable; cannot reshape ViT Grad-CAM tokens.")
    return tuple(grid_size)


def _normalize_heatmaps(cams, eps=1e-8):
    flat = cams.flatten(start_dim=1)
    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    normalized = (flat - mins) / (maxs - mins).clamp_min(eps)
    return normalized.view_as(cams)


def compute_vit_gradcam_batch(model, images, target_class=1, return_raw=False):
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}.")

    images = images.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    tokens, logits = _forward_tokens_and_logits(model, images)
    num_classes = logits.shape[1]
    if target_class < 0 or target_class >= num_classes:
        raise ValueError(f"Target class {target_class} is out of range for {num_classes} classes.")

    target_scores = logits[:, target_class].sum()
    token_grads = torch.autograd.grad(target_scores, tokens, retain_graph=False, create_graph=False)[0]

    num_prefix_tokens = _get_num_prefix_tokens(model)
    grid_h, grid_w = _get_patch_grid_size(model)
    patch_tokens = tokens[:, num_prefix_tokens:, :]
    patch_grads = token_grads[:, num_prefix_tokens:, :]

    expected_patch_count = grid_h * grid_w
    if patch_tokens.shape[1] != expected_patch_count:
        raise ValueError(
            "Unexpected ViT token count for Grad-CAM reshaping: "
            f"got {patch_tokens.shape[1]} patch tokens, expected {expected_patch_count}."
        )

    channel_weights = patch_grads.mean(dim=1, keepdim=True)
    cams = torch.relu(torch.sum(patch_tokens * channel_weights, dim=-1))
    cams = cams.view(images.shape[0], 1, grid_h, grid_w)
    cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    raw_cams = cams
    cams = _normalize_heatmaps(raw_cams)

    target_probs = torch.softmax(logits, dim=1)[:, target_class]
    if return_raw:
        return cams.detach(), target_probs.detach(), raw_cams.detach()
    return cams.detach(), target_probs.detach()


def _denormalize_image(image_tensor):
    mean = IMAGENET_MEAN.to(device=image_tensor.device, dtype=image_tensor.dtype)
    std = IMAGENET_STD.to(device=image_tensor.device, dtype=image_tensor.dtype)
    return torch.clamp(image_tensor * std + mean, 0.0, 1.0)


def _image_tensor_to_rgb_uint8(image_tensor):
    image = _denormalize_image(image_tensor).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _heatmap_to_rgb(cam_tensor):
    heatmap = np.clip(np.rint(cam_tensor.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap, mode="L")
    colored = ImageOps.colorize(heatmap_img, black="#000000", mid="#ff7f0e", white="#ffff66")
    return np.asarray(colored, dtype=np.uint8)


def _mask_to_rgb(mask_tensor):
    mask = (mask_tensor.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    return np.repeat(mask[..., None], 3, axis=2)


def _overlay_heatmap_on_image(image_rgb, cam_tensor, max_alpha=0.8):
    heatmap_rgb = _heatmap_to_rgb(cam_tensor).astype(np.float32)
    alpha = np.clip(cam_tensor.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)[..., None] * max_alpha
    overlay = ((1.0 - alpha) * image_rgb.astype(np.float32)) + (alpha * heatmap_rgb)
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def _compute_mask_edge(mask):
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask for edge extraction, got shape {mask.shape}.")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    eroded = np.zeros_like(mask, dtype=bool)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~eroded


def _overlay_soft_consensus(base_rgb, soft_mask_tensor, outline_threshold=0.5, max_alpha=0.55):
    soft_mask = np.clip(soft_mask_tensor.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    alpha = (soft_mask[..., None] * max_alpha).astype(np.float32)
    fill_color = np.asarray([255.0, 64.0, 64.0], dtype=np.float32)

    overlay = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * fill_color.reshape(1, 1, 3)
    overlay = np.clip(np.rint(overlay), 0, 255).astype(np.uint8)

    outline_mask = _compute_mask_edge(soft_mask >= outline_threshold)
    if outline_mask.any():
        overlay[outline_mask] = np.asarray([255, 255, 255], dtype=np.uint8)

    return overlay


def _build_labeled_panel(columns, labels):
    if len(columns) != len(labels):
        raise ValueError("Each visualization column must have a matching label.")

    font = ImageFont.load_default()
    header_height = 18
    column_width = columns[0].shape[1]
    column_height = columns[0].shape[0]
    canvas = Image.new("RGB", (column_width * len(columns), column_height + header_height), color="white")
    draw = ImageDraw.Draw(canvas)

    for idx, (column, label) in enumerate(zip(columns, labels)):
        x_offset = idx * column_width
        canvas.paste(Image.fromarray(column), (x_offset, header_height))
        draw.text((x_offset + 4, 3), label, fill="black", font=font)

    return np.asarray(canvas, dtype=np.uint8)


def _build_wandb_example(image_tensor, cam_tensor, pred_mask, gt_mask, image_path, target_class, class_prob, dice_score):
    image_rgb = _image_tensor_to_rgb_uint8(image_tensor)
    overlay_rgb = _overlay_heatmap_on_image(image_rgb, cam_tensor)
    pred_mask_rgb = _mask_to_rgb(pred_mask)
    gt_mask_rgb = _mask_to_rgb(gt_mask)
    panel = _build_labeled_panel(
        [image_rgb, overlay_rgb, pred_mask_rgb, gt_mask_rgb],
        ["image", "gradcam", "cam mask", "gt mask"],
    )

    dice_text = "skipped (empty mask)" if not np.isfinite(dice_score) else f"{dice_score:.3f}"
    caption = (
        f"{Path(image_path).name} | p(class {target_class})={class_prob:.3f} | "
        f"dice={dice_text}"
    )
    return wandb.Image(panel, caption=caption)


def parse_thresholds(thresholds):
    if isinstance(thresholds, str):
        parts = [part.strip() for part in thresholds.split(",")]
        thresholds = [float(part) for part in parts if part]
    thresholds = [float(threshold) for threshold in thresholds]
    if not thresholds:
        raise ValueError("At least one Grad-CAM threshold is required.")

    deduped = []
    for threshold in sorted(thresholds):
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Grad-CAM threshold must be between 0 and 1, got {threshold}.")
        if deduped and np.isclose(threshold, deduped[-1]):
            continue
        deduped.append(threshold)
    return deduped


def build_expert_consensus_masks(expert_masks):
    expert_masks = expert_masks.to(torch.float32)
    expert_masks_bool = expert_masks > 0
    expert_count = expert_masks_bool.shape[0]
    majority_count = (expert_count // 2) + 1
    union_mask = expert_masks_bool.any(dim=0)
    majority_mask = expert_masks_bool.sum(dim=0) >= majority_count
    soft_consensus = expert_masks.mean(dim=0)
    return union_mask.to(torch.float32), majority_mask.to(torch.float32), soft_consensus


def compute_binary_iou_score(pred_mask, target_mask, eps=1e-8):
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.detach().cpu().numpy()

    pred_mask = np.asarray(pred_mask).astype(bool)
    target_mask = np.asarray(target_mask).astype(bool)
    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            f"IoU masks must share the same shape, got {pred_mask.shape} and {target_mask.shape}."
        )

    intersection = float(np.logical_and(pred_mask, target_mask).sum())
    union = float(np.logical_or(pred_mask, target_mask).sum())
    if union == 0.0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def compute_pixel_average_precision(score_map, target_mask):
    if torch.is_tensor(score_map):
        score_map = score_map.detach().cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.detach().cpu().numpy()

    score_map = np.asarray(score_map, dtype=np.float64)
    target_mask = np.asarray(target_mask).astype(bool)
    if score_map.shape != target_mask.shape:
        raise ValueError(
            "Grad-CAM score map and target mask must share the same shape, "
            f"got {score_map.shape} and {target_mask.shape}."
        )
    if not np.any(target_mask):
        return float("nan")
    return float(average_precision_score(target_mask.reshape(-1).astype(np.uint8), score_map.reshape(-1)))


def compute_soft_mask_mass(score_map, soft_mask, eps=1e-8):
    if torch.is_tensor(score_map):
        score_map = score_map.detach().cpu().numpy()
    if torch.is_tensor(soft_mask):
        soft_mask = soft_mask.detach().cpu().numpy()

    score_map = np.asarray(score_map, dtype=np.float64)
    soft_mask = np.asarray(soft_mask, dtype=np.float64)
    if score_map.shape != soft_mask.shape:
        raise ValueError(
            "Grad-CAM score map and soft consensus mask must share the same shape, "
            f"got {score_map.shape} and {soft_mask.shape}."
        )

    total_mass = float(np.maximum(score_map, 0.0).sum())
    if total_mass <= eps:
        return 0.0
    return float(np.sum(np.maximum(score_map, 0.0) * soft_mask) / total_mass)


def compute_peak_hit(score_map, target_mask):
    if torch.is_tensor(score_map):
        score_map = score_map.detach().cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.detach().cpu().numpy()

    score_map = np.asarray(score_map, dtype=np.float64)
    target_mask = np.asarray(target_mask).astype(bool)
    if score_map.shape != target_mask.shape:
        raise ValueError(
            "Grad-CAM score map and target mask must share the same shape, "
            f"got {score_map.shape} and {target_mask.shape}."
        )
    peak_index = int(np.argmax(score_map))
    return float(target_mask.reshape(-1)[peak_index])


def compute_pairwise_mask_ious(expert_masks):
    if torch.is_tensor(expert_masks):
        expert_masks = expert_masks.detach().cpu().numpy()
    expert_masks = np.asarray(expert_masks).astype(bool)
    ious = []
    for idx in range(expert_masks.shape[0]):
        for jdx in range(idx + 1, expert_masks.shape[0]):
            ious.append(compute_binary_iou_score(expert_masks[idx], expert_masks[jdx]))
    return ious


def _finite_mean(values):
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _finite_median(values):
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.median(finite_values))


def _normalized_curve_auc(values, thresholds):
    finite_pairs = [
        (float(threshold), float(value))
        for threshold, value in zip(thresholds, values)
        if np.isfinite(value)
    ]
    if not finite_pairs:
        return float("nan")
    if len(finite_pairs) == 1:
        return finite_pairs[0][1]

    xs = np.asarray([pair[0] for pair in finite_pairs], dtype=np.float64)
    ys = np.asarray([pair[1] for pair in finite_pairs], dtype=np.float64)
    span = float(xs[-1] - xs[0])
    if span <= 0.0:
        return float(np.mean(ys))
    return float(np.trapz(ys, xs) / span)


def _compute_cam_activation_stats(raw_cam_tensor):
    raw_cam = raw_cam_tensor.detach().cpu().numpy().astype(np.float64)
    raw_max_activation = float(np.max(raw_cam))
    raw_std_activation = float(np.std(raw_cam))
    is_flat_or_near_zero = (
        raw_std_activation <= FLAT_CAM_STD_EPS
        or raw_max_activation <= NEAR_ZERO_CAM_MAX_EPS
    )
    return {
        "raw_max_activation": raw_max_activation,
        "raw_std_activation": raw_std_activation,
        "is_flat_or_near_zero": float(is_flat_or_near_zero),
    }


def _format_threshold(threshold):
    return f"{threshold:.2f}"


def _compute_threshold_overlap_scores(cam_tensor, expert_masks, thresholds):
    threshold_metrics = []
    for threshold in thresholds:
        pred_mask = (cam_tensor >= threshold).to(torch.float32)
        dice_scores = [
            compute_binary_dice_score(pred_mask, expert_mask)
            for expert_mask in expert_masks
        ]
        iou_scores = [
            compute_binary_iou_score(pred_mask, expert_mask)
            for expert_mask in expert_masks
        ]
        threshold_metrics.append({
            "threshold": float(threshold),
            "mean_dice": _finite_mean(dice_scores),
            "mean_iou": _finite_mean(iou_scores),
            "area_fraction": float(pred_mask.detach().cpu().numpy().mean()),
        })
    return threshold_metrics


def _build_barrett_wandb_example(
    image_tensor,
    cam_tensor,
    expert_masks,
    image_path,
    target_class,
    class_prob,
    summary_text,
    display_threshold=0.5,
):
    _, _, soft_consensus = build_expert_consensus_masks(expert_masks)

    image_rgb = _image_tensor_to_rgb_uint8(image_tensor)
    overlay_rgb = _overlay_heatmap_on_image(image_rgb, cam_tensor)
    comparison_rgb = _overlay_soft_consensus(
        overlay_rgb,
        soft_consensus,
        outline_threshold=display_threshold,
    )

    columns = [image_rgb, overlay_rgb, comparison_rgb]
    labels = ["image", "gradcam", "gradcam + consensus"]

    panel = _build_labeled_panel(columns, labels)
    caption = (
        f"{Path(image_path).name} | p(class {target_class})={class_prob:.3f} | "
        f"{summary_text}"
    )
    return wandb.Image(panel, caption=caption)


def evaluate_gradcam_barrett_dataset(
    model,
    loader,
    device,
    thresholds,
    target_class=1,
    display_threshold=0.5,
    log_best_k=8,
    log_worst_k=8,
    log_hard_neg_k=8,
    prefix="gradcam",
    dataset_qa=None,
):
    thresholds = parse_thresholds(thresholds)
    if display_threshold < 0.0 or display_threshold > 1.0:
        raise ValueError(f"display_threshold must be between 0 and 1, got {display_threshold}.")

    was_training = model.training
    model.eval()

    positive_entries = []
    negative_entries = []
    pairwise_iou_values = []
    positive_threshold_dice = {threshold: [] for threshold in thresholds}
    positive_threshold_iou = {threshold: [] for threshold in thresholds}
    negative_empty_masks = 0
    negative_nonempty_masks = 0
    positive_majority_empty_count = 0

    try:
        with torch.enable_grad():
            for images, labels, expert_masks_batch, image_paths in loader:
                images = images.to(device)
                labels = labels.to(device)
                expert_masks_batch = expert_masks_batch.to(device)

                cams, probs, raw_cams = compute_vit_gradcam_batch(
                    model,
                    images,
                    target_class=target_class,
                    return_raw=True,
                )

                for sample_idx in range(images.shape[0]):
                    label = int(labels[sample_idx].item())
                    expert_masks = expert_masks_batch[sample_idx]
                    image_path = image_paths[sample_idx]
                    cam_tensor = cams[sample_idx]
                    raw_cam_tensor = raw_cams[sample_idx]
                    cam_activation_stats = _compute_cam_activation_stats(raw_cam_tensor)
                    target_prob = float(probs[sample_idx].item())
                    union_mask, majority_mask, soft_consensus = build_expert_consensus_masks(expert_masks)

                    if label == 1:
                        pairwise_iou_values.extend(compute_pairwise_mask_ious(expert_masks))
                        if not torch.any(majority_mask > 0):
                            positive_majority_empty_count += 1

                        ap_consensus = compute_pixel_average_precision(cam_tensor, majority_mask)
                        expert_aps = [
                            compute_pixel_average_precision(cam_tensor, expert_mask)
                            for expert_mask in expert_masks
                        ]
                        threshold_metrics = _compute_threshold_overlap_scores(
                            cam_tensor,
                            expert_masks,
                            thresholds,
                        )
                        for row in threshold_metrics:
                            positive_threshold_dice[row["threshold"]].append(row["mean_dice"])
                            positive_threshold_iou[row["threshold"]].append(row["mean_iou"])

                        positive_entries.append({
                            "image_path": image_path,
                            "image_tensor": images[sample_idx].detach().cpu(),
                            "cam_tensor": cam_tensor.detach().cpu(),
                            "expert_masks": expert_masks.detach().cpu(),
                            "target_prob": target_prob,
                            "ap_consensus": ap_consensus,
                            "mean_expert_ap": _finite_mean(expert_aps),
                            "expert_ap_std": float(np.std(expert_aps)),
                            "soft_consensus_mass": compute_soft_mask_mass(cam_tensor, soft_consensus),
                            "peak_hit_union": compute_peak_hit(cam_tensor, union_mask),
                            "peak_hit_majority": compute_peak_hit(cam_tensor, majority_mask),
                            "raw_max_activation": cam_activation_stats["raw_max_activation"],
                            "raw_std_activation": cam_activation_stats["raw_std_activation"],
                            "is_flat_or_near_zero": cam_activation_stats["is_flat_or_near_zero"],
                        })
                    else:
                        nonempty_expert_mask_count = int(torch.sum(torch.any(expert_masks > 0, dim=(1, 2))).item())
                        negative_nonempty_masks += nonempty_expert_mask_count
                        negative_empty_masks += expert_masks.shape[0] - nonempty_expert_mask_count

                        negative_entries.append({
                            "image_path": image_path,
                            "image_tensor": images[sample_idx].detach().cpu(),
                            "cam_tensor": cam_tensor.detach().cpu(),
                            "expert_masks": expert_masks.detach().cpu(),
                            "target_prob": target_prob,
                            "raw_max_activation": cam_activation_stats["raw_max_activation"],
                            "raw_std_activation": cam_activation_stats["raw_std_activation"],
                            "is_flat_or_near_zero": cam_activation_stats["is_flat_or_near_zero"],
                            "soft_consensus_mass": compute_soft_mask_mass(cam_tensor, soft_consensus),
                            "peak_hit_majority": compute_peak_hit(cam_tensor, majority_mask),
                        })
    finally:
        if was_training:
            model.train()

    combined_dataset_qa = dict(dataset_qa or {})
    combined_dataset_qa.update({
        "negative_empty_mask_count": negative_empty_masks,
        "negative_nonempty_mask_count": negative_nonempty_masks,
        "positive_majority_empty_count": positive_majority_empty_count,
    })
    all_entries = positive_entries + negative_entries

    summary_payload = {
        f"{prefix}/overall/mean_target_class_probability": _finite_mean(
            entry["target_prob"] for entry in all_entries
        ),
        f"{prefix}/overall/mean_soft_consensus_mass": _finite_mean(
            entry["soft_consensus_mass"] for entry in all_entries
        ),
        f"{prefix}/overall/mean_peak_hit_majority": _finite_mean(
            entry["peak_hit_majority"] for entry in all_entries
        ),
        f"{prefix}/overall/fraction_flat_or_near_zero_cams": _finite_mean(
            entry["is_flat_or_near_zero"] for entry in all_entries
        ),
        f"{prefix}/positive/mAP_consensus": _finite_mean(entry["ap_consensus"] for entry in positive_entries),
        f"{prefix}/positive/mAP_expert_mean": _finite_mean(entry["mean_expert_ap"] for entry in positive_entries),
        f"{prefix}/positive/mAP_expert_std_mean": _finite_mean(entry["expert_ap_std"] for entry in positive_entries),
        f"{prefix}/positive/soft_consensus_mass": _finite_mean(
        entry["soft_consensus_mass"] for entry in positive_entries
        ),
        f"{prefix}/positive/peak_hit_union": _finite_mean(entry["peak_hit_union"] for entry in positive_entries),
        f"{prefix}/positive/peak_hit_majority": _finite_mean(
        entry["peak_hit_majority"] for entry in positive_entries
        ),
        f"{prefix}/negative/mean_positive_class_probability": _finite_mean(
        entry["target_prob"] for entry in negative_entries
        ),
        f"{prefix}/negative/mean_raw_max_activation": _finite_mean(
        entry["raw_max_activation"] for entry in negative_entries
        ),
        f"{prefix}/dataset/positive_pairwise_iou_mean": _finite_mean(pairwise_iou_values),
        f"{prefix}/dataset/positive_pairwise_iou_median": _finite_median(pairwise_iou_values),
        f"{prefix}/dataset/negative_empty_mask_count": negative_empty_masks,
        f"{prefix}/dataset/negative_nonempty_mask_count": negative_nonempty_masks,
        f"{prefix}/dataset/positive_majority_empty_count": positive_majority_empty_count,
    }

    positive_dice_curve = []
    positive_iou_curve = []
    for threshold in thresholds:
        mean_dice = _finite_mean(positive_threshold_dice[threshold])
        mean_iou = _finite_mean(positive_threshold_iou[threshold])
        positive_dice_curve.append(mean_dice)
        positive_iou_curve.append(mean_iou)

    summary_payload[f"{prefix}/positive/dice_auc"] = _normalized_curve_auc(positive_dice_curve, thresholds)
    summary_payload[f"{prefix}/positive/iou_auc"] = _normalized_curve_auc(positive_iou_curve, thresholds)

    best_positive_entries = sorted(
        [entry for entry in positive_entries if np.isfinite(entry["ap_consensus"])],
        key=lambda entry: entry["ap_consensus"],
        reverse=True,
    )[:log_best_k]
    worst_positive_entries = sorted(
        [entry for entry in positive_entries if np.isfinite(entry["ap_consensus"])],
        key=lambda entry: entry["ap_consensus"],
    )[:log_worst_k]
    hard_negative_entries = sorted(
        negative_entries,
        key=lambda entry: (
            entry["target_prob"],
            entry["raw_max_activation"],
        ),
        reverse=True,
    )[:log_hard_neg_k]

    media_payload = {}
    if best_positive_entries:
        media_payload[f"{prefix}/positive/examples_best"] = [
            _build_barrett_wandb_example(
                entry["image_tensor"],
                entry["cam_tensor"],
                entry["expert_masks"],
                entry["image_path"],
                target_class,
                entry["target_prob"],
                (
                    f"AP={entry['ap_consensus']:.3f} | expertAP={entry['mean_expert_ap']:.3f} | "
                    f"mass={entry['soft_consensus_mass']:.3f}"
                ),
                display_threshold=display_threshold,
            )
            for entry in best_positive_entries
        ]
    if worst_positive_entries:
        media_payload[f"{prefix}/positive/examples_worst"] = [
            _build_barrett_wandb_example(
                entry["image_tensor"],
                entry["cam_tensor"],
                entry["expert_masks"],
                entry["image_path"],
                target_class,
                entry["target_prob"],
                (
                    f"AP={entry['ap_consensus']:.3f} | expertAP={entry['mean_expert_ap']:.3f} | "
                    f"mass={entry['soft_consensus_mass']:.3f}"
                ),
                display_threshold=display_threshold,
            )
            for entry in worst_positive_entries
        ]
    if hard_negative_entries:
        media_payload[f"{prefix}/negative/examples_hard"] = [
            _build_barrett_wandb_example(
                entry["image_tensor"],
                entry["cam_tensor"],
                entry["expert_masks"],
                entry["image_path"],
                target_class,
                entry["target_prob"],
                (
                    f"neg prob={entry['target_prob']:.3f} | raw max={entry['raw_max_activation']:.4f}"
                ),
                display_threshold=display_threshold,
            )
            for entry in hard_negative_entries
        ]

    ranking_metadata = {
        "best_positive": [
            {"image_path": entry["image_path"], "ap_consensus": entry["ap_consensus"]}
            for entry in best_positive_entries
        ],
        "worst_positive": [
            {"image_path": entry["image_path"], "ap_consensus": entry["ap_consensus"]}
            for entry in worst_positive_entries
        ],
        "hard_negative": [
            {
                "image_path": entry["image_path"],
                "target_prob": entry["target_prob"],
                "raw_max_activation": entry["raw_max_activation"],
            }
            for entry in hard_negative_entries
        ],
    }
    scalar_payload = {
        f"{prefix}/overall/mean_target_class_probability": summary_payload[
            f"{prefix}/overall/mean_target_class_probability"
        ],
        f"{prefix}/overall/mean_soft_consensus_mass": summary_payload[
            f"{prefix}/overall/mean_soft_consensus_mass"
        ],
        f"{prefix}/overall/mean_peak_hit_majority": summary_payload[
            f"{prefix}/overall/mean_peak_hit_majority"
        ],
        f"{prefix}/overall/fraction_flat_or_near_zero_cams": summary_payload[
            f"{prefix}/overall/fraction_flat_or_near_zero_cams"
        ],
    }
    return {
        "media_payload": media_payload,
        "scalar_payload": scalar_payload,
        "summary_payload": summary_payload,
        "dataset_qa": combined_dataset_qa,
        "ranking_metadata": ranking_metadata,
    }


def evaluate_gradcam_segmentation_dataset(
    model,
    loader,
    device,
    target_class=1,
    threshold=0.5,
    max_log_samples=8,
    skip_empty_masks=True,
    split_name="segmentation",
):
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Grad-CAM threshold must be between 0 and 1, got {threshold}.")
    if max_log_samples < 0:
        raise ValueError(f"max_log_samples must be non-negative, got {max_log_samples}.")

    was_training = model.training
    model.eval()

    all_dice_scores = []
    target_probs = []
    total_samples = 0
    skipped_empty_masks = 0
    logged_examples = []

    try:
        with torch.enable_grad():
            for images, masks, image_paths, _ in loader:
                images = images.to(device)
                masks = masks.to(device)

                cams, probs = compute_vit_gradcam_batch(model, images, target_class=target_class)
                pred_masks = (cams >= threshold).to(dtype=masks.dtype)

                batch_dice_scores, batch_skipped = compute_batch_binary_dice_scores(
                    pred_masks,
                    masks,
                    ignore_empty_targets=skip_empty_masks,
                )

                total_samples += images.shape[0]
                skipped_empty_masks += batch_skipped
                target_probs.extend(probs.detach().cpu().tolist())
                all_dice_scores.extend(score for score in batch_dice_scores if np.isfinite(score))

                remaining_slots = max_log_samples - len(logged_examples)
                if remaining_slots <= 0:
                    continue

                for sample_idx in range(min(images.shape[0], remaining_slots)):
                    logged_examples.append(
                        _build_wandb_example(
                            images[sample_idx],
                            cams[sample_idx],
                            pred_masks[sample_idx],
                            masks[sample_idx],
                            image_paths[sample_idx],
                            target_class,
                            probs[sample_idx].item(),
                            batch_dice_scores[sample_idx],
                        )
                    )
    finally:
        if was_training:
            model.train()

    payload = {
        f"{split_name}/mean_dice": float(np.mean(all_dice_scores)) if all_dice_scores else float("nan"),
        f"{split_name}/dice_scored_samples": len(all_dice_scores),
        f"{split_name}/dice_total_samples": total_samples,
        f"{split_name}/dice_skipped_empty_masks": skipped_empty_masks,
        f"{split_name}/gradcam_target_class": target_class,
        f"{split_name}/gradcam_threshold": threshold,
        f"{split_name}/mean_target_class_probability": (
            float(np.mean(target_probs)) if target_probs else float("nan")
        ),
    }
    if logged_examples:
        payload[f"{split_name}/gradcam_examples"] = logged_examples
    return payload
