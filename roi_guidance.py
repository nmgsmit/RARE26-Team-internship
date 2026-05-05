import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = (
    "_mask",
    "_seg",
    "_segmentation",
    "_annotation",
    "_label",
    "-mask",
    "-seg",
    "-segmentation",
    "-annotation",
    "-label",
)


def list_image_files(root_dir, recursive=True):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def candidate_match_keys(path_or_name):
    stem = Path(path_or_name).stem.lower()
    keys = [stem]
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            trimmed = stem[:-len(suffix)]
            if trimmed:
                keys.append(trimmed)

    deduped = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def build_mask_index(mask_paths):
    mask_index = {}
    duplicate_keys = {}
    for mask_path in mask_paths:
        for key in candidate_match_keys(mask_path):
            if key in mask_index and mask_index[key] != mask_path:
                duplicate_keys.setdefault(key, set()).update({mask_index[key], mask_path})
                continue
            mask_index.setdefault(key, mask_path)

    if duplicate_keys:
        duplicate_message = ", ".join(
            f"{key}: {[str(path) for path in sorted(paths)]}" for key, paths in duplicate_keys.items()
        )
        raise ValueError(f"Found duplicate ROI masks for the same image key. {duplicate_message}")

    return mask_index


def _as_binary_array(mask_like):
    mask_array = np.asarray(mask_like)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return mask_array > 0


def compute_normalized_bbox_from_binary_mask(mask_like):
    mask = _as_binary_array(mask_like)
    if not mask.any():
        return None

    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    x0 = float(xs.min()) / float(width)
    y0 = float(ys.min()) / float(height)
    x1 = float(xs.max() + 1) / float(width)
    y1 = float(ys.max() + 1) / float(height)
    return (x0, y0, x1, y1)


def build_roi_record_from_binary_mask(mask_like, source, score=1.0):
    mask = _as_binary_array(mask_like)
    bbox = compute_normalized_bbox_from_binary_mask(mask)
    if bbox is None:
        return None

    return {
        "bbox": bbox,
        "coverage": float(mask.mean()),
        "score": float(score),
        "source": str(source),
    }


def build_roi_record_from_cam(cam_like, threshold, score):
    cam = np.asarray(cam_like, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D CAM array, got shape {cam.shape}.")

    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    active_mask = cam >= float(threshold)
    return build_roi_record_from_binary_mask(active_mask, source="gradcam", score=score)


def load_roi_records_from_masks(image_paths, masks_dir):
    mask_paths = list_image_files(masks_dir, recursive=True)
    if len(mask_paths) == 0:
        raise ValueError(f"No ROI mask files found in {masks_dir}")

    mask_index = build_mask_index(mask_paths)
    roi_records = {}
    unmatched_images = []
    matched_images = []

    for image_path in image_paths:
        image_path = Path(image_path)
        matched_mask_path = None
        for key in candidate_match_keys(image_path):
            if key in mask_index:
                matched_mask_path = mask_index[key]
                break

        if matched_mask_path is None:
            unmatched_images.append(str(image_path))
            continue

        mask = Image.open(matched_mask_path).convert("L")
        roi_record = build_roi_record_from_binary_mask(mask, source="mask", score=1.0)
        if roi_record is None:
            continue

        roi_record["mask_path"] = str(matched_mask_path)
        roi_records[str(image_path)] = roi_record
        matched_images.append(str(image_path))

    return roi_records, matched_images, unmatched_images


def _normalize_roi_record(record):
    normalized_record = dict(record)
    bbox = normalized_record.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"ROI record is missing a valid bbox: {record}")
    normalized_record["bbox"] = tuple(float(value) for value in bbox)
    if "coverage" in normalized_record:
        normalized_record["coverage"] = float(normalized_record["coverage"])
    if "score" in normalized_record:
        normalized_record["score"] = float(normalized_record["score"])
    if "source" in normalized_record:
        normalized_record["source"] = str(normalized_record["source"])
    return normalized_record


def save_roi_records_to_json(path, roi_records, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized_records = {}
    for image_path, record in roi_records.items():
        normalized_record = _normalize_roi_record(record)
        serialized_record = dict(normalized_record)
        serialized_record["bbox"] = list(normalized_record["bbox"])
        serialized_records[str(image_path)] = serialized_record

    payload = {
        "metadata": dict(metadata or {}),
        "roi_records": serialized_records,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_roi_records_from_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "roi_records" in payload:
        raw_records = payload.get("roi_records", {})
        metadata = payload.get("metadata", {})
    else:
        raw_records = payload
        metadata = {}

    roi_records = {
        str(image_path): _normalize_roi_record(record)
        for image_path, record in dict(raw_records).items()
    }
    return roi_records, dict(metadata)


def crop_image_to_roi(
    image,
    bbox,
    context_scale=2.0,
    min_crop_scale=0.4,
    jitter_xy=(0.0, 0.0),
):
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in bbox]

    x0 = min(max(x0, 0.0), 1.0)
    y0 = min(max(y0, 0.0), 1.0)
    x1 = min(max(x1, x0 + 1e-6), 1.0)
    y1 = min(max(y1, y0 + 1e-6), 1.0)

    roi_width = x1 - x0
    roi_height = y1 - y0
    crop_width = min(1.0, max(float(min_crop_scale), roi_width * float(context_scale)))
    crop_height = min(1.0, max(float(min_crop_scale), roi_height * float(context_scale)))

    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    jitter_x = float(jitter_xy[0]) * 0.5 * crop_width
    jitter_y = float(jitter_xy[1]) * 0.5 * crop_height
    center_x += jitter_x
    center_y += jitter_y

    left = min(max(center_x - 0.5 * crop_width, 0.0), 1.0 - crop_width)
    top = min(max(center_y - 0.5 * crop_height, 0.0), 1.0 - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    left_px = max(0, min(width - 1, int(np.floor(left * width))))
    top_px = max(0, min(height - 1, int(np.floor(top * height))))
    right_px = max(left_px + 1, min(width, int(np.ceil(right * width))))
    bottom_px = max(top_px + 1, min(height, int(np.ceil(bottom * height))))
    return image.crop((left_px, top_px, right_px, bottom_px))
