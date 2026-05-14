import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_ROI_MAX_ASPECT_RATIO = 1.5
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


def _clamp_bbox(bbox):
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = min(max(x0, 0.0), 1.0)
    y0 = min(max(y0, 0.0), 1.0)
    x1 = min(max(x1, x0 + 1e-6), 1.0)
    y1 = min(max(y1, y0 + 1e-6), 1.0)
    return (x0, y0, x1, y1)


def compute_roi_geometry_from_bbox(bbox):
    x0, y0, x1, y1 = _clamp_bbox(bbox)
    roi_width = float(x1 - x0)
    roi_height = float(y1 - y0)
    center_x = float(0.5 * (x0 + x1))
    center_y = float(0.5 * (y0 + y1))
    aspect_ratio = float(roi_width / max(roi_height, 1e-6))
    return {
        "bbox": (x0, y0, x1, y1),
        "source_bbox": (x0, y0, x1, y1),
        "center_x": center_x,
        "center_y": center_y,
        "roi_width": roi_width,
        "roi_height": roi_height,
        "roi_aspect_ratio": aspect_ratio,
    }


def compute_bbox_from_roi_geometry(center_x, center_y, roi_width, roi_height):
    center_x = float(center_x)
    center_y = float(center_y)
    roi_width = max(float(roi_width), 1e-6)
    roi_height = max(float(roi_height), 1e-6)
    return _clamp_bbox(
        (
            center_x - 0.5 * roi_width,
            center_y - 0.5 * roi_height,
            center_x + 0.5 * roi_width,
            center_y + 0.5 * roi_height,
        )
    )


def regularize_roi_dimensions(roi_width, roi_height, max_aspect_ratio=DEFAULT_ROI_MAX_ASPECT_RATIO):
    roi_width = max(float(roi_width), 1e-6)
    roi_height = max(float(roi_height), 1e-6)
    max_aspect_ratio = max(float(max_aspect_ratio), 1.0)

    aspect_ratio = roi_width / roi_height
    if aspect_ratio > max_aspect_ratio:
        roi_height = roi_width / max_aspect_ratio
    elif aspect_ratio < (1.0 / max_aspect_ratio):
        roi_width = roi_height / max_aspect_ratio

    return float(roi_width), float(roi_height)


def compute_crop_window_from_roi(
    bbox=None,
    roi_record=None,
    context_scale=2.0,
    min_crop_scale=0.4,
    jitter_xy=(0.0, 0.0),
    max_aspect_ratio=DEFAULT_ROI_MAX_ASPECT_RATIO,
):
    if roi_record is not None:
        normalized_record = _normalize_roi_record(roi_record)
        center_x = normalized_record["center_x"]
        center_y = normalized_record["center_y"]
        roi_width = normalized_record["roi_width"]
        roi_height = normalized_record["roi_height"]
    elif bbox is not None:
        geometry = compute_roi_geometry_from_bbox(bbox)
        center_x = geometry["center_x"]
        center_y = geometry["center_y"]
        roi_width = geometry["roi_width"]
        roi_height = geometry["roi_height"]
    else:
        raise ValueError("compute_crop_window_from_roi requires either bbox or roi_record.")

    roi_width, roi_height = regularize_roi_dimensions(
        roi_width,
        roi_height,
        max_aspect_ratio=max_aspect_ratio,
    )
    crop_width = min(1.0, max(float(min_crop_scale), roi_width * float(context_scale)))
    crop_height = min(1.0, max(float(min_crop_scale), roi_height * float(context_scale)))
    jitter_x = float(jitter_xy[0]) * 0.5 * crop_width
    jitter_y = float(jitter_xy[1]) * 0.5 * crop_height
    center_x += jitter_x
    center_y += jitter_y

    left = min(max(center_x - 0.5 * crop_width, 0.0), 1.0 - crop_width)
    top = min(max(center_y - 0.5 * crop_height, 0.0), 1.0 - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return (left, top, right, bottom)


def build_roi_record_from_binary_mask(mask_like, source, score=1.0):
    mask = _as_binary_array(mask_like)
    bbox = compute_normalized_bbox_from_binary_mask(mask)
    if bbox is None:
        return None

    roi_geometry = compute_roi_geometry_from_bbox(bbox)
    return {
        **roi_geometry,
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

    source_bbox = normalized_record.get("source_bbox")
    if source_bbox is not None:
        if len(source_bbox) != 4:
            raise ValueError(f"ROI record has an invalid source_bbox: {record}")
        normalized_record["source_bbox"] = tuple(float(value) for value in source_bbox)

    bbox = normalized_record.get("bbox")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"ROI record has an invalid bbox: {record}")
        normalized_record["bbox"] = tuple(float(value) for value in bbox)

    has_center_geometry = all(
        key in normalized_record for key in ("center_x", "center_y", "roi_width", "roi_height")
    )
    if has_center_geometry:
        normalized_record["center_x"] = float(normalized_record["center_x"])
        normalized_record["center_y"] = float(normalized_record["center_y"])
        normalized_record["roi_width"] = float(normalized_record["roi_width"])
        normalized_record["roi_height"] = float(normalized_record["roi_height"])
    elif "source_bbox" in normalized_record:
        normalized_record.update(compute_roi_geometry_from_bbox(normalized_record["source_bbox"]))
    elif "bbox" in normalized_record:
        normalized_record.update(compute_roi_geometry_from_bbox(normalized_record["bbox"]))
    else:
        raise ValueError(f"ROI record is missing ROI geometry: {record}")

    if "bbox" not in normalized_record:
        normalized_record["bbox"] = compute_bbox_from_roi_geometry(
            normalized_record["center_x"],
            normalized_record["center_y"],
            normalized_record["roi_width"],
            normalized_record["roi_height"],
        )
    if "source_bbox" not in normalized_record:
        normalized_record["source_bbox"] = tuple(normalized_record["bbox"])

    normalized_record["bbox"] = _clamp_bbox(normalized_record["bbox"])
    normalized_record["source_bbox"] = _clamp_bbox(normalized_record["source_bbox"])
    normalized_record["roi_aspect_ratio"] = float(
        normalized_record.get(
            "roi_aspect_ratio",
            normalized_record["roi_width"] / max(normalized_record["roi_height"], 1e-6),
        )
    )

    if "coverage" in normalized_record:
        normalized_record["coverage"] = float(normalized_record["coverage"])
    if "score" in normalized_record:
        normalized_record["score"] = float(normalized_record["score"])
    if "source" in normalized_record:
        normalized_record["source"] = str(normalized_record["source"])
    if "pixel_area" in normalized_record:
        normalized_record["pixel_area"] = int(normalized_record["pixel_area"])
    if "island_index" in normalized_record:
        normalized_record["island_index"] = int(normalized_record["island_index"])
    if "island_count" in normalized_record:
        normalized_record["island_count"] = int(normalized_record["island_count"])
    if "peak_activation" in normalized_record:
        normalized_record["peak_activation"] = float(normalized_record["peak_activation"])
    if "mean_activation" in normalized_record:
        normalized_record["mean_activation"] = float(normalized_record["mean_activation"])
    if "roi_threshold" in normalized_record:
        normalized_record["roi_threshold"] = float(normalized_record["roi_threshold"])
    if "roi_islands" in normalized_record:
        normalized_record["roi_islands"] = [
            _normalize_roi_record(island_record)
            for island_record in normalized_record["roi_islands"]
        ]
    return normalized_record


def _serialize_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _serialize_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_roi_records_to_json(path, roi_records, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized_records = {}
    for image_path, record in roi_records.items():
        normalized_record = _normalize_roi_record(record)
        serialized_records[str(image_path)] = _serialize_json_safe(normalized_record)

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
    bbox=None,
    roi_record=None,
    context_scale=2.0,
    min_crop_scale=0.4,
    jitter_xy=(0.0, 0.0),
    max_aspect_ratio=DEFAULT_ROI_MAX_ASPECT_RATIO,
):
    width, height = image.size
    left, top, right, bottom = compute_crop_window_from_roi(
        bbox=bbox,
        roi_record=roi_record,
        context_scale=context_scale,
        min_crop_scale=min_crop_scale,
        jitter_xy=jitter_xy,
        max_aspect_ratio=max_aspect_ratio,
    )

    left_px = max(0, min(width - 1, int(np.floor(left * width))))
    top_px = max(0, min(height - 1, int(np.floor(top * height))))
    right_px = max(left_px + 1, min(width, int(np.ceil(right * width))))
    bottom_px = max(top_px + 1, min(height, int(np.ceil(bottom * height))))
    return image.crop((left_px, top_px, right_px, bottom_px))
