from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, InterpolationMode
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
BARRETT_EXPERT_MASK_PATTERN = re.compile(r"^(?P<base>.+)_exp(?P<expert>\d+)$", re.IGNORECASE)


class ExternalTestsetDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = [infer_testset_label_from_filename(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]

def infer_testset_label_from_filename(image_path):
    stem = image_path.stem.upper()
    if stem.endswith("_ACHD"):
        return 1
    if stem.endswith("_NDBT"):
        return 0
    raise ValueError(
        f"Could not infer class from filename '{image_path.name}'. "
        "Expected suffix _ACHD or _NDBT before extension."
    )


def _list_image_files(root_dir, recursive=False):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return paths


def _candidate_match_keys(path):
    stem = path.stem.lower()
    keys = [stem]
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            keys.append(stem[:-len(suffix)])

    deduped_keys = []
    for key in keys:
        if key and key not in deduped_keys:
            deduped_keys.append(key)
    return deduped_keys


def _build_mask_index(mask_paths):
    mask_index = {}
    duplicate_keys = {}
    for mask_path in mask_paths:
        for key in _candidate_match_keys(mask_path):
            if key in mask_index and mask_index[key] != mask_path:
                duplicate_keys.setdefault(key, set()).update({mask_index[key], mask_path})
                continue
            mask_index.setdefault(key, mask_path)

    if duplicate_keys:
        duplicate_message = ", ".join(
            f"{key}: {[str(path) for path in sorted(paths)]}" for key, paths in duplicate_keys.items()
        )
        raise ValueError(f"Found duplicate segmentation masks for the same image key. {duplicate_message}")

    return mask_index


def _barrett_mask_sort_key(mask_path):
    match = BARRETT_EXPERT_MASK_PATTERN.match(mask_path.stem)
    if match is None:
        return (float("inf"), mask_path.name.lower())
    return (int(match.group("expert")), mask_path.name.lower())


def strip_barrett_expert_suffix(path_or_stem):
    stem = path_or_stem.stem if isinstance(path_or_stem, Path) else str(path_or_stem)
    match = BARRETT_EXPERT_MASK_PATTERN.match(stem)
    if match is None:
        return stem.lower()
    return match.group("base").lower()


def _extract_barrett_expert_indices(mask_paths):
    expert_indices = []
    for mask_path in mask_paths:
        match = BARRETT_EXPERT_MASK_PATTERN.match(mask_path.stem)
        if match is None:
            raise ValueError(
                "Expected Barrett annotation mask names to end with _expN, "
                f"got '{mask_path.name}'."
            )
        expert_indices.append(int(match.group("expert")))
    return expert_indices


def build_barrett_gradcam_samples(dataset_root):
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations_bmp"

    image_paths = _list_image_files(images_dir, recursive=False)
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in Barrett images directory: {images_dir}")

    annotation_paths = _list_image_files(annotations_dir, recursive=False)
    if len(annotation_paths) == 0:
        raise ValueError(f"No annotation bitmap files found in Barrett annotations directory: {annotations_dir}")

    image_index = {image_path.stem.lower(): image_path for image_path in image_paths}
    annotation_groups = {}
    unmatched_annotation_paths = []

    for annotation_path in annotation_paths:
        image_key = strip_barrett_expert_suffix(annotation_path)
        if image_key not in image_index:
            unmatched_annotation_paths.append(annotation_path)
            continue
        annotation_groups.setdefault(image_key, []).append(annotation_path)

    if unmatched_annotation_paths:
        unmatched_message = ", ".join(str(path.name) for path in unmatched_annotation_paths[:10])
        raise ValueError(
            "Found Barrett annotation bitmaps that do not match any image. "
            f"Examples: {unmatched_message}"
        )

    samples = []
    unmatched_images = []
    positive_count = 0
    negative_count = 0

    for image_path in image_paths:
        image_key = image_path.stem.lower()
        group = sorted(annotation_groups.get(image_key, []), key=_barrett_mask_sort_key)
        if not group:
            unmatched_images.append(image_path)
            continue

        expert_indices = _extract_barrett_expert_indices(group)
        if len(group) != 5 or sorted(expert_indices) != [1, 2, 3, 4, 5]:
            raise ValueError(
                "Expected exactly five Barrett expert masks named _exp1.._exp5 for "
                f"'{image_path.name}', got {[path.name for path in group]}."
            )

        label = infer_testset_label_from_filename(image_path)
        positive_count += int(label == 1)
        negative_count += int(label == 0)
        samples.append({
            "image_path": image_path,
            "label": label,
            "mask_paths": group,
        })

    if unmatched_images:
        unmatched_message = ", ".join(str(path.name) for path in unmatched_images[:10])
        raise ValueError(
            "Found Barrett images without five expert masks. "
            f"Examples: {unmatched_message}"
        )

    qa_stats = {
        "image_count": len(samples),
        "positive_image_count": positive_count,
        "negative_image_count": negative_count,
        "annotation_count": len(annotation_paths),
        "annotation_groups": len(annotation_groups),
        "annotations_per_image_min": min(len(sample["mask_paths"]) for sample in samples),
        "annotations_per_image_max": max(len(sample["mask_paths"]) for sample in samples),
        "unmatched_annotation_count": len(unmatched_annotation_paths),
        "unmatched_image_count": len(unmatched_images),
    }
    return samples, qa_stats


class SegmentationEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_transform, mask_transform):
        self.samples = samples
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor[0]
        mask_tensor = (mask_tensor > 0).to(torch.float32)
        return image_tensor, mask_tensor, str(image_path), str(mask_path)


class BarrettGradcamDataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_transform, mask_transform):
        self.samples = samples
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.image_transform(image)

        expert_masks = []
        for mask_path in sample["mask_paths"]:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor[0]
            expert_masks.append((mask_tensor > 0).to(torch.float32))

        expert_masks = torch.stack(expert_masks, dim=0)
        return image_tensor, sample["label"], expert_masks, str(sample["image_path"])


def load_external_testset(testset_images_dir, batch_size, num_workers, device, input_size=224):
    transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset_image_paths = _list_image_files(testset_images_dir, recursive=False)
    if len(testset_image_paths) == 0:
        raise ValueError(f"No image files found in testset directory: {testset_images_dir}")
    testset_ds = ExternalTestsetDataset(testset_image_paths, transform)
    testset_loader = DataLoader(
        testset_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return testset_loader, testset_ds, testset_image_paths


def load_barrett_gradcam_dataset(dataset_root, batch_size, num_workers, device, input_size=224):
    image_transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = Compose([
        ToImage(),
        Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.float32, scale=False),
    ])

    samples, qa_stats = build_barrett_gradcam_samples(dataset_root)
    dataset = BarrettGradcamDataset(samples, image_transform, mask_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, dataset, samples, qa_stats


def load_segmentation_testset(
    segmentation_images_dir,
    segmentation_masks_dir,
    batch_size,
    num_workers,
    device,
    input_size=224,
):
    image_transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = Compose([
        ToImage(),
        Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.float32, scale=False),
    ])

    image_paths = _list_image_files(segmentation_images_dir, recursive=True)
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in segmentation image directory: {segmentation_images_dir}")

    mask_paths = _list_image_files(segmentation_masks_dir, recursive=True)
    if len(mask_paths) == 0:
        raise ValueError(f"No mask files found in segmentation mask directory: {segmentation_masks_dir}")

    mask_index = _build_mask_index(mask_paths)
    matched_samples = []
    unmatched_images = []

    for image_path in image_paths:
        mask_path = None
        for key in _candidate_match_keys(image_path):
            if key in mask_index:
                mask_path = mask_index[key]
                break

        if mask_path is None:
            unmatched_images.append(image_path)
            continue

        matched_samples.append((image_path, mask_path))

    if len(matched_samples) == 0:
        raise ValueError(
            "Could not match any segmentation masks to the provided images. "
            f"Images dir: {segmentation_images_dir}, masks dir: {segmentation_masks_dir}"
        )

    segmentation_ds = SegmentationEvaluationDataset(matched_samples, image_transform, mask_transform)
    segmentation_loader = DataLoader(
        segmentation_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return segmentation_loader, segmentation_ds, matched_samples, unmatched_images
