import math
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
)

from ROI_helpers.roi_guidance import canonicalize_image_path, crop_image_to_roi

DEFAULT_DATA_DIR = "../data/Challenge_train_data"

# Augmentation intensity presets
AUGMENTATION_PRESETS = {
    1: {  # Low intensity (conservative)
        "name": "low",
        "h_flip_p": 0.1,
        "v_flip_p": 0.0,  # Avoid vertical flip for endoscopy (fold directionality)
        "rotation_deg": 2,
        "color_brightness": 0.05,
        "color_contrast": 0.05,
        "color_saturation": 0.05,
        "color_hue": 0.01,
        "roi_rotation_deg": 2,
        "roi_color_brightness": 0.02,
        "roi_color_contrast": 0.02,
        "roi_color_saturation": 0.02,
        "roi_color_hue": 0.003,
    },
    2: {  # Medium intensity (balanced)
        "name": "medium",
        "h_flip_p": 0.3,
        "v_flip_p": 0.1,
        "rotation_deg": 5,
        "color_brightness": 0.08,
        "color_contrast": 0.08,
        "color_saturation": 0.08,
        "color_hue": 0.015,
        "roi_rotation_deg": 3,
        "roi_color_brightness": 0.03,
        "roi_color_contrast": 0.03,
        "roi_color_saturation": 0.03,
        "roi_color_hue": 0.005,
    },
    3: {  # Strong intensity (aggressive - current default)
        "name": "strong",
        "h_flip_p": 0.5,
        "v_flip_p": 0.2,
        "rotation_deg": 10,
        "color_brightness": 0.1,
        "color_contrast": 0.1,
        "color_saturation": 0.1,
        "color_hue": 0.02,
        "roi_rotation_deg": 5,
        "roi_color_brightness": 0.03,
        "roi_color_contrast": 0.03,
        "roi_color_saturation": 0.03,
        "roi_color_hue": 0.005,
    },
    4: {  # Extreme intensity (very aggressive - for hard augmentation studies)
        "name": "extreme",
        "h_flip_p": 0.5,
        "v_flip_p": 0.5,  # High vertical flip probability
        "rotation_deg": 45,  # Large rotation range
        "color_brightness": 0.2,
        "color_contrast": 0.2,
        "color_saturation": 0.2,
        "color_hue": 0.03,
        "roi_rotation_deg": 15,
        "roi_color_brightness": 0.05,
        "roi_color_contrast": 0.05,
        "roi_color_saturation": 0.05,
        "roi_color_hue": 0.007,
    },
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_seeded_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class TwoViewDataset(Dataset):
    def __init__(
        self,
        df,
        transform1,
        transform2,
        roi_transform2=None,
        roi_target_label=1,
        roi_focus_prob=1.0,
        roi_negative_focus_prob=0.0,
        roi_context_scale=2.0,
        roi_min_crop_scale=0.4,
        roi_max_crop_scale=1.0,
        roi_center_jitter=0.05,
        roi_max_aspect_ratio=1.5,
        roi_warmup_epochs=0,
    ):
        self.df = df.reset_index(drop=True)
        self.transform1 = transform1
        self.transform2 = transform2
        self.roi_transform2 = roi_transform2 or transform2
        self.roi_target_label = int(roi_target_label)
        self.roi_focus_prob = min(max(float(roi_focus_prob), 0.0), 1.0)
        self.roi_negative_focus_prob = min(max(float(roi_negative_focus_prob), 0.0), 1.0)
        self.roi_context_scale = max(float(roi_context_scale), 1e-6)
        self.roi_min_crop_scale = min(max(float(roi_min_crop_scale), 1e-6), 1.0)
        self.roi_max_crop_scale = min(max(float(roi_max_crop_scale), self.roi_min_crop_scale), 1.0)
        self.roi_center_jitter = max(float(roi_center_jitter), 0.0)
        self.roi_max_aspect_ratio = max(float(roi_max_aspect_ratio), 1.0)
        self.roi_warmup_epochs = max(int(roi_warmup_epochs), 0)
        self.roi_records = {}
        self.roi_guidance_active = False
        self.negative_roi_records = {}
        self.negative_roi_guidance_active = False
        self.current_epoch = 0

    def __len__(self):
        return len(self.df)

    def _select_roi_record(self, roi_record):
        """Randomly select one island if multiple exist, otherwise use primary bbox."""
        roi_islands = roi_record.get("roi_islands")
        if not roi_islands:
            return roi_record

        island_index = int(torch.randint(len(roi_islands), (1,)).item())
        return roi_islands[island_index]

    def set_roi_records(self, roi_records, active=True):
        self.roi_records = {
            canonicalize_image_path(key): value for key, value in roi_records.items()
        }
        self.roi_guidance_active = bool(active) and len(self.roi_records) > 0

    def clear_roi_records(self):
        self.roi_records = {}
        self.roi_guidance_active = False

    def set_negative_roi_records(self, roi_records, active=True):
        """Set ROI records for negative (hard-negative) regions."""
        self.negative_roi_records = {
            canonicalize_image_path(key): value for key, value in roi_records.items()
        }
        self.negative_roi_guidance_active = bool(active) and len(self.negative_roi_records) > 0

    def clear_negative_roi_records(self):
        """Clear negative ROI records."""
        self.negative_roi_records = {}
        self.negative_roi_guidance_active = False

    def set_epoch(self, epoch):
        """Set the current epoch (used for warmup control)."""
        self.current_epoch = int(epoch)

    def get_roi_guidance_stats(self):
        positive_image_paths = self.df.loc[
            self.df["label"] == self.roi_target_label, "img"
        ].astype(str).map(canonicalize_image_path).tolist()
        covered_records = [
            self.roi_records[path] for path in positive_image_paths if path in self.roi_records
        ]
        source_counts = {}
        for record in covered_records:
            source = str(record.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        mean_coverage = 0.0
        if covered_records:
            mean_coverage = float(
                sum(float(record.get("coverage", 0.0)) for record in covered_records)
                / len(covered_records)
            )

        return {
            "roi_guidance_active": bool(self.roi_guidance_active),
            "roi_positive_images": len(covered_records),
            "roi_positive_candidates": len(positive_image_paths),
            "roi_source_counts": source_counts,
            "roi_mean_coverage": mean_coverage,
        }

    def _sample_crop_scale(self):
        return float(torch.empty(1).uniform_(self.roi_min_crop_scale, self.roi_max_crop_scale).item())

    def _sample_jitter(self):
        return (
            (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
            (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
        )

    def _random_full_image_crop(self, image):
        """Random crop at a sampled scale from anywhere in the image (no ROI needed)."""
        scale = self._sample_crop_scale()
        w, h = image.size
        crop_w = max(1, int(scale * w))
        crop_h = max(1, int(scale * h))
        left = int(torch.randint(0, max(1, w - crop_w + 1), (1,)).item())
        top = int(torch.randint(0, max(1, h - crop_h + 1), (1,)).item())
        return image.crop((left, top, left + crop_w, top + crop_h))

    def _random_full_image_crop_fixed_scale(self, image, min_scale, max_scale):
        """
        Random crop from the image with a fixed scale range.

        Args:
            image: PIL Image to crop from
            min_scale: Minimum crop scale (e.g., 0.9 for 90%)
            max_scale: Maximum crop scale (e.g., 1.0 for 100%)

        Returns:
            Cropped PIL Image
        """
        scale = float(torch.empty(1).uniform_(min_scale, max_scale).item())
        w, h = image.size
        crop_w = max(1, int(scale * w))
        crop_h = max(1, int(scale * h))
        left = int(torch.randint(0, max(1, w - crop_w + 1), (1,)).item())
        top = int(torch.randint(0, max(1, h - crop_h + 1), (1,)).item())
        return image.crop((left, top, left + crop_w, top + crop_h))

    def _build_view(self, image, use_roi, roi_record):
        """Build one view: ROI crop (if use_roi and record available) or random crop.

        Both branches sample scale uniformly from [roi_min_crop_scale, 1.0].
        The only difference is where the crop is centred:
          - ROI crop:    centred on the lesion bbox with a small jitter
          - Random crop: top-left corner drawn uniformly over all valid positions
        This holds regardless of whether roi_guidance_active is True or False.
        """
        if use_roi and roi_record is not None:
            selected = self._select_roi_record(roi_record)
            jitter_xy = self._sample_jitter()
            roi_scale = float(torch.empty(1).uniform_(self.roi_min_crop_scale, 1.0).item())
            crop = crop_image_to_roi(
                image=image,
                roi_record=selected,
                context_scale=self.roi_context_scale,
                min_crop_scale=roi_scale,
                jitter_xy=jitter_xy,
                max_aspect_ratio=self.roi_max_aspect_ratio,
            )
            return self.roi_transform2(crop)
        # Random crop: same scale range [min, 1.0], uniform position
        crop = self._random_full_image_crop_fixed_scale(image, self.roi_min_crop_scale, 1.0)
        return self.transform1(crop)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])

        in_warmup = self.current_epoch < self.roi_warmup_epochs
        canonical_path = canonicalize_image_path(img_path)

        # ROI is only available for positives and only after warmup
        roi_record = None
        if (
            label == self.roi_target_label
            and not in_warmup
            and self.roi_guidance_active
        ):
            roi_record = self.roi_records.get(canonical_path)

        # Positives: each view independently decides ROI vs random with roi_focus_prob
        # Negatives: always random crop (0% ROI)
        if label == self.roi_target_label:
            v1_use_roi = float(torch.rand(1).item()) < self.roi_focus_prob
            v2_use_roi = float(torch.rand(1).item()) < self.roi_focus_prob
        else:
            v1_use_roi = False
            v2_use_roi = False

        image1 = self._build_view(image, v1_use_roi, roi_record)
        image2 = self._build_view(image, v2_use_roi, roi_record)
        return image1, image2, label


class SupproROIDataset(Dataset):
    def __init__(
        self,
        df,
        global_transform1,
        global_transform2,
        roi_transform,
        roi_target_label=1,
        roi_context_scale=2.0,
        roi_min_crop_scale=0.4,
        roi_max_crop_scale=1.0,
        roi_center_jitter=0.05,
        roi_max_aspect_ratio=1.5,
        hard_neg_transform=None,
        hard_neg_target_label=0,
    ):
        self.df = df.reset_index(drop=True)
        self.global_transform1 = global_transform1
        self.global_transform2 = global_transform2
        self.roi_transform = roi_transform
        self.roi_target_label = int(roi_target_label)
        self.roi_context_scale = max(float(roi_context_scale), 1e-6)
        self.roi_min_crop_scale = min(max(float(roi_min_crop_scale), 1e-6), 1.0)
        self.roi_max_crop_scale = min(max(float(roi_max_crop_scale), self.roi_min_crop_scale), 1.0)
        self.roi_center_jitter = max(float(roi_center_jitter), 0.0)
        self.roi_max_aspect_ratio = max(float(roi_max_aspect_ratio), 1.0)
        self.roi_records = {}
        self.roi_guidance_active = False
        # Hard-negative mining: ROI crops on ndbe (label==0) images that the previous
        # finetune model false-fired on. These act as extra near-decision-boundary
        # anchors in the SupCon objective ("lesion-like != lesion").
        self.hard_neg_transform = hard_neg_transform if hard_neg_transform is not None else roi_transform
        self.hard_neg_target_label = int(hard_neg_target_label)
        self.hard_neg_roi_records = {}
        self.hard_neg_roi_guidance_active = False

    def __len__(self):
        return len(self.df)

    def set_roi_records(self, roi_records, active=True):
        self.roi_records = {
            canonicalize_image_path(key): value for key, value in roi_records.items()
        }
        self.roi_guidance_active = bool(active) and len(self.roi_records) > 0

    def clear_roi_records(self):
        self.roi_records = {}
        self.roi_guidance_active = False

    def set_hard_neg_roi_records(self, roi_records, active=True):
        """Register hard-negative ROI records (ndbe images flagged by the prior model)."""
        self.hard_neg_roi_records = {
            canonicalize_image_path(key): value for key, value in roi_records.items()
        }
        self.hard_neg_roi_guidance_active = (
            bool(active) and len(self.hard_neg_roi_records) > 0
        )

    def clear_hard_neg_roi_records(self):
        self.hard_neg_roi_records = {}
        self.hard_neg_roi_guidance_active = False

    def get_roi_guidance_stats(self):
        positive_image_paths = self.df.loc[
            self.df["label"] == self.roi_target_label, "img"
        ].astype(str).map(canonicalize_image_path).tolist()
        covered_records = [
            self.roi_records[path] for path in positive_image_paths if path in self.roi_records
        ]
        source_counts = {}
        for record in covered_records:
            source = str(record.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        mean_coverage = 0.0
        if covered_records:
            mean_coverage = float(
                sum(float(record.get("coverage", 0.0)) for record in covered_records)
                / len(covered_records)
            )

        return {
            "roi_guidance_active": bool(self.roi_guidance_active),
            "roi_positive_images": len(covered_records),
            "roi_positive_candidates": len(positive_image_paths),
            "roi_source_counts": source_counts,
            "roi_mean_coverage": mean_coverage,
        }

    def get_hard_neg_roi_guidance_stats(self):
        negative_image_paths = self.df.loc[
            self.df["label"] == self.hard_neg_target_label, "img"
        ].astype(str).map(canonicalize_image_path).tolist()
        covered_records = [
            self.hard_neg_roi_records[path]
            for path in negative_image_paths
            if path in self.hard_neg_roi_records
        ]
        source_counts = {}
        for record in covered_records:
            source = str(record.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        mean_coverage = 0.0
        if covered_records:
            mean_coverage = float(
                sum(float(record.get("coverage", 0.0)) for record in covered_records)
                / len(covered_records)
            )

        return {
            "hard_neg_roi_guidance_active": bool(self.hard_neg_roi_guidance_active),
            "hard_neg_roi_negative_images": len(covered_records),
            "hard_neg_roi_negative_candidates": len(negative_image_paths),
            "hard_neg_roi_source_counts": source_counts,
            "hard_neg_roi_mean_coverage": mean_coverage,
        }

    def _sample_crop_scale(self):
        return float(torch.empty(1).uniform_(self.roi_min_crop_scale, self.roi_max_crop_scale).item())

    def _sample_jitter(self):
        return (
            (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
            (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
        )

    def _select_roi_record(self, roi_record):
        roi_islands = roi_record.get("roi_islands")
        if not roi_islands:
            return roi_record

        island_index = int(torch.randint(len(roi_islands), (1,)).item())
        return roi_islands[island_index]

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])

        global_view1 = self.global_transform1(image)
        global_view2 = self.global_transform2(image)
        roi_view = global_view1.clone()
        has_roi = False
        hard_neg_view = global_view1.clone()
        has_hard_neg = False

        canonical_path = canonicalize_image_path(img_path)
        roi_record = self.roi_records.get(canonical_path)
        if (
            self.roi_guidance_active
            and label == self.roi_target_label
            and roi_record is not None
        ):
            selected_roi_record = self._select_roi_record(roi_record)
            roi_image = crop_image_to_roi(
                image=image,
                roi_record=selected_roi_record,
                context_scale=self.roi_context_scale,
                min_crop_scale=self._sample_crop_scale(),
                jitter_xy=self._sample_jitter(),
                max_aspect_ratio=self.roi_max_aspect_ratio,
            )
            roi_view = self.roi_transform(roi_image)
            has_roi = True

        hard_neg_record = self.hard_neg_roi_records.get(canonical_path)
        if (
            self.hard_neg_roi_guidance_active
            and label == self.hard_neg_target_label
            and hard_neg_record is not None
        ):
            selected_hard_neg_record = self._select_roi_record(hard_neg_record)
            hard_neg_image = crop_image_to_roi(
                image=image,
                roi_record=selected_hard_neg_record,
                context_scale=self.roi_context_scale,
                min_crop_scale=self._sample_crop_scale(),
                jitter_xy=self._sample_jitter(),
                max_aspect_ratio=self.roi_max_aspect_ratio,
            )
            hard_neg_view = self.hard_neg_transform(hard_neg_image)
            has_hard_neg = True

        return (
            global_view1,
            global_view2,
            roi_view,
            hard_neg_view,
            label,
            bool(has_roi),
            bool(has_hard_neg),
        )


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size, pos_ratio=0.5, drop_last=False, generator=None):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if not 0.0 < pos_ratio < 1.0:
            raise ValueError(f"pos_ratio must be in (0, 1), got {pos_ratio}.")

        self.batch_size = int(batch_size)
        self.n_pos = max(1, round(batch_size * pos_ratio))
        self.n_neg = self.batch_size - self.n_pos
        if self.n_neg < 1:
            raise ValueError(
                f"pos_ratio={pos_ratio} with batch_size={batch_size} leaves no room for negatives."
            )
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.positive_indices = [
            index for index, label in enumerate(labels) if int(label) == 1
        ]
        self.negative_indices = [
            index for index, label in enumerate(labels) if int(label) == 0
        ]

        if not self.positive_indices or not self.negative_indices:
            raise ValueError(
                "BalancedBatchSampler requires at least one positive and one negative sample."
            )

        if self.drop_last:
            self.num_batches = len(labels) // self.batch_size
        else:
            self.num_batches = int(math.ceil(len(labels) / float(self.batch_size)))

    def __iter__(self):
        pos_tensor = torch.tensor(self.positive_indices, dtype=torch.long)
        neg_tensor = torch.tensor(self.negative_indices, dtype=torch.long)

        # Shuffle each class pool independently at epoch start; reshuffle when exhausted
        # so each sample is seen floor or ceil times rather than with high replacement variance.
        pos_pool = pos_tensor[torch.randperm(len(pos_tensor), generator=self.generator)]
        neg_pool = neg_tensor[torch.randperm(len(neg_tensor), generator=self.generator)]
        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.num_batches):
            pos_indices = []
            for _ in range(self.n_pos):
                if pos_ptr >= len(pos_pool):
                    pos_pool = pos_tensor[torch.randperm(len(pos_tensor), generator=self.generator)]
                    pos_ptr = 0
                pos_indices.append(pos_pool[pos_ptr].item())
                pos_ptr += 1

            neg_indices = []
            for _ in range(self.n_neg):
                if neg_ptr >= len(neg_pool):
                    neg_pool = neg_tensor[torch.randperm(len(neg_tensor), generator=self.generator)]
                    neg_ptr = 0
                neg_indices.append(neg_pool[neg_ptr].item())
                neg_ptr += 1

            batch = torch.tensor(pos_indices + neg_indices, dtype=torch.long)
            permutation = torch.randperm(batch.numel(), generator=self.generator)
            yield batch[permutation].tolist()

    def __len__(self):
        return self.num_batches


class SimpleDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])
        return self.transform(image), label


def build_eval_transform(input_size):
    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_roi_focus_transform(input_size, augmentation_intensity=3):
    """
    Build ROI-focus training transform with configurable augmentation intensity.

    Args:
        input_size: Target image size
        augmentation_intensity: 1 (low), 2 (medium), 3 (strong)
    """
    if augmentation_intensity not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"augmentation_intensity must be in {list(AUGMENTATION_PRESETS.keys())}, "
            f"got {augmentation_intensity}"
        )

    config = AUGMENTATION_PRESETS[augmentation_intensity]

    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=config["h_flip_p"]),
        RandomVerticalFlip(p=config["v_flip_p"]),
        RandomRotation(degrees=config["rotation_deg"]),
        ColorJitter(
            brightness=config["color_brightness"],
            contrast=config["color_contrast"],
            saturation=config["color_saturation"],
            hue=config["color_hue"]
        ),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_suppro_roi_train_transform(input_size, augmentation_intensity=3):
    """
    Build SupPro ROI training transform with configurable augmentation intensity.
    Uses lighter augmentation than full-frame training.

    Args:
        input_size: Target image size
        augmentation_intensity: 1 (low), 2 (medium), 3 (strong)
    """
    if augmentation_intensity not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"augmentation_intensity must be in {list(AUGMENTATION_PRESETS.keys())}, "
            f"got {augmentation_intensity}"
        )

    config = AUGMENTATION_PRESETS[augmentation_intensity]

    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=config["h_flip_p"]),
        RandomVerticalFlip(p=config["v_flip_p"]),
        RandomRotation(degrees=config["roi_rotation_deg"]),
        ColorJitter(
            brightness=config["roi_color_brightness"],
            contrast=config["roi_color_contrast"],
            saturation=config["roi_color_saturation"],
            hue=config["roi_color_hue"]
        ),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataset_dataframe(data_dir):
    centers = sorted([
        folder
        for folder in os.listdir(data_dir)
        if folder.startswith("center") and os.path.isdir(os.path.join(data_dir, folder))
    ])
    if not centers:
        raise ValueError(f"No center folders found in {data_dir}.")

    all_images = []
    all_labels = []
    all_centers = []
    class_names = None

    for center in centers:
        center_path = os.path.join(data_dir, center)
        ds = ImageFolder(root=center_path)

        if ds.class_to_idx != {"ndbe": 0, "neo": 1}:
            raise ValueError(f"Class mapping mismatch in {center}: {ds.class_to_idx}")

        if class_names is None:
            class_names = sorted(ds.class_to_idx.keys(), key=lambda name: ds.class_to_idx[name])

        for img_path, label in sorted(
            ds.samples,
            key=lambda item: (int(item[1]), canonicalize_image_path(item[0])),
        ):
            all_images.append(img_path)
            all_labels.append(label)
            all_centers.append(center)

    df = pd.DataFrame({"img": all_images, "label": all_labels, "center": all_centers})
    df["img"] = df["img"].map(canonicalize_image_path)
    df = df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    return df, class_names


def build_train_val_dataframes(data_dir, test_size=0.2, random_state=42):
    df, class_names = build_dataset_dataframe(data_dir)
    split_df = df.copy()
    split_df["stratify_col"] = (
        split_df["center"].astype(str) + "_" + split_df["label"].astype(str)
    )

    train_df, val_df = train_test_split(
        split_df,
        test_size=test_size,
        stratify=split_df["stratify_col"],
        random_state=random_state,
    )
    train_df = train_df.drop(columns=["stratify_col"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["stratify_col"]).reset_index(drop=True)
    train_df = train_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    return train_df, val_df, class_names


def prepare_datasets(args, device):
    input_size = getattr(args, "input_size", 336)
    data_dir = DEFAULT_DATA_DIR
    num_folds = int(getattr(args, "num_folds", 1))
    fold_index = int(getattr(args, "fold_index", 0))
    seed = int(getattr(args, "seed", 42))
    augmentation_intensity = int(getattr(args, "augmentation_intensity", 3))
    train_generator = build_seeded_generator(seed)
    valid_generator = build_seeded_generator(seed + 1)
    print(f"Using input size: {input_size}x{input_size}")
    print(f"Using augmentation intensity: {augmentation_intensity} ({AUGMENTATION_PRESETS[augmentation_intensity]['name']})")

    # View 1: full-frame light augmentation (no random crop — preserve full context).
    train_transform_1 = build_roi_focus_transform(input_size, augmentation_intensity=augmentation_intensity)
    # View 2 fallback (ndbe, or neo without a record): same full-frame light aug.
    train_transform_2 = build_roi_focus_transform(input_size, augmentation_intensity=augmentation_intensity)
    # View 2 when an ROI record exists for a neo sample: crop to ROI then light aug.
    train_roi_transform_2 = build_roi_focus_transform(input_size, augmentation_intensity=augmentation_intensity)
    valid_transform = build_eval_transform(input_size)

    df, class_names = build_dataset_dataframe(data_dir)
    df["stratify_col"] = df["center"].astype(str) + "_" + df["label"].astype(str)

    if getattr(args, "loco", False):
        # Leave-one-center-out cross-validation.
        # fold_index k -> the k-th center (sorted alphabetically) is the validation set,
        # all other centers are the training set. With 2 centers, k=0 holds out center_1
        # (val on center_1, train on center_2) and k=1 holds out center_2.
        centers_sorted = sorted(df["center"].unique())
        if len(centers_sorted) < 2:
            raise ValueError(
                f"--loco requires at least 2 centers in the dataset, got {centers_sorted}."
            )
        if not (0 <= fold_index < len(centers_sorted)):
            raise ValueError(
                f"--fold-index must be in [0, {len(centers_sorted) - 1}] when --loco is set, "
                f"got {fold_index}."
            )
        holdout_center = centers_sorted[fold_index]
        train_centers = [c for c in centers_sorted if c != holdout_center]
        val_mask = df["center"] == holdout_center
        train_df = df.loc[~val_mask].copy()
        val_df = df.loc[val_mask].copy()
        print(
            f"LOCO split | fold {fold_index}: "
            f"train centers={train_centers} ({len(train_df)} imgs) | "
            f"val center={holdout_center} ({len(val_df)} imgs)"
        )
        train_label_counts = train_df["label"].value_counts().to_dict()
        val_label_counts = val_df["label"].value_counts().to_dict()
        print(
            f"LOCO split | label counts: "
            f"train={train_label_counts} | val={val_label_counts}"
        )
    elif num_folds <= 1:
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["stratify_col"],
            random_state=seed,
        )
        print("Using single validation split (80/20 stratified).")
    else:
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(
                f"fold_index must be in [0, {num_folds - 1}], got {fold_index}."
            )
        splitter = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=getattr(args, "seed", 42),
        )
        split_indices = list(splitter.split(df, df["stratify_col"]))
        train_indices, val_indices = split_indices[fold_index]
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        print(
            f"Using {num_folds}-fold cross-validation | "
            f"fold {fold_index + 1}/{num_folds} as validation."
        )

    train_df = train_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    val_df = val_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)

    train_ds = TwoViewDataset(
        train_df,
        train_transform_1,
        train_transform_2,
        roi_transform2=train_roi_transform_2,
        roi_target_label=getattr(args, "gradcam_target_class", 1),
        roi_focus_prob=getattr(args, "roi_focus_prob", 1.0),
        roi_negative_focus_prob=getattr(args, "roi_negative_focus_prob", 0.0),
        roi_context_scale=getattr(args, "roi_context_scale", 2.0),
        roi_min_crop_scale=getattr(args, "roi_min_crop_scale", 0.6),
        roi_max_crop_scale=getattr(args, "roi_max_crop_scale", 1.0),
        roi_center_jitter=getattr(args, "roi_center_jitter", 0.05),
        roi_max_aspect_ratio=getattr(args, "roi_max_aspect_ratio", 1.5),
        roi_warmup_epochs=getattr(args, "roi_warmup_epochs", 0),
    )
    valid_ds = SimpleDataset(val_df, valid_transform)

    if getattr(args, "balanced_sampler", False):
        train_loader = DataLoader(
            train_ds,
            batch_sampler=BalancedBatchSampler(
                labels=train_df["label"].tolist(),
                batch_size=args.batch_size,
                pos_ratio=getattr(args, "pos_ratio", 0.5),
                generator=train_generator,
            ),
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=train_generator,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=train_generator,
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=valid_generator,
    )
    return train_loader, valid_loader, train_ds, valid_ds, class_names
