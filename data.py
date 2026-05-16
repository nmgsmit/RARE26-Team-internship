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

from roi_guidance import canonicalize_image_path, crop_image_to_roi

DEFAULT_DATA_DIR = "../data/Challenge_train_data"


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
        roi_context_scale=2.0,
        roi_min_crop_scale=0.4,
        roi_center_jitter=0.05,
        roi_max_aspect_ratio=1.5,
    ):
        self.df = df.reset_index(drop=True)
        self.transform1 = transform1
        self.transform2 = transform2
        self.roi_transform2 = roi_transform2 or transform2
        self.roi_target_label = int(roi_target_label)
        self.roi_focus_prob = min(max(float(roi_focus_prob), 0.0), 1.0)
        self.roi_context_scale = max(float(roi_context_scale), 1e-6)
        self.roi_min_crop_scale = min(max(float(roi_min_crop_scale), 1e-6), 1.0)
        self.roi_center_jitter = max(float(roi_center_jitter), 0.0)
        self.roi_max_aspect_ratio = max(float(roi_max_aspect_ratio), 1.0)
        self.roi_records = {}
        self.roi_guidance_active = False

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

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])
        view1 = self.transform1(image)
        transform2 = self.transform2
        image2 = image

        roi_record = self.roi_records.get(canonicalize_image_path(img_path))
        use_roi = (
            self.roi_guidance_active
            and label == self.roi_target_label
            and roi_record is not None
            and float(torch.rand(1).item()) <= self.roi_focus_prob
        )
        if use_roi:
            # Randomly select one island if multiple exist
            selected_roi_record = self._select_roi_record(roi_record)

            jitter_xy = (
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
            )
            image2 = crop_image_to_roi(
                image=image,
                roi_record=selected_roi_record,
                context_scale=self.roi_context_scale,
                min_crop_scale=self.roi_min_crop_scale,
                jitter_xy=jitter_xy,
                max_aspect_ratio=self.roi_max_aspect_ratio,
            )
            transform2 = self.roi_transform2

        return view1, transform2(image2), label


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
            jitter_xy = (
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
            )
            roi_image = crop_image_to_roi(
                image=image,
                roi_record=selected_roi_record,
                context_scale=self.roi_context_scale,
                min_crop_scale=self.roi_min_crop_scale,
                jitter_xy=jitter_xy,
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
            jitter_xy = (
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
            )
            hard_neg_image = crop_image_to_roi(
                image=image,
                roi_record=selected_hard_neg_record,
                context_scale=self.roi_context_scale,
                min_crop_scale=self.roi_min_crop_scale,
                jitter_xy=jitter_xy,
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
    def __init__(self, labels, batch_size, drop_last=False, generator=None):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if batch_size % 2 != 0:
            raise ValueError(
                f"BalancedBatchSampler requires an even batch size, got {batch_size}."
            )

        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.half_batch_size = self.batch_size // 2
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
        positive_indices = torch.tensor(self.positive_indices, dtype=torch.long)
        negative_indices = torch.tensor(self.negative_indices, dtype=torch.long)

        for _ in range(self.num_batches):
            pos_choice = positive_indices[
                torch.randint(
                    len(positive_indices),
                    (self.half_batch_size,),
                    generator=self.generator,
                )
            ]
            neg_choice = negative_indices[
                torch.randint(
                    len(negative_indices),
                    (self.half_batch_size,),
                    generator=self.generator,
                )
            ]
            batch = torch.cat([pos_choice, neg_choice], dim=0)
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


def build_roi_focus_transform(input_size):
    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_suppro_roi_train_transform(input_size):
    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=5),
        ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.005),
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
    train_generator = build_seeded_generator(seed)
    valid_generator = build_seeded_generator(seed + 1)
    print(f"Using input size: {input_size}x{input_size}")

    # View 1: full-frame light augmentation (no random crop — preserve full context).
    train_transform_1 = build_roi_focus_transform(input_size)
    # View 2 fallback (ndbe, or neo without a record): same full-frame light aug.
    train_transform_2 = build_roi_focus_transform(input_size)
    # View 2 when an ROI record exists for a neo sample: crop to ROI then light aug.
    train_roi_transform_2 = build_roi_focus_transform(input_size)
    valid_transform = build_eval_transform(input_size)

    df, class_names = build_dataset_dataframe(data_dir)
    df["stratify_col"] = df["center"].astype(str) + "_" + df["label"].astype(str)

    if num_folds <= 1:
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
        roi_context_scale=getattr(args, "roi_context_scale", 2.0),
        roi_min_crop_scale=getattr(args, "roi_min_crop_scale", 0.4),
        roi_center_jitter=getattr(args, "roi_center_jitter", 0.05),
        roi_max_aspect_ratio=getattr(args, "roi_max_aspect_ratio", 1.5),
    )
    valid_ds = SimpleDataset(val_df, valid_transform)

    if getattr(args, "balanced_sampler", False):
        train_loader = DataLoader(
            train_ds,
            batch_sampler=BalancedBatchSampler(
                labels=train_df["label"].tolist(),
                batch_size=args.batch_size,
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
