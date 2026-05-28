"""Datasets, augmentation presets, and split logic for the clean baseline.

Design choices for this branch:
  - No ROI-guided sampling. Two SupPro views are independent full-frame
    augmentations of the same image at the chosen intensity preset.
  - LOCO (leave-one-center-out) cross-validation is the default; pass
    ``--num-folds 1`` for a single 80/20 stratified split.
  - BalancedBatchSampler with a configurable positive ratio (default 0.2,
    i.e. the 20/80 mix that drove the best run).
"""

from __future__ import annotations

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
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
)


DEFAULT_DATA_DIR = "../data/Challenge_train_data"
CLASS_TO_IDX = {"ndbe": 0, "neo": 1}


# ---------------------------------------------------------------------------
# Augmentation presets
# ---------------------------------------------------------------------------
AUGMENTATION_PRESETS = {
    1: dict(name="low",     h_flip_p=0.1, v_flip_p=0.0, rotation_deg=2,  cb=0.05, cc=0.05, cs=0.05, ch=0.01),
    2: dict(name="medium",  h_flip_p=0.3, v_flip_p=0.1, rotation_deg=5,  cb=0.08, cc=0.08, cs=0.08, ch=0.015),
    3: dict(name="strong",  h_flip_p=0.5, v_flip_p=0.2, rotation_deg=10, cb=0.10, cc=0.10, cs=0.10, ch=0.02),
    4: dict(name="extreme", h_flip_p=0.5, v_flip_p=0.5, rotation_deg=45, cb=0.20, cc=0.20, cs=0.20, ch=0.03),
}


def build_eval_transform(input_size: int):
    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_train_transform(input_size: int, augmentation_intensity: int):
    if augmentation_intensity not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"augmentation_intensity must be one of {sorted(AUGMENTATION_PRESETS)}, "
            f"got {augmentation_intensity}"
        )
    p = AUGMENTATION_PRESETS[augmentation_intensity]
    return Compose([
        ToImage(),
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=p["h_flip_p"]),
        RandomVerticalFlip(p=p["v_flip_p"]),
        RandomRotation(degrees=p["rotation_deg"]),
        ColorJitter(brightness=p["cb"], contrast=p["cc"],
                     saturation=p["cs"], hue=p["ch"]),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------
def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_seeded_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class TwoViewDataset(Dataset):
    """Returns (view1, view2, label). Each view is an independent draw from
    the same train transform applied to the same source image."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        image = Image.open(row["img"]).convert("RGB")
        return self.transform(image), self.transform(image), int(row["label"])


class SimpleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        image = Image.open(row["img"]).convert("RGB")
        return self.transform(image), int(row["label"])


# ---------------------------------------------------------------------------
# BalancedBatchSampler: fixed positive:negative ratio per batch.
# ---------------------------------------------------------------------------
class BalancedBatchSampler(BatchSampler):
    """Each batch contains exactly ``round(batch_size * pos_ratio)`` positives
    and the rest negatives. Pools reshuffle when exhausted so every sample is
    seen roughly the same number of times per epoch."""

    def __init__(self, labels, batch_size: int, pos_ratio: float = 0.2,
                 drop_last: bool = False, generator: torch.Generator | None = None):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if not 0.0 < pos_ratio < 1.0:
            raise ValueError(f"pos_ratio must be in (0,1), got {pos_ratio}.")

        self.batch_size = int(batch_size)
        self.n_pos = max(1, round(batch_size * pos_ratio))
        self.n_neg = self.batch_size - self.n_pos
        if self.n_neg < 1:
            raise ValueError(
                f"pos_ratio={pos_ratio} with batch_size={batch_size} leaves no room for negatives."
            )
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.positive_indices = [i for i, y in enumerate(labels) if int(y) == 1]
        self.negative_indices = [i for i, y in enumerate(labels) if int(y) == 0]
        if not self.positive_indices or not self.negative_indices:
            raise ValueError(
                "BalancedBatchSampler requires at least one positive and one negative sample."
            )
        if self.drop_last:
            self.num_batches = len(labels) // self.batch_size
        else:
            self.num_batches = int(math.ceil(len(labels) / float(self.batch_size)))

    def __iter__(self):
        pos = torch.tensor(self.positive_indices, dtype=torch.long)
        neg = torch.tensor(self.negative_indices, dtype=torch.long)
        pos_pool = pos[torch.randperm(len(pos), generator=self.generator)]
        neg_pool = neg[torch.randperm(len(neg), generator=self.generator)]
        pos_ptr = neg_ptr = 0

        for _ in range(self.num_batches):
            pos_batch = []
            for _ in range(self.n_pos):
                if pos_ptr >= len(pos_pool):
                    pos_pool = pos[torch.randperm(len(pos), generator=self.generator)]
                    pos_ptr = 0
                pos_batch.append(int(pos_pool[pos_ptr]))
                pos_ptr += 1
            neg_batch = []
            for _ in range(self.n_neg):
                if neg_ptr >= len(neg_pool):
                    neg_pool = neg[torch.randperm(len(neg), generator=self.generator)]
                    neg_ptr = 0
                neg_batch.append(int(neg_pool[neg_ptr]))
                neg_ptr += 1
            batch = torch.tensor(pos_batch + neg_batch, dtype=torch.long)
            order = torch.randperm(batch.numel(), generator=self.generator)
            yield batch[order].tolist()

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Dataframe construction + splits.
# ---------------------------------------------------------------------------
def build_dataset_dataframe(data_dir: str):
    centers = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("center") and os.path.isdir(os.path.join(data_dir, f))
    )
    if not centers:
        raise ValueError(f"No center_* folders found under {data_dir}")

    all_images, all_labels, all_centers = [], [], []
    class_names = None
    for center in centers:
        ds = ImageFolder(root=os.path.join(data_dir, center))
        if ds.class_to_idx != CLASS_TO_IDX:
            raise ValueError(
                f"Class mapping mismatch in {center}: {ds.class_to_idx} (expected {CLASS_TO_IDX})"
            )
        if class_names is None:
            class_names = sorted(ds.class_to_idx, key=lambda c: ds.class_to_idx[c])
        for img_path, label in sorted(ds.samples, key=lambda kv: (int(kv[1]), kv[0])):
            all_images.append(img_path)
            all_labels.append(label)
            all_centers.append(center)

    df = pd.DataFrame({"img": all_images, "label": all_labels, "center": all_centers})
    df = df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    return df, class_names


def split_dataframe(df: pd.DataFrame, args):
    """Returns (train_df, val_df, holdout_center_or_None) using LOCO if
    ``args.loco`` else stratified k-fold (k=1 -> 80/20)."""
    fold_index = int(getattr(args, "fold_index", 0))
    if getattr(args, "loco", False):
        centers_sorted = sorted(df["center"].unique())
        if len(centers_sorted) < 2:
            raise ValueError(f"--loco needs >=2 centers, got {centers_sorted}")
        if not (0 <= fold_index < len(centers_sorted)):
            raise ValueError(
                f"--fold-index must be in [0, {len(centers_sorted) - 1}], got {fold_index}"
            )
        holdout = centers_sorted[fold_index]
        train_df = df.loc[df["center"] != holdout].copy()
        val_df = df.loc[df["center"] == holdout].copy()
        print(
            f"LOCO split | fold {fold_index} | holdout=center {holdout} | "
            f"train n={len(train_df)} | val n={len(val_df)}"
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), holdout

    num_folds = int(getattr(args, "num_folds", 1))
    seed = int(getattr(args, "seed", 42))
    stratify = df["center"].astype(str) + "_" + df["label"].astype(str)
    if num_folds <= 1:
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=stratify, random_state=seed,
        )
        print(f"Stratified 80/20 split | train n={len(train_df)} | val n={len(val_df)}")
    else:
        if not (0 <= fold_index < num_folds):
            raise ValueError(
                f"--fold-index must be in [0, {num_folds - 1}], got {fold_index}"
            )
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        idx = list(splitter.split(df, stratify))[fold_index]
        train_df = df.iloc[idx[0]].copy()
        val_df = df.iloc[idx[1]].copy()
        print(
            f"K-fold split | k={num_folds} | fold={fold_index} | "
            f"train n={len(train_df)} | val n={len(val_df)}"
        )

    train_df = train_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values(["center", "label", "img"], kind="stable").reset_index(drop=True)
    return train_df, val_df, None


# ---------------------------------------------------------------------------
# Public entry point used by train.py
# ---------------------------------------------------------------------------
def prepare_datasets(args, device):
    """Returns (train_loader, valid_loader, train_ds, valid_ds, class_names, train_df, val_df).
    The train loader uses BalancedBatchSampler when ``args.balanced_sampler`` is set."""
    del device  # not used; kept for backwards-compat call sites
    input_size = int(getattr(args, "input_size", 336))
    data_dir = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    aug_intensity = int(getattr(args, "augmentation_intensity", 3))
    seed = int(getattr(args, "seed", 42))

    print(f"Input size: {input_size}x{input_size}")
    print(f"Augmentation intensity: {aug_intensity} "
          f"({AUGMENTATION_PRESETS[aug_intensity]['name']})")

    train_transform = build_train_transform(input_size, aug_intensity)
    eval_transform = build_eval_transform(input_size)

    df, class_names = build_dataset_dataframe(data_dir)
    train_df, val_df, _holdout = split_dataframe(df, args)

    train_ds = TwoViewDataset(train_df, train_transform)
    valid_ds = SimpleDataset(val_df, eval_transform)
    train_generator = build_seeded_generator(seed)

    if getattr(args, "balanced_sampler", False):
        sampler = BalancedBatchSampler(
            labels=train_df["label"].tolist(),
            batch_size=int(args.batch_size),
            pos_ratio=float(getattr(args, "pos_ratio", 0.2)),
            drop_last=False,
            generator=train_generator,
        )
        print(
            f"BalancedBatchSampler | batch_size={sampler.batch_size} | "
            f"pos:{sampler.n_pos} neg:{sampler.n_neg} "
            f"(realized {sampler.n_pos / sampler.batch_size:.2%} positives)"
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=int(args.num_workers), pin_memory=True,
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=int(args.batch_size), shuffle=True,
            num_workers=int(args.num_workers), pin_memory=True,
            generator=train_generator, worker_init_fn=seed_worker,
        )

    valid_loader = DataLoader(
        valid_ds, batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return train_loader, valid_loader, train_ds, valid_ds, class_names, train_df, val_df
