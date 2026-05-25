"""
Dataloader. Based on the user's training-time data.py.
Added on top of the original:
- EvalArgs dataclass so callers can build an args-like object without argparse
- eval_only_loader() helper that returns only the validation loader,
  with the same train/val split as training (controlled by seed)
- flat_eval_loader() helper for evaluation sets where images sit in a single
  flat directory and labels are encoded in the filename (e.g. EVC_Barretts:
  pat01_im1_NDBT.png, pat08_im1_ACHD.png). This is for held-out test sets
  that don't follow the centerN/{ndbe,neo}/ structure.
"""
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TwoViewDataset(Dataset):
    def __init__(self, df, transform1, transform2):
        self.df = df.reset_index(drop=True)
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        img = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])
        x1 = self.transform1(img)
        x2 = self.transform2(img)
        return x1, x2, label


class SimpleDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        img = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])
        return self.transform(img), label, str(img_path)


def prepare_datasets(args, device=None):
    """
    Verbatim from the training pipeline. Returns
    (train_loader, valid_loader, train_ds, valid_ds, class_names).
    """
    input_size = getattr(args, "input_size", 336)
    print(f"Using input size: {input_size}x{input_size}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_transform_1 = Compose([
        ToImage(),
        RandomResizedCrop((input_size, input_size), scale=(0.6, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform_2 = Compose([
        ToImage(),
        RandomResizedCrop((input_size, input_size), scale=(0.6, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valid_transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    centers = [
        f for f in os.listdir(args.data_dir)
        if f.startswith("center") and os.path.isdir(os.path.join(args.data_dir, f))
    ]
    if not centers:
        raise ValueError(f"No center folders found in {args.data_dir}.")

    all_images, all_labels, all_centers = [], [], []
    class_names = None
    for center in sorted(centers):
        center_path = os.path.join(args.data_dir, center)
        ds = ImageFolder(root=center_path)
        if ds.class_to_idx != {"ndbe": 0, "neo": 1}:
            raise ValueError(f"Class mapping mismatch in {center}: {ds.class_to_idx}")
        if class_names is None:
            class_names = list(ds.class_to_idx.keys())
        for img_path, label in ds.samples:
            all_images.append(img_path)
            all_labels.append(label)
            all_centers.append(center)

    df = pd.DataFrame({"img": all_images, "label": all_labels, "center": all_centers})
    df["stratify_col"] = df["center"].astype(str) + "_" + df["label"].astype(str)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["stratify_col"],
        random_state=args.seed,
    )
    val_df = val_df.reset_index(drop=True)

    train_ds = TwoViewDataset(train_df, train_transform_1, train_transform_2)
    valid_ds = SimpleDataset(val_df, valid_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, valid_loader, train_ds, valid_ds, class_names


@dataclass
class EvalArgs:
    """Minimal args object for evaluation. Mirrors the fields
    `prepare_datasets` reads, with sensible defaults."""
    data_dir: str
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    input_size: int = 336


def eval_only_loader(
    data_dir,
    seed=42,
    batch_size=32,
    num_workers=4,
    input_size=336,
    split="val",
):
    """
    Build a loader for feature extraction on the *training-style* dataset
    (centerN/{ndbe,neo}/*.png).

    Args:
        split: one of "val" (held-out 20%), "train" (training 80%), or "all"
               (every image). All three use the deterministic validation
               transform (resize + normalize, no augmentation), so the model
               sees exactly the same view it would at inference time.

    Returns (loader, dataset, class_names).

    Note: the train/val split is reproduced from the training-time `seed`,
    so "val" gives you the same held-out images the model was evaluated
    against during training. Pass the same seed your training run used.
    """
    if split not in {"val", "train", "all"}:
        raise ValueError(f"split must be 'val', 'train', or 'all'; got {split!r}.")

    print(f"Using input size: {input_size}x{input_size}, split={split!r}")

    valid_transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Discover centers + collect (path, label, center)
    centers = [
        f for f in os.listdir(data_dir)
        if f.startswith("center") and os.path.isdir(os.path.join(data_dir, f))
    ]
    if not centers:
        raise ValueError(f"No center folders found in {data_dir}.")

    all_images, all_labels, all_centers = [], [], []
    class_names = None
    for center in sorted(centers):
        center_path = os.path.join(data_dir, center)
        ds = ImageFolder(root=center_path)
        if ds.class_to_idx != {"ndbe": 0, "neo": 1}:
            raise ValueError(f"Class mapping mismatch in {center}: {ds.class_to_idx}")
        if class_names is None:
            class_names = list(ds.class_to_idx.keys())
        for img_path, label in ds.samples:
            all_images.append(img_path)
            all_labels.append(label)
            all_centers.append(center)

    df = pd.DataFrame({"img": all_images, "label": all_labels, "center": all_centers})

    if split == "all":
        chosen_df = df
    else:
        # Reproduce the training-time split exactly
        df["stratify_col"] = df["center"].astype(str) + "_" + df["label"].astype(str)
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df["stratify_col"], random_state=seed,
        )
        chosen_df = (val_df if split == "val" else train_df).reset_index(drop=True)

    print(f"  loader will see {len(chosen_df)} images")
    dataset = SimpleDataset(chosen_df.reset_index(drop=True), valid_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    return loader, dataset, class_names


# -----------------------------------------------------------------------------
# Flat evaluation loader (e.g. EVC_Barretts test set)
# -----------------------------------------------------------------------------

# Filename-suffix to class-index mapping. Keep this aligned with the training
# class mapping {"ndbe": 0, "neo": 1}.
DEFAULT_SUFFIX_TO_LABEL = {
    "NDBT": 0,  # ndbe
    "ACHD": 1,  # neo
}


def _parse_label_from_filename(
    filename: str,
    suffix_to_label: dict[str, int],
):
    """
    Extract the label from a filename like 'pat01_im1_NDBT.png'.
    Looks for the trailing _<SUFFIX> right before the extension.
    Returns the integer label, or None if no recognised suffix is found.
    """
    stem = Path(filename).stem  # 'pat01_im1_NDBT'
    # Last underscore-delimited token is the suffix
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    suffix = parts[1].upper()  # be case-tolerant
    return suffix_to_label.get(suffix)

def flat_eval_loader(
    data_dir,
    batch_size=32,
    num_workers=4,
    input_size=336,
    suffix_to_label=None,
    class_names=None,
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
):  
    """
    Build a loader for a flat evaluation directory where labels are encoded
    in filenames, e.g.:

        data_dir/
            pat01_im1_NDBT.png   -> label 0 (ndbe)
            pat08_im1_ACHD.png   -> label 1 (neo)
            ...

    Uses the same validation transform (resize + normalize, no augmentation)
    as the training pipeline, so the model sees images identically to
    inference time.

    Args:
        data_dir: flat directory containing images.
        suffix_to_label: dict mapping uppercased filename suffix -> class index.
            Defaults to {"NDBT": 0, "ACHD": 1}.
        class_names: list of class names indexed by label. Defaults to
            ["ndbe", "neo"] to match the training pipeline.
        image_extensions: file extensions to consider.

    Returns:
        (loader, dataset, class_names)
    """
    suffix_to_label = suffix_to_label or DEFAULT_SUFFIX_TO_LABEL
    class_names = class_names or ["ndbe", "neo"]

    print(f"Using input size: {input_size}x{input_size}, flat eval set")
    print(f"  suffix mapping: {suffix_to_label}")

    valid_transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir_path = Path(data_dir)
    if not data_dir_path.is_dir():
        raise ValueError(f"Flat data_dir does not exist or is not a directory: {data_dir}")

    image_paths = sorted(
        p for p in data_dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )
    if not image_paths:
        raise ValueError(f"No image files found in {data_dir}.")

    rows = []
    skipped = []
    suffix_counts: dict[str, int] = {}
    for p in image_paths:
        label = _parse_label_from_filename(p.name, suffix_to_label)
        # Track which suffix this had (for diagnostics), even if unmapped
        stem_parts = p.stem.rsplit("_", 1)
        suffix = stem_parts[1].upper() if len(stem_parts) == 2 else "<no_suffix>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

        if label is None:
            skipped.append(p.name)
            continue
        rows.append({"img": str(p), "label": label})

    print(f"  found {len(image_paths)} files; suffix counts: {suffix_counts}")
    if skipped:
        print(f"  WARNING: skipped {len(skipped)} files with unrecognised suffix "
              f"(first few: {skipped[:5]})")
    if not rows:
        raise ValueError(
            f"No images in {data_dir} matched any suffix in {suffix_to_label}. "
            f"Saw suffixes: {list(suffix_counts.keys())}"
        )

    df = pd.DataFrame(rows)
    print(f"  loader will see {len(df)} images")
    print(f"  class balance: "
          f"{dict(zip(*np.unique(df['label'].values, return_counts=True)))}")

    dataset = SimpleDataset(df, valid_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    return loader, dataset, class_names