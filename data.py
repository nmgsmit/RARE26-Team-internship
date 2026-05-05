import os

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

from roi_guidance import crop_image_to_roi


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
        self.roi_records = {}
        self.roi_guidance_active = False

    def __len__(self):
        return len(self.df)

    def set_roi_records(self, roi_records, active=True):
        self.roi_records = {str(key): value for key, value in roi_records.items()}
        self.roi_guidance_active = bool(active) and len(self.roi_records) > 0

    def clear_roi_records(self):
        self.roi_records = {}
        self.roi_guidance_active = False

    def get_roi_guidance_stats(self):
        positive_image_paths = self.df.loc[self.df["label"] == self.roi_target_label, "img"].astype(str).tolist()
        covered_records = [self.roi_records[path] for path in positive_image_paths if path in self.roi_records]
        source_counts = {}
        for record in covered_records:
            source = str(record.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        mean_coverage = 0.0
        if covered_records:
            mean_coverage = float(
                sum(float(record.get("coverage", 0.0)) for record in covered_records) / len(covered_records)
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

        roi_record = self.roi_records.get(str(img_path))
        use_roi = (
            self.roi_guidance_active
            and label == self.roi_target_label
            and roi_record is not None
            and float(torch.rand(1).item()) <= self.roi_focus_prob
        )
        if use_roi:
            jitter_xy = (
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
                (2.0 * float(torch.rand(1).item()) - 1.0) * self.roi_center_jitter,
            )
            image2 = crop_image_to_roi(
                image=image,
                bbox=roi_record["bbox"],
                context_scale=self.roi_context_scale,
                min_crop_scale=self.roi_min_crop_scale,
                jitter_xy=jitter_xy,
            )
            transform2 = self.roi_transform2

        return view1, transform2(image2), label


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


def prepare_datasets(args, device):
    input_size = getattr(args, "input_size", 336)
    print(f"Using input size: {input_size}x{input_size}")

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
    train_roi_transform_2 = build_roi_focus_transform(input_size)
    valid_transform = build_eval_transform(input_size)

    centers = [
        folder
        for folder in os.listdir(args.data_dir)
        if folder.startswith("center") and os.path.isdir(os.path.join(args.data_dir, folder))
    ]
    if not centers:
        raise ValueError(f"No center folders found in {args.data_dir}.")

    all_images = []
    all_labels = []
    all_centers = []
    class_names = None

    for center in centers:
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
        random_state=42,
    )
    val_df = val_df.reset_index(drop=True)

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
    )
    valid_ds = SimpleDataset(val_df, valid_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, valid_loader, train_ds, valid_ds, class_names
