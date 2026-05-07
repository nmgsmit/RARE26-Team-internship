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

DEFAULT_DATA_DIR = "./data/Challenge_train_data"


class TwoViewDataset(Dataset):
    def __init__(self, df, transform1, transform2):
        self.df = df.reset_index(drop=True)
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img"]
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "label"])
        return self.transform1(image), self.transform2(image), label


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


def prepare_datasets(args, device):
    input_size = getattr(args, "input_size", 336)
    data_dir = DEFAULT_DATA_DIR
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
    valid_transform = build_eval_transform(input_size)

    centers = [
        folder
        for folder in os.listdir(data_dir)
        if folder.startswith("center") and os.path.isdir(os.path.join(data_dir, folder))
    ]
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

    train_ds = TwoViewDataset(train_df, train_transform_1, train_transform_2)
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
