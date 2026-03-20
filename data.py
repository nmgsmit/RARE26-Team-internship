import os
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize

# --- Custom Dataset Classes ---


class SimpleDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'img']
        img = Image.open(img_path).convert('RGB')
        label = int(self.df.loc[idx, 'label'])
        return self.transform(img), label

# --- Helper Functions ---



# --- Main Preparation Function ---

def prepare_datasets(args, device):
    # Compose transforms
    transform = Compose([
        ToImage(),
        Resize((224, 224)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Identify Center Folders
    centers = [
        f for f in os.listdir(args.data_dir)
        if f.startswith('center') and os.path.isdir(os.path.join(args.data_dir, f))
    ]
    if not centers:
        raise ValueError(f"No center folders found in {args.data_dir}.")

    # 2. Collect all image metadata
    all_images, all_labels, all_centers = [], [], []
    
    class_names = None
    for center in centers:
        center_path = os.path.join(args.data_dir, center)
        ds = ImageFolder(root=center_path) # We just need the paths/labels here
        # Verify class mapping consistency
        if ds.class_to_idx != {'ndbe': 0, 'neo': 1}:
            raise ValueError(f"Class mapping mismatch in {center}: {ds.class_to_idx}")
        if class_names is None:
            # Save class names from the first center (should be the same for all)
            class_names = list(ds.class_to_idx.keys())
        for img_path, label in ds.samples:
            all_images.append(img_path)
            all_labels.append(label)
            all_centers.append(center)

    # 3. Stratification and Splitting
    df = pd.DataFrame({'img': all_images, 'label': all_labels, 'center': all_centers})
    df['stratify_col'] = df['center'].astype(str) + '_' + df['label'].astype(str)
    
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, # Added default split ratio
        stratify=df['stratify_col'], 
        random_state=42
    )

    # Match the challenge prevalence in validation by oversampling negatives
    # until positives make up approximately 1% of the validation set.
    pos_label = 1
    neg_label = 0
    pos_df = val_df[val_df["label"] == pos_label]
    neg_df = val_df[val_df["label"] == neg_label]

    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError(
            f"Validation split must contain both classes, found positives={len(pos_df)} negatives={len(neg_df)}."
        )

    target_negatives = len(pos_df) * 99
    if target_negatives > len(neg_df):
        extra_negatives = neg_df.sample(
            n=target_negatives - len(neg_df),
            replace=True,
            random_state=args.seed,
        )
        val_df = pd.concat([pos_df, neg_df, extra_negatives], ignore_index=True)
    else:
        val_df = pd.concat([pos_df, neg_df], ignore_index=True)

    val_df = val_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_pos_prevalence = len(pos_df) / len(val_df)
    print(
        f"Validation prevalence adjusted to {val_pos_prevalence:.2%} positives "
        f"({len(pos_df)} positive / {len(val_df) - len(pos_df)} negative)."
    )

    # 4. Create Datasets and Loaders
    train_ds = SimpleDataset(train_df, transform)
    valid_ds = SimpleDataset(val_df, transform)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, valid_loader, train_ds, valid_ds, class_names
