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
    # Keep validation on the raw stratified split. The 1% setting is projected later in metrics/train,
    # so changing val_df here will also change threshold selection, W&B logging, and checkpoint ranking.

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
