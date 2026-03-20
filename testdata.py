import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from PIL import Image

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

def load_external_testset(testset_images_dir, batch_size, num_workers, device):
    transform = Compose([
        ToImage(),
        Resize((224, 224)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset_images_dir = Path(testset_images_dir)
    if not testset_images_dir.exists():
        raise FileNotFoundError(f"Testset images directory not found: {testset_images_dir}")
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    testset_image_paths = sorted(
        p for p in testset_images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes
    )
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
