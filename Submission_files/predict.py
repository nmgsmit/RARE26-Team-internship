import csv
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

from model import Model, load_model_checkpoint, resolve_model_kwargs_from_checkpoint

TEST_DIR = "/data/test"
MODEL_PATH = "/app/model.pt"
OUT_FILE = "/output/predictions.csv"
BATCH_SIZE = 32
DEFAULT_MODEL_KWARGS = {
    "in_channels": 3,
    "n_classes": 2,
    "backbone_name": "vit_base_patch14_reg4_dinov2",
    "input_size": 336,
    "pretrained": False,
}


class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = test_dir
        self.transform = transform
        self.samples = sorted([
            fname for fname in os.listdir(test_dir)
            if os.path.isfile(os.path.join(test_dir, fname))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name = self.samples[idx]
        image_path = os.path.join(self.test_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        sample_id = os.path.splitext(image_name)[0]
        return self.transform(image), sample_id


def main():
    os.makedirs("/output", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_kwargs = resolve_model_kwargs_from_checkpoint(
        checkpoint,
        fallback_kwargs=DEFAULT_MODEL_KWARGS,
    )
    model_kwargs["pretrained"] = False

    input_size = int(model_kwargs.get("input_size", DEFAULT_MODEL_KWARGS["input_size"]))
    n_classes = int(model_kwargs.get("n_classes", DEFAULT_MODEL_KWARGS["n_classes"]))
    backbone_name = model_kwargs.get("backbone_name", DEFAULT_MODEL_KWARGS["backbone_name"])

    transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TestDataset(TEST_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(
        f"Resolved checkpoint model | backbone={backbone_name} | "
        f"input_size={input_size} | n_classes={n_classes} | "
        f"head_type={model_kwargs.get('head_type', 'unknown')}"
    )
    model = Model(
        **model_kwargs,
    ).to(device)
    load_model_checkpoint(model, MODEL_PATH, strict=True, map_location=device)
    model.eval()

    with open(OUT_FILE, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])

        with torch.no_grad():
            for images, sample_ids in loader:
                images = images.to(device)
                probabilities = torch.softmax(model(images), dim=1)[:, 1].cpu().tolist()
                for sample_id, probability in zip(sample_ids, probabilities):
                    writer.writerow([sample_id, probability])

    print(f"Wrote {len(dataset)} predictions to {OUT_FILE}")


if __name__ == "__main__":
    main()
