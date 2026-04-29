import csv
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

from model import Model

TEST_DIR = "/data/test"
MODEL_PATH = "/app/model.pt"
OUT_FILE = "/output/predictions.csv"
BATCH_SIZE = 32
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
INPUT_SIZE = 336


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

    transform = Compose([
        ToImage(),
        Resize((INPUT_SIZE, INPUT_SIZE)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TestDataset(TEST_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        in_channels=3,
        n_classes=2,
        backbone_name=BACKBONE_NAME,
        input_size=INPUT_SIZE,
        pretrained=False,
    ).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
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
