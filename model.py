import math
from pathlib import Path
from PIL import Image
import random
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================

DATA_ROOT = r"C:\Users\tkorz\OneDrive\Documents\Masters-tkorzhan\Q3\Team internship\dataset\train_anoniem"
IMG_SIZE = 518
EPOCHS = 10
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CHECKPOINT LOADER (your code, unchanged)
# ==========================================

def _unwrap_checkpoint_state_dict(checkpoint):
    for key in ("teacher", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _extract_backbone_state_dict(checkpoint):
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    backbone_state = {}

    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        if clean_key.startswith("backbone."):
            clean_key = clean_key.removeprefix("backbone.")
        else:
            continue

        if clean_key == "register_tokens":
            clean_key = "reg_token"

        backbone_state[clean_key] = value

    return backbone_state


def _adapt_position_embeddings(backbone_state, backbone):
    if "pos_embed" not in backbone_state:
        return backbone_state

    pos_embed = backbone_state["pos_embed"]

    if pos_embed.shape != backbone.pos_embed.shape:
        pos_embed = resample_abs_pos_embed(
            pos_embed,
            new_size=backbone.patch_embed.grid_size,
            num_prefix_tokens=1,
        )

    backbone_state["pos_embed"] = pos_embed
    return backbone_state


def load_backbone_weights(backbone, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = _extract_backbone_state_dict(checkpoint)
    state = _adapt_position_embeddings(state, backbone)
    backbone.load_state_dict(state, strict=False)


# ==========================================
# DATA
# ==========================================

def build_samples(root):
    samples = []
    label_map = {"ndbe": 0, "neo": 1}

    for center in ["center_1", "center_2"]:
        for cls in label_map:
            folder = f"{root}/{center}/{cls}"
            for ext in ["*.jpg", "*.png"]:
                for f in glob(f"{folder}/{ext}"):
                    samples.append((f, label_map[cls]))

    return samples


samples = build_samples(DATA_ROOT)
labels = [s[1] for s in samples]

train_samples, val_samples = train_test_split(
    samples, test_size=0.2, stratify=labels
)

# ==========================================
# TRANSFORMS
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# ==========================================
# BACKBONE (DINOv2-style)
# ==========================================

class DinoBackbone(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=(weights_path is None),
            num_classes=0
        )

        if weights_path is not None:
            load_backbone_weights(self.model, weights_path)

        self.embed_dim = self.model.embed_dim

    def forward(self, x):
        out = self.model.forward_features(x)

        if isinstance(out, dict):
            tokens = out["x_norm_patchtokens"]
        else:
            tokens = out[:, 1:, :]

        return tokens.squeeze(0)


# ==========================================
# MIL MODEL
# ==========================================

class AttentionMIL(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.cls = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, feats):
        A = self.attn(feats)
        A = torch.softmax(A.squeeze(-1), dim=0)

        M = torch.sum(A.unsqueeze(-1) * feats, dim=0)

        logits = self.cls(M.unsqueeze(0))
        return logits, A


# ==========================================
# INIT
# ==========================================

backbone = DinoBackbone(
    weights_path="gastro_checkpoint.pth"  
).to(device)

model = AttentionMIL(768).to(device)

# Freeze most layers (recommended)
for name, p in backbone.named_parameters():
    if "blocks.11" in name or "blocks.10" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(model.parameters()) + 
    [p for p in backbone.parameters() if p.requires_grad],
    lr=LR
)

# ==========================================
# TRAIN
# ==========================================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    random.shuffle(train_samples)

    for path, label in train_samples:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        feats = backbone(img)
        logits, _ = model(feats)

        loss = criterion(logits, torch.tensor([label]).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_samples):.4f}")

