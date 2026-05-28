"""Docker entry point for challenge submissions.

Supports two checkpoint shapes:
  1. Single fitted-head .pt  (model_state_dict + sklearn_head_state pickle)
  2. LOCO ensemble bundle    (is_ensemble=True, folds=[per-fold payloads])

The ensemble path averages each fold's positive-class probabilities at
inference time, mirroring how the LOCO ensemble is scored in train.py.
"""

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
MODEL_METADATA_KEYS = {
    "backbone_preset",
    "num_folds",
    "fold_index",
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


def _clean_model_kwargs(model_kwargs):
    model_kwargs = dict(model_kwargs)
    model_kwargs["pretrained"] = False
    model_kwargs.pop("backbone_weights_path", None)
    dropped = {k: model_kwargs.pop(k) for k in tuple(model_kwargs) if k in MODEL_METADATA_KEYS}
    return model_kwargs, dropped


def _restore_sklearn_head(model, payload):
    """Restore a fitted sklearn estimator into model.head._clf. Supports both
    the new top-level key 'sklearn_head_state' (pickled bytes) and the old
    'sklearn_knn_estimator' / 'sklearn_svm_estimator' (live object) layouts."""
    if not model.is_sklearn_head:
        return False
    import pickle as _pickle
    state = payload.get("sklearn_head_state")
    if state is not None:
        model.head._clf = _pickle.loads(state)
        model.head._fitted = True
        return True
    for key in ("sklearn_knn_estimator", "sklearn_svm_estimator", "sklearn_estimator"):
        estimator = payload.get(key)
        if estimator is not None:
            model.head._clf = estimator
            model.head._fitted = True
            return True
    return False


def _build_single_model(payload, device):
    fallback = dict(DEFAULT_MODEL_KWARGS)
    model_kwargs = resolve_model_kwargs_from_checkpoint(payload, fallback_kwargs=fallback)
    model_kwargs, _dropped = _clean_model_kwargs(model_kwargs)
    input_size = int(model_kwargs.get("input_size", DEFAULT_MODEL_KWARGS["input_size"]))
    model = Model(**model_kwargs).to(device)
    state_dict = payload.get("model_state_dict")
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    restored = _restore_sklearn_head(model, payload)
    if model.is_sklearn_head and not restored:
        raise RuntimeError(
            "Checkpoint uses a sklearn head but no fitted estimator was found "
            "(expected 'sklearn_head_state' or 'sklearn_*_estimator' key)."
        )
    model.eval()
    return model, model_kwargs.get("head_type", "unknown"), input_size


@torch.no_grad()
def _predict_pos_probs(model, images):
    return torch.softmax(model(images), dim=1)[:, 1]


def main():
    os.makedirs("/output", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: sklearn estimators inside the checkpoint are pickled,
    # which PyTorch 2.6+'s safe loader refuses to unpickle by default. The
    # checkpoint comes from our own training scripts so trusting its pickle is fine.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    is_ensemble = isinstance(checkpoint, dict) and bool(checkpoint.get("is_ensemble"))

    if is_ensemble:
        fold_payloads = checkpoint.get("folds", [])
        if len(fold_payloads) < 2:
            raise ValueError(f"Ensemble bundle has {len(fold_payloads)} folds; need >=2.")
        print(
            f"Detected ensemble bundle | n_folds={len(fold_payloads)} | "
            f"fold_indices={checkpoint.get('fold_indices')}"
        )
        fold_models = []
        input_size_ref = None
        for idx, fp in enumerate(fold_payloads):
            print(f"[Fold {idx}]")
            model, _head_type, input_size = _build_single_model(fp, device)
            fold_models.append(model)
            input_size_ref = input_size_ref or input_size
            if input_size != input_size_ref:
                raise ValueError(
                    f"Inconsistent input_size across folds: {input_size} vs {input_size_ref}."
                )
        input_size = input_size_ref
    else:
        model, head_type, input_size = _build_single_model(checkpoint, device)
        # If this was the original single-fold path that also has 'model_state_dict'
        # at top level, load_model_checkpoint would also work; _build_single_model
        # already handles it.
        print(f"Single-model checkpoint | head_type={head_type} | input_size={input_size}")
        fold_models = [model]

    transform = Compose([
        ToImage(),
        Resize((input_size, input_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = TestDataset(TEST_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with open(OUT_FILE, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])
        with torch.no_grad():
            for images, sample_ids in loader:
                images = images.to(device)
                if len(fold_models) == 1:
                    probs = _predict_pos_probs(fold_models[0], images).cpu().tolist()
                else:
                    fold_probs = [_predict_pos_probs(m, images) for m in fold_models]
                    probs = torch.stack(fold_probs, dim=0).mean(dim=0).cpu().tolist()
                for sample_id, probability in zip(sample_ids, probs):
                    writer.writerow([sample_id, probability])

    print(f"Wrote {len(dataset)} predictions to {OUT_FILE}")


if __name__ == "__main__":
    main()
