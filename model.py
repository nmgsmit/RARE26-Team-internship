"""Backbone + projection-head + sklearn (KNN / SVM) classifier head.

Linear / MLP / cosine / residual heads were removed from this branch: the only
supported heads are post-hoc-fit sklearn KNN and SVM, which is what the best run
used.

Backbone zoo unchanged: supports gastronet (DINOv2 ViT-B/14 reg4), DINOv3,
SimCLR / MoCo-v2 (ResNet-50), and ImageNet ResNet-50. All checkpoint
state-dict adapters from the original model.py are preserved so legacy
Gastronet / SimCLR / MoCo weights still load.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# State-dict housekeeping
# ---------------------------------------------------------------------------
def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint).__name__}.")
    for key in ("model_state_dict", "teacher", "state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    return checkpoint


def _clean_state_dict_keys(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _normalize_classifier_head_keys(state_dict):
    """Old checkpoints used 'cls_head.' as the head prefix; rename to 'head.'."""
    if any(k.startswith("cls_head.") for k in state_dict) and not any(
        k.startswith("head.") for k in state_dict
    ):
        state_dict = {
            (k.replace("cls_head.", "head.", 1) if k.startswith("cls_head.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _infer_state_dict_backbone_prefix(state_dict):
    if (
        "backbone.conv1.weight" in state_dict
        and "backbone.layer1.0.conv1.weight" in state_dict
        and "backbone.layer4.2.conv3.weight" in state_dict
    ):
        return "backbone."
    if (
        "conv1.weight" in state_dict
        and "layer1.0.conv1.weight" in state_dict
        and "layer4.2.conv3.weight" in state_dict
    ):
        return ""
    return None


def _extract_backbone_state_dict(checkpoint):
    state_dict = _clean_state_dict_keys(_unwrap_checkpoint_state_dict(checkpoint))
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            clean_key = key.removeprefix("backbone.")
        else:
            continue
        if clean_key == "register_tokens":
            clean_key = "reg_token"
        backbone_state[clean_key] = value
    if backbone_state:
        return backbone_state
    direct = dict(state_dict)
    if "register_tokens" in direct and "reg_token" not in direct:
        direct["reg_token"] = direct.pop("register_tokens")
    return direct


def _strip_extra_prefix_position_token(pos_embed):
    token_count = pos_embed.shape[1]
    patch_only = token_count - 1
    grid = math.isqrt(patch_only)
    if grid * grid == patch_only:
        return pos_embed[:, 1:]
    return pos_embed


def _adapt_position_embeddings(backbone_state, backbone):
    if "pos_embed" not in backbone_state:
        return backbone_state
    pos_embed = backbone_state["pos_embed"]
    model_pos_embed = backbone.pos_embed
    if getattr(backbone, "no_embed_class", False):
        pos_embed = _strip_extra_prefix_position_token(pos_embed)
        num_prefix_tokens = 0
    else:
        num_prefix_tokens = getattr(backbone, "num_prefix_tokens", 1)
    if pos_embed.shape != model_pos_embed.shape:
        pos_embed = resample_abs_pos_embed(
            pos_embed,
            new_size=backbone.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
        )
    adapted = dict(backbone_state)
    adapted["pos_embed"] = pos_embed
    return adapted


def load_backbone_weights(backbone, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone_state = _adapt_position_embeddings(
        _extract_backbone_state_dict(checkpoint),
        backbone,
    )
    incompatible = backbone.load_state_dict(backbone_state, strict=False)
    allowed_unexpected = {"mask_token"}
    unexpected = sorted(set(incompatible.unexpected_keys) - allowed_unexpected)
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Backbone checkpoint is incompatible with the requested model. "
            f"Missing keys: {incompatible.missing_keys}. "
            f"Unexpected keys: {unexpected}."
        )


def extract_model_state_dict(checkpoint):
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected dict state_dict, got {type(state_dict).__name__}.")
    return _normalize_classifier_head_keys(_clean_state_dict_keys(state_dict))


# ---------------------------------------------------------------------------
# Config inference (KNN/SVM only on this branch)
# ---------------------------------------------------------------------------
def infer_backbone_input_config_from_state_dict(state_dict):
    inferred = {}
    resnet_prefix = _infer_state_dict_backbone_prefix(state_dict)
    if resnet_prefix is not None:
        conv1_weight = state_dict.get(f"{resnet_prefix}conv1.weight")
        if conv1_weight is not None and conv1_weight.ndim == 4:
            inferred["in_channels"] = int(conv1_weight.shape[1])
            inferred.setdefault("input_size", 224)
        return inferred

    patch_proj = state_dict.get("backbone.patch_embed.proj.weight")
    if patch_proj is None:
        return inferred
    inferred["in_channels"] = int(patch_proj.shape[1])

    pos_embed = state_dict.get("backbone.pos_embed")
    if pos_embed is None:
        return inferred

    token_count = int(pos_embed.shape[1])
    candidates = [0]
    cls_token = state_dict.get("backbone.cls_token")
    if cls_token is not None:
        candidates.append(int(cls_token.shape[1]))
    reg_token = state_dict.get("backbone.reg_token")
    if reg_token is not None:
        candidates.append(int(reg_token.shape[1]))
        if cls_token is not None:
            candidates.append(int(cls_token.shape[1] + reg_token.shape[1]))

    patch_grid = None
    for prefix_tokens in candidates:
        patch_tokens = token_count - prefix_tokens
        if patch_tokens <= 0:
            continue
        candidate_grid = math.isqrt(patch_tokens)
        if candidate_grid * candidate_grid == patch_tokens:
            patch_grid = candidate_grid
            break
    if patch_grid is None:
        return inferred

    patch_h = int(patch_proj.shape[-2])
    patch_w = int(patch_proj.shape[-1])
    if patch_h == patch_w:
        inferred["input_size"] = patch_grid * patch_h
    return inferred


def infer_backbone_architecture_from_state_dict(state_dict):
    if _infer_state_dict_backbone_prefix(state_dict) is not None:
        return {"backbone_name": "resnet50", "pretrained": False}
    return {}


def infer_model_config_from_state_dict(state_dict, checkpoint=None):
    """For KNN/SVM checkpoints the head_type lives in model_config, since the
    head itself isn't a tensor module. Other head types are not supported on
    this branch."""
    if checkpoint and isinstance(checkpoint, dict):
        cfg = checkpoint.get("model_config", {})
        if isinstance(cfg, dict):
            head_type = cfg.get("head_type")
            if head_type in ("knn", "svm"):
                return {
                    "head_type": head_type,
                    "n_classes": int(cfg.get("n_classes", 2)),
                }
    raise ValueError(
        "Could not infer head_type from checkpoint. Expected model_config with "
        "head_type in {'knn', 'svm'}."
    )


def resolve_model_kwargs_from_checkpoint(checkpoint, fallback_kwargs=None):
    fallback_kwargs = dict(fallback_kwargs or {})
    checkpoint_kwargs = {}
    if isinstance(checkpoint, dict):
        raw_cfg = checkpoint.get("model_config")
        if isinstance(raw_cfg, dict):
            checkpoint_kwargs.update(raw_cfg)

    state_dict = extract_model_state_dict(checkpoint)
    arch_kwargs = infer_backbone_architecture_from_state_dict(state_dict)
    input_kwargs = infer_backbone_input_config_from_state_dict(state_dict)
    head_kwargs = infer_model_config_from_state_dict(state_dict, checkpoint=checkpoint)

    resolved = dict(fallback_kwargs)
    resolved.update(arch_kwargs)
    resolved.update(input_kwargs)
    resolved.update(head_kwargs)
    resolved.update(checkpoint_kwargs)
    resolved.setdefault("classifier_input", "pooled")
    return resolved


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------
def create_model_checkpoint(model, model_config, extra_metadata=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    if model.is_sklearn_head and model.head._fitted:
        payload["sklearn_head_state"] = pickle.dumps(model.head._clf)
    return payload


def _adapt_encoder_proj_head_state_dict(proj_head_state, model_proj_head_state):
    cleaned = _clean_state_dict_keys(proj_head_state)
    expected = set(model_proj_head_state.keys())
    if set(cleaned.keys()) == expected:
        return cleaned, False

    adapted = dict(model_proj_head_state)
    used = set()
    for target_key in ("0.weight", "0.bias", "2.weight", "2.bias"):
        target_tensor = model_proj_head_state[target_key]
        if target_key in cleaned and cleaned[target_key].shape == target_tensor.shape:
            adapted[target_key] = cleaned[target_key]
            used.add(target_key)
    # First-layer weight fallback (any 2D tensor with the right shape)
    if "0.weight" not in used:
        for k, t in cleaned.items():
            if t.ndim == 2 and t.shape == model_proj_head_state["0.weight"].shape:
                adapted["0.weight"] = t
                used.add(k)
                break
    # Last-layer weight fallback
    if "2.weight" not in used:
        candidates = [
            k for k, t in cleaned.items()
            if t.ndim == 2 and t.shape == model_proj_head_state["2.weight"].shape and k not in used
        ]
        if candidates:
            adapted["2.weight"] = cleaned[candidates[-1]]
            used.add(candidates[-1])
    for bias_key in ("0.bias", "2.bias"):
        if bias_key in used:
            continue
        for k, t in cleaned.items():
            if (
                t.ndim == 1
                and t.shape == model_proj_head_state[bias_key].shape
                and ".running_" not in k
                and not k.endswith("num_batches_tracked")
                and k not in used
            ):
                adapted[bias_key] = t
                used.add(k)
                break
    ok = all(
        adapted[k].shape == model_proj_head_state[k].shape
        for k in ("0.weight", "0.bias", "2.weight", "2.bias")
    )
    return adapted, ok


def load_encoder_checkpoint(model, checkpoint_path, strict=True):
    """Loads `backbone` + `proj_head` from a pretrain encoder.pt. Used by the
    finetune stage to warm-start before fitting the sklearn head."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict encoder checkpoint, got {type(checkpoint).__name__}.")

    backbone_state = checkpoint.get("backbone")
    proj_head_state = checkpoint.get("proj_head")
    if not isinstance(backbone_state, dict) or not isinstance(proj_head_state, dict):
        raise ValueError("Encoder checkpoint must contain 'backbone' and 'proj_head' state dicts.")

    backbone_incompat = model.backbone.load_state_dict(
        _clean_state_dict_keys(backbone_state), strict=strict,
    )
    model_proj_state = model.proj_head.state_dict()
    adapted_proj_state, adapted = _adapt_encoder_proj_head_state_dict(proj_head_state, model_proj_state)
    proj_incompat = model.proj_head.load_state_dict(adapted_proj_state, strict=strict)

    if strict:
        if backbone_incompat.missing_keys or backbone_incompat.unexpected_keys:
            raise RuntimeError(
                f"Backbone weights incompatible. Missing: {backbone_incompat.missing_keys}. "
                f"Unexpected: {backbone_incompat.unexpected_keys}."
            )
        if proj_incompat.missing_keys or proj_incompat.unexpected_keys:
            raise RuntimeError(
                f"Projection head weights incompatible. Missing: {proj_incompat.missing_keys}. "
                f"Unexpected: {proj_incompat.unexpected_keys}."
            )
        if adapted:
            print("[INFO] Adapted legacy encoder projection-head weights.")


def load_model_checkpoint(model, checkpoint_path, strict=True, map_location="cpu"):
    """Load a full (backbone + proj_head + head + fitted sklearn) checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("backbone"), dict)
        and isinstance(checkpoint.get("proj_head"), dict)
        and "model_state_dict" not in checkpoint
    ):
        raise ValueError(
            "Checkpoint only contains encoder weights. Use a finetuned checkpoint."
        )

    state_dict = extract_model_state_dict(checkpoint)
    incompat = model.load_state_dict(state_dict, strict=strict)
    if strict and (incompat.missing_keys or incompat.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint incompatible. Missing: {incompat.missing_keys}. "
            f"Unexpected: {incompat.unexpected_keys}."
        )

    if model.is_sklearn_head and "sklearn_head_state" in checkpoint:
        model.head._clf = pickle.loads(checkpoint["sklearn_head_state"])
        model.head._fitted = True
    return checkpoint, incompat


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
class SklearnKNNHead(nn.Module):
    """k-NN classifier wrapped as an nn.Module. Returns log-probabilities so
    softmax(log_p) == p downstream."""

    def __init__(self, n_neighbors=5, n_classes=2):
        super().__init__()
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for the KNN head.")
        self.n_neighbors = int(n_neighbors)
        self.n_classes = int(n_classes)
        self._clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self._fitted = False
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit(self, X, y):
        self._clf.fit(X, y)
        self._fitted = True

    def forward(self, features):
        if not self._fitted:
            raise RuntimeError("KNN head not fitted. Call fit() first.")
        X = features.detach().cpu().numpy()
        proba = self._clf.predict_proba(X).astype(np.float32)
        log_proba = np.log(np.clip(proba, 1e-8, 1.0))
        return torch.from_numpy(log_proba).to(device=features.device)


class SklearnSVMHead(nn.Module):
    """RBF-kernel SVM with Platt-scaled probabilities."""

    def __init__(self, C=2.0, n_classes=2):
        super().__init__()
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for the SVM head.")
        self.C = float(C)
        self.n_classes = int(n_classes)
        self._clf = SVC(C=self.C, kernel="rbf", probability=True)
        self._fitted = False
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit(self, X, y):
        self._clf.fit(X, y)
        self._fitted = True

    def forward(self, features):
        if not self._fitted:
            raise RuntimeError("SVM head not fitted. Call fit() first.")
        X = features.detach().cpu().numpy()
        proba = self._clf.predict_proba(X).astype(np.float32)
        log_proba = np.log(np.clip(proba, 1e-8, 1.0))
        return torch.from_numpy(log_proba).to(device=features.device)


def _build_classifier_head(in_features, n_classes, head_type, knn_neighbors=5, svm_C=2.0):
    if head_type == "knn":
        k = int(knn_neighbors)
        return SklearnKNNHead(n_neighbors=k, n_classes=n_classes), f"KNN (k={k})"
    if head_type == "svm":
        c = float(svm_C)
        return SklearnSVMHead(C=c, n_classes=n_classes), f"SVM RBF (C={c})"
    raise ValueError(
        f"Unsupported head_type '{head_type}' on this branch. Allowed: 'knn', 'svm'."
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """Backbone + projection MLP (for SupPro) + sklearn classifier head."""

    def __init__(
        self,
        in_channels=3,
        n_classes=2,
        backbone_name="vit_base_patch14_reg4_dinov2",
        backbone_weights_path=None,
        input_size=336,
        freeze_backbone=False,
        pretrained=None,
        proj_dim=128,
        classifier_input="pooled",
        head_type="knn",
        knn_neighbors=5,
        svm_C=2.0,
        **kwargs,  # forwarded to timm.create_model
    ):
        super().__init__()
        if pretrained is None:
            pretrained = backbone_weights_path is None

        backbone_kwargs = dict(
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            **kwargs,
        )
        try:
            self.backbone = timm.create_model(backbone_name, img_size=input_size, **backbone_kwargs)
        except TypeError as exc:
            if "img_size" not in str(exc):
                raise
            self.backbone = timm.create_model(backbone_name, **backbone_kwargs)

        if backbone_weights_path is not None:
            load_backbone_weights(self.backbone, backbone_weights_path)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError(f"Could not infer num_features for backbone={backbone_name}")

        self.feat_dim = feat_dim
        self.feature_dim = feat_dim
        if classifier_input not in {"pooled", "projection"}:
            raise ValueError(f"classifier_input must be 'pooled' or 'projection', got {classifier_input!r}.")
        self.classifier_input = classifier_input
        self.proj_dim = int(proj_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )

        self.head_type = head_type
        head_in_features = self.proj_dim if self.classifier_input == "projection" else feat_dim
        self.head, classifier_description = _build_classifier_head(
            head_in_features, n_classes,
            head_type=head_type,
            knn_neighbors=knn_neighbors,
            svm_C=svm_C,
        )
        self.classifier_description = f"{classifier_description} on {self.classifier_input} features"

    @property
    def cls_head(self):
        return self.head

    @property
    def is_sklearn_head(self):
        return isinstance(self.head, (SklearnKNNHead, SklearnSVMHead))

    def forward_tokens(self, x):
        return self.backbone.forward_features(x)

    def pooled_features_from_tokens(self, tokens):
        return self.backbone.forward_head(tokens, pre_logits=True)

    def encode(self, x):
        return self.pooled_features_from_tokens(self.forward_tokens(x))

    def project(self, feat):
        return F.normalize(self.proj_head(feat), dim=-1)

    def classifier_features_from_pooled(self, feat):
        if self.classifier_input == "projection":
            return self.project(feat)
        return feat

    def classify(self, classifier_features):
        return self.head(classifier_features)

    def forward(self, x, return_embedding=False):
        tokens = self.forward_tokens(x)
        feat = self.pooled_features_from_tokens(tokens)
        classifier_features = self.classifier_features_from_pooled(feat)
        logits = self.classify(classifier_features)
        if not return_embedding:
            return logits
        return {
            "logits": logits,
            "embedding": self.project(feat),
            "classifier_features": classifier_features,
        }
