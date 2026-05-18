import math
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


def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint at load time to be a dict, got {type(checkpoint).__name__}."
        )

    for key in ("model_state_dict", "teacher", "state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    return checkpoint


def _clean_state_dict_keys(state_dict):
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _normalize_classifier_head_keys(state_dict):
    if any(key.startswith("cls_head.") for key in state_dict) and not any(
        key.startswith("head.") for key in state_dict
    ):
        state_dict = {
            (key.replace("cls_head.", "head.", 1) if key.startswith("cls_head.") else key): value
            for key, value in state_dict.items()
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

    direct_state = dict(state_dict)
    if "register_tokens" in direct_state and "reg_token" not in direct_state:
        direct_state["reg_token"] = direct_state.pop("register_tokens")
    return direct_state


def _strip_extra_prefix_position_token(pos_embed):
    token_count = pos_embed.shape[1]
    patch_only_tokens = token_count - 1
    patch_grid = math.isqrt(patch_only_tokens)
    if patch_grid * patch_grid == patch_only_tokens:
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

    adapted_state = dict(backbone_state)
    adapted_state["pos_embed"] = pos_embed
    return adapted_state


def load_backbone_weights(backbone, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone_state = _extract_backbone_state_dict(checkpoint)
    backbone_state = _adapt_position_embeddings(backbone_state, backbone)

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
        raise TypeError(
            f"Expected checkpoint state_dict to be a dict, got {type(state_dict).__name__}."
        )
    state_dict = _clean_state_dict_keys(state_dict)
    return _normalize_classifier_head_keys(state_dict)


def _detect_fullwidth_mlp_config(state_dict):
    linear_layer_indices = []
    for key in state_dict:
        if not key.startswith("head.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "head" or parts[2] != "weight":
            continue
        if parts[1].isdigit():
            linear_layer_indices.append(int(parts[1]))

    if not linear_layer_indices:
        return None

    linear_layer_indices = sorted(linear_layer_indices)
    first_linear_weight = state_dict[f"head.{linear_layer_indices[0]}.weight"]
    last_linear_weight = state_dict[f"head.{linear_layer_indices[-1]}.weight"]
    return {
        "head_type": "mlp_fullwidth",
        "mlp_hidden_layers": len(linear_layer_indices) - 1,
        "mlp_hidden_dim": int(first_linear_weight.shape[0]),
        "classifier_input_dim": int(first_linear_weight.shape[1]),
        "mlp_dropout": 0.0,
        "n_classes": int(last_linear_weight.shape[0]),
    }


def _infer_classifier_input_from_state_dict(state_dict, inferred_config):
    proj_out_weight = state_dict.get("proj_head.2.weight")
    if proj_out_weight is None:
        return "pooled"

    proj_dim = int(proj_out_weight.shape[0])
    classifier_input_dim = inferred_config.get("classifier_input_dim")
    if classifier_input_dim is None:
        return "pooled"
    return "projection" if int(classifier_input_dim) == proj_dim else "pooled"


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
    prefix_token_candidates = [0]

    cls_token = state_dict.get("backbone.cls_token")
    if cls_token is not None:
        prefix_token_candidates.append(int(cls_token.shape[1]))

    reg_token = state_dict.get("backbone.reg_token")
    if reg_token is not None:
        prefix_token_candidates.append(int(reg_token.shape[1]))
        if cls_token is not None:
            prefix_token_candidates.append(int(cls_token.shape[1] + reg_token.shape[1]))

    patch_grid = None
    for prefix_tokens in prefix_token_candidates:
        patch_token_count = token_count - prefix_tokens
        if patch_token_count <= 0:
            continue
        candidate_grid = math.isqrt(patch_token_count)
        if candidate_grid * candidate_grid == patch_token_count:
            patch_grid = candidate_grid
            break

    if patch_grid is None:
        return inferred

    patch_height = int(patch_proj.shape[-2])
    patch_width = int(patch_proj.shape[-1])
    if patch_height == patch_width:
        inferred["input_size"] = patch_grid * patch_height

    return inferred


def infer_backbone_architecture_from_state_dict(state_dict):
    if _infer_state_dict_backbone_prefix(state_dict) is not None:
        return {
            "backbone_name": "resnet50",
            "pretrained": False,
        }
    return {}


def infer_model_config_from_state_dict(state_dict):
    state_keys = set(state_dict.keys())

    if "head.logit_scale" in state_keys:
        return {
            "head_type": "cosine_linear",
            "classifier_input_dim": int(state_dict["head.weight"].shape[1]),
            "n_classes": int(state_dict["head.weight"].shape[0]),
        }

    if "head.input_norm.weight" in state_keys and "head.classifier.weight" in state_keys:
        return {
            "head_type": "residual_bottleneck",
            "classifier_input_dim": int(state_dict["head.up_proj.weight"].shape[1]),
            "head_hidden_dim": int(state_dict["head.up_proj.weight"].shape[0]),
            "head_dropout": 0.1,
            "n_classes": int(state_dict["head.classifier.weight"].shape[0]),
        }

    if "head.net.1.weight" in state_keys and "head.net.4.weight" in state_keys:
        return {
            "head_type": "mlp_bottleneck",
            "classifier_input_dim": int(state_dict["head.net.1.weight"].shape[1]),
            "head_hidden_dim": int(state_dict["head.net.1.weight"].shape[0]),
            "head_dropout": 0.0,
            "n_classes": int(state_dict["head.net.4.weight"].shape[0]),
        }

    if "head.norm.weight" in state_keys and "head.classifier.weight" in state_keys:
        return {
            "head_type": "ln_linear",
            "classifier_input_dim": int(state_dict["head.classifier.weight"].shape[1]),
            "n_classes": int(state_dict["head.classifier.weight"].shape[0]),
        }

    fullwidth_mlp_config = _detect_fullwidth_mlp_config(state_dict)
    if fullwidth_mlp_config is not None:
        return fullwidth_mlp_config

    if "head.weight" in state_keys and "head.bias" in state_keys:
        return {
            "head_type": "linear",
            "classifier_input_dim": int(state_dict["head.weight"].shape[1]),
            "n_classes": int(state_dict["head.weight"].shape[0]),
        }

    raise ValueError(
        "Could not infer classifier head config from checkpoint state_dict. "
        "Expected a supported head layout under the 'head.' prefix."
    )


def resolve_model_kwargs_from_checkpoint(checkpoint, fallback_kwargs=None):
    fallback_kwargs = dict(fallback_kwargs or {})
    checkpoint_kwargs = {}
    if isinstance(checkpoint, dict):
        raw_model_config = checkpoint.get("model_config")
        if isinstance(raw_model_config, dict):
            checkpoint_kwargs.update(raw_model_config)

    state_dict = extract_model_state_dict(checkpoint)
    inferred_backbone_arch_kwargs = infer_backbone_architecture_from_state_dict(state_dict)
    inferred_backbone_kwargs = infer_backbone_input_config_from_state_dict(state_dict)
    inferred_kwargs = infer_model_config_from_state_dict(state_dict)
    inferred_kwargs["classifier_input"] = _infer_classifier_input_from_state_dict(
        state_dict, inferred_kwargs
    )
    inferred_kwargs.pop("classifier_input_dim", None)

    resolved_kwargs = dict(fallback_kwargs)
    resolved_kwargs.update(inferred_backbone_arch_kwargs)
    resolved_kwargs.update(inferred_backbone_kwargs)
    resolved_kwargs.update(inferred_kwargs)
    resolved_kwargs.update(checkpoint_kwargs)
    return resolved_kwargs


def create_model_checkpoint(model, model_config, extra_metadata=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def _adapt_encoder_proj_head_state_dict(proj_head_state, model_proj_head_state):
    cleaned_state = _clean_state_dict_keys(proj_head_state)

    expected_keys = set(model_proj_head_state.keys())
    if set(cleaned_state.keys()) == expected_keys:
        return cleaned_state, False

    adapted_state = dict(model_proj_head_state)
    used_source_keys = set()

    for target_key in ("0.weight", "0.bias", "2.weight", "2.bias"):
        target_tensor = model_proj_head_state[target_key]
        if target_key in cleaned_state and cleaned_state[target_key].shape == target_tensor.shape:
            adapted_state[target_key] = cleaned_state[target_key]
            used_source_keys.add(target_key)

    if "0.weight" not in used_source_keys:
        for source_key, source_tensor in cleaned_state.items():
            if (
                source_tensor.ndim == 2
                and source_tensor.shape == model_proj_head_state["0.weight"].shape
            ):
                adapted_state["0.weight"] = source_tensor
                used_source_keys.add(source_key)
                break

    if "2.weight" not in used_source_keys:
        matching_linear_keys = [
            source_key
            for source_key, source_tensor in cleaned_state.items()
            if (
                source_tensor.ndim == 2
                and source_tensor.shape == model_proj_head_state["2.weight"].shape
                and source_key not in used_source_keys
            )
        ]
        if matching_linear_keys:
            adapted_state["2.weight"] = cleaned_state[matching_linear_keys[-1]]
            used_source_keys.add(matching_linear_keys[-1])

    if "0.bias" not in used_source_keys:
        for source_key, source_tensor in cleaned_state.items():
            if (
                source_tensor.ndim == 1
                and source_tensor.shape == model_proj_head_state["0.bias"].shape
                and ".running_" not in source_key
                and not source_key.endswith("num_batches_tracked")
                and source_key not in used_source_keys
            ):
                adapted_state["0.bias"] = source_tensor
                used_source_keys.add(source_key)
                break

    if "2.bias" not in used_source_keys:
        for source_key, source_tensor in cleaned_state.items():
            if (
                source_tensor.ndim == 1
                and source_tensor.shape == model_proj_head_state["2.bias"].shape
                and ".running_" not in source_key
                and not source_key.endswith("num_batches_tracked")
                and source_key not in used_source_keys
            ):
                adapted_state["2.bias"] = source_tensor
                used_source_keys.add(source_key)
                break

    successfully_adapted = (
        adapted_state["0.weight"].shape == model_proj_head_state["0.weight"].shape
        and adapted_state["0.bias"].shape == model_proj_head_state["0.bias"].shape
        and adapted_state["2.weight"].shape == model_proj_head_state["2.weight"].shape
        and adapted_state["2.bias"].shape == model_proj_head_state["2.bias"].shape
    )
    return adapted_state, successfully_adapted


def load_encoder_checkpoint(model, checkpoint_path, strict=True):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected encoder checkpoint to be a dict, got {type(checkpoint).__name__}."
        )

    backbone_state = checkpoint.get("backbone")
    proj_head_state = checkpoint.get("proj_head")
    if not isinstance(backbone_state, dict) or not isinstance(proj_head_state, dict):
        raise ValueError(
            "Encoder checkpoint must contain 'backbone' and 'proj_head' state dicts."
        )

    backbone_incompatible = model.backbone.load_state_dict(
        _clean_state_dict_keys(backbone_state),
        strict=strict,
    )
    model_proj_head_state = model.proj_head.state_dict()
    adapted_proj_head_state, proj_head_adapted = _adapt_encoder_proj_head_state_dict(
        proj_head_state,
        model_proj_head_state,
    )
    proj_incompatible = model.proj_head.load_state_dict(
        adapted_proj_head_state,
        strict=strict,
    )

    if strict:
        if backbone_incompatible.missing_keys or backbone_incompatible.unexpected_keys:
            raise RuntimeError(
                "Backbone weights in the encoder checkpoint are incompatible with the "
                f"requested model. Missing keys: {backbone_incompatible.missing_keys}. "
                f"Unexpected keys: {backbone_incompatible.unexpected_keys}."
            )
        if proj_incompatible.missing_keys or proj_incompatible.unexpected_keys:
            raise RuntimeError(
                "Projection head weights in the encoder checkpoint are incompatible with "
                f"the requested model. Missing keys: {proj_incompatible.missing_keys}. "
                f"Unexpected keys: {proj_incompatible.unexpected_keys}."
            )
        if proj_head_adapted:
            print(
                "[INFO] Adapted legacy encoder projection-head weights to the current "
                "projection-head architecture."
            )


def load_model_checkpoint(model, checkpoint_path, strict=True, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("backbone"), dict)
        and isinstance(checkpoint.get("proj_head"), dict)
        and "model_state_dict" not in checkpoint
        and "state_dict" not in checkpoint
        and "model" not in checkpoint
        and "teacher" not in checkpoint
    ):
        raise ValueError(
            "The supplied checkpoint only contains encoder weights. "
            "Use a baseline/finetuned checkpoint for full-model loading or Grad-CAM evaluation."
        )

    state_dict = extract_model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "Checkpoint is incompatible with the requested model architecture. "
            f"Missing keys: {incompatible.missing_keys}. "
            f"Unexpected keys: {incompatible.unexpected_keys}."
        )
    return checkpoint, incompatible


class LayerNormLinearHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.classifier(self.norm(x))


class MLPBottleneckHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBottleneckHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.up_proj = nn.Linear(in_features, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.down_proj = nn.Linear(hidden_dim, in_features)
        self.output_norm = nn.LayerNorm(in_features)
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        x = residual + x
        x = self.output_norm(x)
        return self.classifier(x)


class CosineLinearHead(nn.Module):
    def __init__(self, in_features, n_classes, init_scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        normalized_features = F.normalize(x, dim=-1)
        normalized_weight = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp()
        return scale * normalized_features @ normalized_weight.t()


class SklearnKNNHead(nn.Module):
    """k-Nearest-Neighbours classifier wrapped as an nn.Module.

    forward() returns log-probabilities so that torch.softmax() and
    torch.argmax() behave consistently with the rest of the training loop.
    Call fit(X_numpy, y_numpy) once before using forward().
    """

    def __init__(self, n_neighbors=5, n_classes=2):
        super().__init__()
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for the KNN head.  "
                "Install it with:  pip install scikit-learn"
            )
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes
        self._clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._fitted = False
        # Placeholder so state_dict() / module.parameters() work normally.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit(self, X, y):
        self._clf.fit(X, y)
        self._fitted = True

    def forward(self, features):
        if not self._fitted:
            raise RuntimeError("KNN head has not been fitted yet. Call fit() first.")
        X = features.detach().cpu().numpy()
        proba = self._clf.predict_proba(X).astype(np.float32)
        # softmax(log(p)) == p when p sums to 1, so all downstream code works unchanged.
        log_proba = np.log(np.clip(proba, 1e-8, 1.0))
        return torch.from_numpy(log_proba).to(device=features.device)


class SklearnSVMHead(nn.Module):
    """SVM classifier (RBF kernel, C=margin) wrapped as an nn.Module.

    forward() returns log-probabilities (Platt-scaled via probability=True).
    Call fit(X_numpy, y_numpy) once before using forward().
    """

    def __init__(self, C=2.0, n_classes=2):
        super().__init__()
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for the SVM head.  "
                "Install it with:  pip install scikit-learn"
            )
        self.C = C
        self.n_classes = n_classes
        self._clf = SVC(C=C, kernel="rbf", probability=True)
        self._fitted = False
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit(self, X, y):
        self._clf.fit(X, y)
        self._fitted = True

    def forward(self, features):
        if not self._fitted:
            raise RuntimeError("SVM head has not been fitted yet. Call fit() first.")
        X = features.detach().cpu().numpy()
        proba = self._clf.predict_proba(X).astype(np.float32)
        log_proba = np.log(np.clip(proba, 1e-8, 1.0))
        return torch.from_numpy(log_proba).to(device=features.device)


def _build_fullwidth_mlp_head(
    in_features,
    n_classes,
    mlp_hidden_layers=1,
    mlp_hidden_dim=None,
    mlp_dropout=0.0,
):
    if mlp_hidden_layers < 0:
        raise ValueError(f"mlp_hidden_layers must be >= 0, got {mlp_hidden_layers}.")
    if not 0.0 <= mlp_dropout < 1.0:
        raise ValueError(f"mlp_dropout must be in [0, 1), got {mlp_dropout}.")

    if mlp_hidden_layers == 0:
        return nn.Linear(in_features, n_classes), None

    hidden_dim = mlp_hidden_dim or in_features
    if hidden_dim <= 0:
        raise ValueError(f"mlp_hidden_dim must be > 0, got {hidden_dim}.")

    layers = []
    current_dim = in_features
    for _ in range(mlp_hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.GELU())
        if mlp_dropout > 0.0:
            layers.append(nn.Dropout(mlp_dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, n_classes))
    return nn.Sequential(*layers), hidden_dim


def _build_classifier_head(
    in_features,
    n_classes,
    head_type="mlp_fullwidth",
    head_hidden_dim=None,
    head_dropout=0.0,
    mlp_hidden_layers=1,
    mlp_hidden_dim=None,
    mlp_dropout=0.0,
    knn_neighbors=5,
    svm_C=2.0,
):
    if head_hidden_dim is not None and head_hidden_dim <= 0:
        raise ValueError(f"head_hidden_dim must be > 0, got {head_hidden_dim}.")
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError(f"head_dropout must be in [0, 1), got {head_dropout}.")

    if head_type == "linear":
        return nn.Linear(in_features, n_classes), "linear probe", None
    if head_type == "ln_linear":
        return LayerNormLinearHead(in_features, n_classes), "LayerNorm + Linear", None
    if head_type == "mlp_fullwidth":
        head, resolved_hidden_dim = _build_fullwidth_mlp_head(
            in_features,
            n_classes,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
        )
        if mlp_hidden_layers == 0:
            description = "linear probe"
        else:
            description = (
                f"full-width MLP with {mlp_hidden_layers} hidden layer(s) "
                f"of width {resolved_hidden_dim}"
            )
        return head, description, resolved_hidden_dim
    if head_type == "mlp_bottleneck":
        hidden_dim = head_hidden_dim or 128
        head = MLPBottleneckHead(in_features, hidden_dim, n_classes, dropout=head_dropout)
        return head, f"bottleneck MLP ({hidden_dim})", hidden_dim
    if head_type == "residual_bottleneck":
        hidden_dim = head_hidden_dim or 128
        head = ResidualBottleneckHead(in_features, hidden_dim, n_classes, dropout=head_dropout)
        return head, f"residual bottleneck MLP ({hidden_dim}, dropout={head_dropout})", hidden_dim
    if head_type == "cosine_linear":
        return CosineLinearHead(in_features, n_classes), "cosine linear head", None
    if head_type == "knn":
        k = int(knn_neighbors)
        head = SklearnKNNHead(n_neighbors=k, n_classes=n_classes)
        return head, f"KNN (k={k})", None
    if head_type == "svm":
        c = float(svm_C)
        head = SklearnSVMHead(C=c, n_classes=n_classes)
        return head, f"SVM RBF (C={c})", None

    raise ValueError(f"Unsupported head_type '{head_type}'.")


class Model(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=2,
        backbone_name="vit_base_patch16_dinov3.lvd1689m",
        backbone_weights_path=None,
        input_size=224,
        freeze_backbone=False,
        pretrained=None,
        proj_dim=128,
        classifier_input="pooled",
        head_type="mlp_fullwidth",
        head_hidden_dim=None,
        head_dropout=0.0,
        mlp_hidden_layers=1,
        mlp_hidden_dim=None,
        mlp_dropout=0.0,
        knn_neighbors=5,
        svm_C=2.0,
        **kwargs,
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
            self.backbone = timm.create_model(
                backbone_name,
                img_size=input_size,
                **backbone_kwargs,
            )
        except TypeError as exc:
            if "img_size" not in str(exc):
                raise
            self.backbone = timm.create_model(
                backbone_name,
                **backbone_kwargs,
            )

        if backbone_weights_path is not None:
            load_backbone_weights(self.backbone, backbone_weights_path)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError(f"Could not infer num_features for backbone={backbone_name}")

        self.feat_dim = feat_dim
        self.feature_dim = feat_dim
        if classifier_input not in {"pooled", "projection"}:
            raise ValueError(
                f"classifier_input must be 'pooled' or 'projection', got {classifier_input!r}."
            )
        self.classifier_input = classifier_input
        self.proj_dim = int(proj_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        self.head_type = head_type
        self.classifier_hidden_layers = mlp_hidden_layers
        head_in_features = self.proj_dim if self.classifier_input == "projection" else feat_dim
        self.head, classifier_description, resolved_hidden_dim = _build_classifier_head(
            head_in_features,
            n_classes,
            head_type=head_type,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
            knn_neighbors=knn_neighbors,
            svm_C=svm_C,
        )
        self.classifier_hidden_dim = resolved_hidden_dim
        self.classifier_description = (
            f"{classifier_description} on {self.classifier_input} features"
        )

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
        tokens = self.forward_tokens(x)
        return self.pooled_features_from_tokens(tokens)

    def project(self, feat):
        return F.normalize(self.proj_head(feat), dim=-1)

    def classifier_features_from_pooled(self, feat):
        if self.classifier_input == "projection":
            return self.project(feat)
        return feat

    def classify(self, classifier_features):
        return self.head(classifier_features)

    def logits_from_pooled_features(self, feat):
        classifier_features = self.classifier_features_from_pooled(feat)
        return self.classify(classifier_features)

    def forward_from_tokens(self, tokens, return_embedding=False):
        feat = self.pooled_features_from_tokens(tokens)
        classifier_features = self.classifier_features_from_pooled(feat)
        logits = self.classify(classifier_features)

        if not return_embedding:
            return logits

        embedding = self.project(feat)
        return {
            "logits": logits,
            "embedding": embedding,
            "classifier_features": classifier_features,
        }

    def forward(self, x, return_embedding=False):
        tokens = self.forward_tokens(x)
        return self.forward_from_tokens(tokens, return_embedding=return_embedding)
