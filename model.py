import math
from pathlib import Path

import torch
import torch.nn as nn
import timm
from timm.layers import resample_abs_pos_embed


def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint at load time to be a dict, got {type(checkpoint).__name__}."
        )

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

    if not backbone_state:
        raise ValueError(
            "Could not find backbone weights in the supplied checkpoint. "
            "Expected keys with a 'backbone.' prefix or a nested teacher/state_dict/model entry."
        )
    return backbone_state


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

    backbone_state = dict(backbone_state)
    backbone_state["pos_embed"] = pos_embed
    return backbone_state


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

class Model(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=2,
        backbone_name='vit_base_patch16_dinov3.lvd1689m',
        backbone_weights_path=None,
        input_size=224,
        freeze_backbone=True,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()
        if pretrained is None:
            pretrained = backbone_weights_path is None

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            img_size=input_size,
            **kwargs,
        )

        if backbone_weights_path is not None:
            load_backbone_weights(self.backbone, backbone_weights_path)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        backbone_out = self.backbone.num_features
        self.head = nn.Linear(backbone_out, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

