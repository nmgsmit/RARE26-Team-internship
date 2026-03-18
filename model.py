import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    """Image classifier wrapper around a timm backbone."""

    def __init__(
        self,
        in_channels=3,
        n_classes=2,
        backbone_name="vit_base_patch16_dinov3",
        pretrained=True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=n_classes,
        )

    def forward(self, x):
        return self.backbone(x)

