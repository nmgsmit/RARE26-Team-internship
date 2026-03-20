import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, backbone_name='vit_base_patch16_dinov3.lvd1689m', **kwargs):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            in_chans=in_channels,
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        backbone_out = self.backbone.num_features
        self.head = nn.Linear(backbone_out, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

