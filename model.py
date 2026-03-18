import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, backbone_name='vit_base_patch16_dinov3', pretrained=True, class_prior=None, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove head
            in_chans=in_channels
        )
        backbone_out = self.backbone.num_features if hasattr(self.backbone, 'num_features') else self.backbone.head.in_features
        self.head = nn.Linear(backbone_out, n_classes)

        # Optional: Add a bias for class imbalance (logit adjustment)
        if class_prior is not None:
            # class_prior: list or np.array of class probabilities, e.g. [0.9, 0.1]
            prior = torch.log(torch.tensor(class_prior, dtype=torch.float32))
            self.head.bias = nn.Parameter(self.head.bias.data + prior)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

