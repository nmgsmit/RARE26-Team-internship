import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    """DINOv3 image classifier with a simple linear classification head."""
    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        backbone_name="vit_base_patch16_dinov3",
        pretrained=True,
    ):
        """Build a DINOv3 backbone and a linear classifier.

        Args:
            in_channels (int): Number of image input channels.
            n_classes (int): Number of target classes.
            backbone_name (str): Timm model name for the DINOv3 backbone.
            pretrained (bool): Whether to load pretrained backbone weights.
        """
        super().__init__()

        self.in_channels = in_channels
# Transfer learning: set DINOv3 backbone, put num_classes to 0 to remove output layer
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
        )
        self.classifier = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        """Forward pass returning classification logits of shape (B, n_classes)."""
# Check if input channel is RGB
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
# Get image features using the DINOv3 Backbone
        features = self.backbone(x)
# Get correct output of features
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features[:, 0]
    
# Put a binary classifier over the features
        return self.classifier(features)