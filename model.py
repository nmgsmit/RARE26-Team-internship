import torch
import torch.nn as nn


class Model(nn.Module):
    """Basic CNN image classifier baseline."""
    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        backbone_name="vit_base_patch16_dinov3",
        pretrained=True,
    ):
        """Build a simple CNN classifier.

        Args:
            in_channels (int): Number of image input channels.
            n_classes (int): Number of target classes.
            backbone_name (str): Unused, kept for train-script compatibility.
            pretrained (bool): Unused, kept for train-script compatibility.
        """
        super().__init__()

        self.in_channels = in_channels
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """Forward pass returning classification logits of shape (B, n_classes)."""
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        features = self.features(x)
        return self.classifier(features)