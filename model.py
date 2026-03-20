import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, image_size=224, hidden_dim=32, **kwargs):
        super().__init__()
        input_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        pos_logit = self.net(x)
        return torch.cat([-pos_logit, pos_logit], dim=1)

