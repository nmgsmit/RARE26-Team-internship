import timm
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Dino backbone + projection head (for TTC/SupCon) + classifier head.
    """
    def __init__(
        self,
        n_classes=2,
        backbone_name="vit_base_patch16_dinov3",
        pretrained=True,
        proj_dim=128,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError(f"Could not infer num_features for backbone={backbone_name}")

        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        self.cls_head = nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_embedding=False):
        feat = self.backbone(x)                
        logits = self.cls_head(feat)           

        if not return_embedding:
            return logits

        embedding = self.proj_head(feat)       
        embedding = F.normalize(embedding, dim=-1)
        return {"logits": logits, "embedding": embedding}