"""Torchvision model wrappers with optional pretrained weights."""

from typing import Optional

import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V2_Weights,
    ViT_B_16_Weights,
    mobilenet_v2,
    resnet50,
    vit_b_16,
)

__all__ = [
    "ResNet50Model",
    "MobileNetV2Model",
    "ViTB16Model",
]


class ResNet50Model(nn.Module):
    """ResNet-50 wrapper that optionally loads pretrained weights."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        weights: Optional[ResNet50_Weights] = None,
    ) -> None:
        super().__init__()
        if pretrained and weights is None:
            weights = ResNet50_Weights.DEFAULT
        if not pretrained:
            weights = None
        self.model = resnet50(weights=weights)
        if num_classes != self.model.fc.out_features:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV2Model(nn.Module):
    """MobileNetV2 wrapper that optionally loads pretrained weights."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        weights: Optional[MobileNet_V2_Weights] = None,
    ) -> None:
        super().__init__()
        if pretrained and weights is None:
            weights = MobileNet_V2_Weights.DEFAULT
        if not pretrained:
            weights = None
        self.model = mobilenet_v2(weights=weights)
        last_linear = self.model.classifier[-1]
        if num_classes != last_linear.out_features:
            in_features = last_linear.in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ViTB16Model(nn.Module):
    """Vision Transformer (ViT-B/16) wrapper that optionally loads pretrained weights."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        weights: Optional[ViT_B_16_Weights] = None,
    ) -> None:
        super().__init__()
        if pretrained and weights is None:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        if not pretrained:
            weights = None
        self.model = vit_b_16(weights=weights)

        heads_module = self.model.heads
        # Torchvision exposes the final head either as .head or as the last child module.
        if hasattr(heads_module, "head") and isinstance(heads_module.head, nn.Linear):
            last_linear = heads_module.head
            if num_classes != last_linear.out_features:
                in_features = last_linear.in_features
                heads_module.head = nn.Linear(in_features, num_classes)
        else:
            children = list(heads_module.children())
            if children and isinstance(children[-1], nn.Linear):
                last_linear = children[-1]
                if num_classes != last_linear.out_features:
                    in_features = last_linear.in_features
                    children[-1] = nn.Linear(in_features, num_classes)
                    self.model.heads = nn.Sequential(*children)

    def forward(self, x):
        return self.model(x)
