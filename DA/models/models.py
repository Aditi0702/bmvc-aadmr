"""Factory helpers for common torchvision backbones with optional pretrained weights."""

import argparse
from typing import Optional, Sequence

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    mobilenet_v2,
    resnet50,
    vit_b_16,
)

__all__ = [
    "create_pretrained_model",
    "add_model_args",
    "model_kwargs_from_args",
]


def _load_checkpoint(model: nn.Module, path: str) -> None:
    """Load a checkpoint to allow fine-tuning from custom weights."""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


def _reset_classifier(model: nn.Module, num_classes: int, *, head_attr: str, layer_index: Optional[int] = None) -> None:
    """Replace the final classifier layer if the requested number of classes differs."""
    head = getattr(model, head_attr)
    if layer_index is not None:
        head = head[layer_index]

    if isinstance(head, nn.Linear) and head.out_features != num_classes:
        new_head = nn.Linear(head.in_features, num_classes)
        if layer_index is None:
            setattr(model, head_attr, new_head)
        else:
            getattr(model, head_attr)[layer_index] = new_head  # type: ignore[index]


def _reset_vit_classifier(model: nn.Module, num_classes: int) -> None:
    """Torchvision ViT exposes the classification head via .heads."""
    heads = model.heads
    if hasattr(heads, "head") and isinstance(heads.head, nn.Linear):
        if heads.head.out_features != num_classes:
            heads.head = nn.Linear(heads.head.in_features, num_classes)
    else:
        children = list(heads.children())
        if children and isinstance(children[-1], nn.Linear) and children[-1].out_features != num_classes:
            children[-1] = nn.Linear(children[-1].in_features, num_classes)
            model.heads = nn.Sequential(*children)


def create_pretrained_model(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Build a torchvision backbone with an optional pretrained head reset.

    Parameters
    ----------
    model_name:
        One of ``resnet50``, ``mobilenet_v2``, ``vit_b_16``.
    num_classes:
        Target number of classes. Replaces the classifier layer when it differs from the pretrained head.
    pretrained:
        When True, load torchvision's default weights. Set False to initialise randomly.
    checkpoint_path:
        Optional path to a custom checkpoint. Loaded after the backbone is constructed.
    """
    name = model_name.lower()

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        _reset_classifier(model, num_classes, head_attr="fc")
    elif name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        _reset_classifier(model, num_classes, head_attr="classifier", layer_index=-1)
    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_16(weights=weights)
        _reset_vit_classifier(model, num_classes)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Expected resnet50, mobilenet_v2, or vit_b_16.")

    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)

    return model


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Register shared model CLI arguments."""
    group = parser.add_argument_group("model")
    group.add_argument(
        "--model-name",
        default="resnet50",
        choices=("resnet50", "mobilenet_v2", "vit_b_16"),
        help="Backbone architecture to instantiate.",
    )
    group.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of output classes. Falls back to dataset metadata when omitted.",
    )
    group.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable loading torchvision's default pretrained weights.",
    )
    group.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint to load after the model is built.",
    )


def model_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Translate CLI args into kwargs for ``create_pretrained_model``."""
    return {
        "model_name": getattr(args, "model_name", "resnet50"),
        "num_classes": getattr(args, "num_classes", None),
        "pretrained": not getattr(args, "no_pretrained", False),
        "checkpoint_path": getattr(args, "checkpoint_path", None),
    }


def _cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Instantiate a torchvision backbone and report its size.")
    add_model_args(parser)
    args = parser.parse_args(argv)

    kwargs = model_kwargs_from_args(args)
    if kwargs["num_classes"] is None:
        kwargs["num_classes"] = 1000

    model = create_pretrained_model(**kwargs)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"\nTotal parameters: {params:,}")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    _cli()
