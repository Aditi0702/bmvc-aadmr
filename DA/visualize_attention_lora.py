#!/usr/bin/env python3
"""Generate simple Grad-CAM overlays to see what the classifier attends to."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from DA.data_loader import CSVImageDataset, ImageFolderDataset, _default_image_loader
from DA.models.models import create_pretrained_model
from DA.train import IMAGENET_MEAN, IMAGENET_STD, _resolve_device

try:
    import matplotlib.cm as cm
except ImportError:
    cm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM heatmaps per class.")
    parser.add_argument("--lora-path", type=Path, required=True, help="Path to a .safetensors LoRA weights file.")
    parser.add_argument("--model-name", type=str, default="resnet50", help="Name of the base model architecture.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a .ckpt produced by train.py.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory with class sub-folders.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns (subject, classname, img). Use for StateFarm splits.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to <checkpoint_dir>/config.json when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cam_outputs"),
        help="Where to save the overlaid images.",
    )
    parser.add_argument("--per-class", type=int, default=3, help="Number of examples to visualize per class.")
    parser.add_argument("--device", default="auto", help="Device spec, e.g. 'cuda', 'cuda:1', 'cpu', or 'auto'.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed to keep example picks reproducible.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image_size if config.json is missing that field.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=None,
        help="Override resize_size if config.json is missing that field.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON mapping from class name to a readable description.",
    )
    return parser.parse_args()


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with path.open() as handle:
        return json.load(handle)


def _resolve_config_path(checkpoint: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override.expanduser().resolve()
    candidate = checkpoint.parent / "config.json"
    return candidate if candidate.exists() else None


def _resolve_sizes(args: argparse.Namespace, config: Mapping) -> tuple[int, int]:
    cfg_args = config.get("args", {}) if isinstance(config, Mapping) else {}
    image_size = args.image_size or cfg_args.get("image_size", 224)
    resize_size = args.resize_size or cfg_args.get("resize_size", 256)
    return int(image_size), int(resize_size)


def _build_cam_transforms(image_size: int, resize_size: int):
    base = [
        transforms.Resize(resize_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]
    normalized = transforms.Compose(base + [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    visual = transforms.Compose(base)
    return normalized, visual


def _create_dataset(args: argparse.Namespace):
    if args.labels_csv:
        return CSVImageDataset(args.data_dir, args.labels_csv)
    return ImageFolderDataset(args.data_dir)


def _sample_paths_per_class(
    dataset,
    per_class: int,
    seed: int,
) -> Dict[int, List[Path]]:
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset.samples), generator=rng).tolist()
    per_class_paths: Dict[int, List[Path]] = {i: [] for i in range(len(dataset.classes))}

    for idx in indices:
        path, label = dataset.samples[idx]
        if len(per_class_paths[label]) < per_class:
            per_class_paths[label].append(Path(path))
        if all(len(paths) >= per_class for paths in per_class_paths.values()):
            break

    missing = [dataset.classes[i] for i, paths in per_class_paths.items() if len(paths) < per_class]
    if missing:
        raise RuntimeError(
            f"Unable to collect {per_class} samples for every class. Missing: {', '.join(missing)}"
        )
    return per_class_paths


def _load_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if all(key.startswith("module.") for key in state_dict):
        cleaned = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(cleaned)
        return

    raise


def _select_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    name = model_name.lower()
    if name == "resnet50":
        return base.layer4[-1]
    if name == "mobilenet_v2":
        return base.features[-1]
    raise ValueError(f"Grad-CAM target layer not defined for model '{model_name}'.")


class GradCAM:
    """Lightweight Grad-CAM helper for CNN backbones."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, _module, grad_inputs, grad_outputs) -> None:
        del grad_inputs  # unused
        self.gradients = grad_outputs[0]

    def compute(self, scores: torch.Tensor, class_idx: int, input_size: Sequence[int]) -> torch.Tensor:
        if self.activations is None:
            raise RuntimeError("Forward hook did not capture activations.")
        self.model.zero_grad(set_to_none=True)
        scores[:, class_idx].sum().backward(retain_graph=True)
        if self.gradients is None:
            raise RuntimeError("Backward hook did not capture gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.squeeze(0).squeeze(0)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _to_colormap(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    if cm is not None:
        mapped = cm.get_cmap("magma")(cam)[..., :3]
        return (mapped * 255).astype(np.uint8)

    # Fallback: simple red-yellow heatmap without matplotlib.
    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 0] = cam  # red
    heatmap[..., 1] = np.sqrt(cam)  # yellow-ish as values increase
    heatmap[..., 2] = (1.0 - cam) * 0.3
    return (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)


def _overlay_heatmap(image_tensor: torch.Tensor, cam: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    base = image_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    base_uint8 = (base * 255).astype(np.uint8)
    heatmap = _to_colormap(cam.detach().cpu().numpy())
    blended = np.clip((1.0 - alpha) * base_uint8 + alpha * heatmap, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _load_label_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' not found.")

    config_path = _resolve_config_path(checkpoint, args.config)
    config = _load_config(config_path)
    model_kwargs = config.get("model", {})
    if not model_kwargs:
        raise ValueError("Model settings were not found; pass a valid config.json via --config.")

    if args.model_name:
        model_kwargs["model_name"] = args.model_name

    dataset = _create_dataset(args)
    if model_kwargs.get("num_classes") is None:
        model_kwargs["num_classes"] = len(dataset.classes)

    image_size, resize_size = _resolve_sizes(args, config)
    norm_transform, visual_transform = _build_cam_transforms(image_size, resize_size)

    model = create_pretrained_model(**model_kwargs).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    _load_state_dict(model, state_dict)
    model.eval()

    target_layer = _select_target_layer(model, model_kwargs.get("model_name", ""))
    cam_extractor = GradCAM(model, target_layer)

    samples = _sample_paths_per_class(dataset, args.per_class, args.seed)
    class_names = list(dataset.classes)
    label_map = _load_label_map(args.label_map)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.enable_grad():
        for class_idx, paths in samples.items():
            class_name = class_names[class_idx]
            readable = label_map.get(class_name, class_name)
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for path in paths:
                raw = _default_image_loader(path)
                model_input = norm_transform(raw).unsqueeze(0).to(device)
                vis_tensor = visual_transform(raw)

                scores = model(model_input)
                pred_idx = int(scores.argmax(dim=1))
                cam = cam_extractor.compute(scores, class_idx, input_size=model_input.shape[-2:])
                overlay = _overlay_heatmap(vis_tensor, cam)

                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                filename = f"{path.stem}_true-{class_name}_pred-{pred_name}.png"
                overlay.save(class_dir / filename)

                print(
                    f"[{class_name}] {path.name} -> pred={pred_name} | saved {filename}"
                    + (f" ({readable})" if readable != class_name else "")
                )

    cam_extractor.close()
    print(f"\nSaved Grad-CAM overlays to {output_dir}")


if __name__ == "__main__":
    main()
