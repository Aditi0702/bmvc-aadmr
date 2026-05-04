#!/usr/bin/env python3
"""Visualize self-attention maps for a CLIP LoRA model (DINO-style CLS attention maps)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from DA.data_loader import CSVImageDataset, ImageFolderDataset

try:
    import matplotlib.cm as cm
except ImportError:
    cm = None


# ----------------- argument parsing & helpers ----------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ViT self-attention maps (CLS → patches) for a CLIP LoRA model."
    )
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32", help="Base CLIP model ID.")
    parser.add_argument("--lora-path", type=Path, required=True, help="Path to a directory with a LoRA adapter.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory with class sub-folders.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns (subject, classname, img). Use for StateFarm splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("self_attention_outputs_clip_lora"),
        help="Where to save the attention maps.",
    )
    parser.add_argument("--per-class", type=int, default=5, help="Number of examples to visualize per class.")
    parser.add_argument("--device", default="auto", help="Device spec, e.g. 'cuda', 'cuda:1', 'cpu', or 'auto'.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed to keep example picks reproducible.")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON mapping from class name to a readable description.",
    )
    parser.add_argument("--text-template", default="A photo of a driver doing {label}.", help="Prompt template.")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="If set, also save original+heatmap overlay images."
    )
    return parser.parse_args()


def _resolve_device(spec: str) -> torch.device:
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _create_dataset(args: argparse.Namespace, transform=None) -> Union[CSVImageDataset, ImageFolderDataset]:
    # Loader returns PIL Images for visualization.
    loader = lambda path: Image.open(path).convert("RGB")
    if args.labels_csv:
        return CSVImageDataset(args.data_dir, args.labels_csv, transform=transform, loader=loader)
    return ImageFolderDataset(args.data_dir, transform=transform, loader=loader)


def _sample_paths_per_class(
    dataset,
    per_class: int,
    seed: int,
) -> Dict[int, List[Tuple[str, int]]]:
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset.samples), generator=rng).tolist()
    per_class_paths: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(len(dataset.classes))}

    for idx in indices:
        path, label = dataset.samples[idx]
        if len(per_class_paths[label]) < per_class:
            per_class_paths[label].append((path, label))
        if all(len(paths) >= per_class for paths in per_class_paths.values()):
            break

    missing = [dataset.classes[i] for i, paths in per_class_paths.items() if len(paths) < per_class]
    if missing:
        raise RuntimeError(
            f"Unable to collect {per_class} samples for every class. Missing: {', '.join(missing)}"
        )
    return per_class_paths


def _get_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying CLIPModel even if wrapped by PEFT."""
    if hasattr(model, "base_model"):
        # PeftModel wraps the underlying CLIPModel as base_model.model
        return model.base_model.model
    return model


# ----------------- visualization helpers ----------------- #

def _to_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Convert attention map [H, W] in [0,1] to a DINO-like heatmap:
    use matplotlib 'viridis' if available (purple→green→yellow).
    """
    cam = np.clip(cam, 0.0, 1.0)

    if cm is not None:
        mapped = cm.get_cmap("viridis")(cam)[..., :3]
        return (mapped * 255).astype(np.uint8)

    # Fallback: simple purple/green-ish map
    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 1] = cam  # green
    heatmap[..., 2] = cam * 0.5  # blue
    return (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)


def _overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """
    Alpha-blend heatmap onto the image.
    cam is [H, W] in [0,1] and will be resized to match the image.
    alpha ~0.3–0.4 keeps the real image clearly visible.
    """
    h, w = cam.shape
    if image.size != (w, h):
        image = image.resize((w, h), Image.BICUBIC)

    base_uint8 = np.array(image)
    heatmap = _to_colormap(cam)
    blended = np.clip((1.0 - alpha) * base_uint8 + alpha * heatmap, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ----------------- label map & prompts ----------------- #

def _load_label_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open() as handle:
        return json.load(handle)


def _format_prompts(classes: Sequence[str], label_map: Dict[str, str], template: str) -> Sequence[str]:
    return [template.format(label=label_map.get(cls, cls)) for cls in classes]


# ----------------- main ----------------- #

def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    # Load Model and Processor
    processor = CLIPProcessor.from_pretrained(args.model_id)
    base_model = CLIPModel.from_pretrained(args.model_id)
    base_model.to(device)

    if args.lora_path:
        lora_path = args.lora_path.expanduser().resolve()
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA adapter path '{lora_path}' not found.")

        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
        print(f"Loaded LoRA adapter from '{lora_path}'")
    else:
        model = base_model

    model.to(device)
    model.eval()
    backbone = _get_backbone(model)

    # Force attention implementation that supports output_attentions
    if hasattr(backbone.config, "_attn_implementation"):
        backbone.config._attn_implementation = "eager"
    if hasattr(backbone, "_attn_implementation"):
        backbone._attn_implementation = "eager"
    if hasattr(backbone.vision_model.config, "attn_implementation"):
        backbone.vision_model.config.attn_implementation = "eager"
    if hasattr(backbone.vision_model, "_attn_implementation"):
        backbone.vision_model._attn_implementation = "eager"

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory '{args.data_dir}' not found.")

    # Raw PIL images; processor handles transforms
    dataset = _create_dataset(args, transform=None)
    samples = _sample_paths_per_class(dataset, args.per_class, args.seed)
    class_names = list(dataset.classes)
    label_map = _load_label_map(args.label_map)
    prompts = _format_prompts(class_names, label_map, args.text_template)

    # Text features just to report predictions (not needed for attention)
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for class_idx, paths_and_labels in tqdm(samples.items(), desc="Generating self-attention maps"):
            class_name = class_names[class_idx]
            readable = label_map.get(class_name, class_name)
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for path, _ in paths_and_labels:
                raw_image = Image.open(path).convert("RGB")

                # Processor to tensors
                inputs = processor(images=raw_image, return_tensors="pt").to(device)
                pixel_values = inputs["pixel_values"]  # [1, 3, 224, 224]

                # Forward with attentions
                vision_outputs = backbone.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                )

                attentions = vision_outputs.attentions
                if attentions is None:
                    raise RuntimeError(
                        "vision_outputs.attentions is None. "
                        "Check that attn_implementation is set to 'eager'."
                    )

                # attentions: tuple(num_layers) of [B, heads, seq, seq]
                last_attn = attentions[-1][0]  # [heads, seq_len, seq_len] for batch 0

                # Average over heads
                attn_mean = last_attn.mean(dim=0)  # [seq_len, seq_len]

                # CLS token is index 0; take its attention to all patch tokens
                cls_attn = attn_mean[0, 1:]  # [seq_len-1]

                num_patches = cls_attn.shape[0]
                grid_size = int(num_patches ** 0.5)  # e.g. 7x7 for ViT-B/32
                cam = cls_attn.reshape(1, 1, grid_size, grid_size)  # [1,1,H,W]

                # Upscale to 224×224 (image size)
                cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
                cam = cam.squeeze(0).squeeze(0)  # [H,W]

                # Normalize to [0,1]
                cam_min, cam_max = cam.min().item(), cam.max().item()
                if cam_max - cam_min > 1e-6:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = torch.zeros_like(cam)

                cam_np = cam.cpu().numpy()  # [224,224]

                # Predicted class (for filename)
                pooled_output = vision_outputs.pooler_output  # [1, hidden_dim]
                image_features = backbone.visual_projection(pooled_output)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = backbone.logit_scale.exp()
                scores = logit_scale * image_features @ text_features.t()
                pred_idx = int(scores.argmax(dim=1))
                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

                stem = Path(path).stem

                # Pure heatmap (224×224) if you want it
                heatmap_rgb = _to_colormap(cam_np)
                heatmap_img = Image.fromarray(heatmap_rgb)
                heatmap_name = f"{stem}_selfatt_true-{class_name}_pred-{pred_name}.png"
                heatmap_img.save(class_dir / heatmap_name)

                # Overlay on original image (what you care about)
                if args.overlay:
                    resized = raw_image.resize((224, 224))
                    overlay_img = _overlay_heatmap(resized, cam_np)
                    overlay_name = f"{stem}_overlay_selfatt_true-{class_name}_pred-{pred_name}.png"
                    overlay_img.save(class_dir / overlay_name)

                print(
                    f"[{class_name}] {Path(path).name} -> pred={pred_name} | "
                    f"saved {heatmap_name}"
                    + (f" ({readable})" if readable != class_name else "")
                )

    print(f"\nSaved self-attention maps to {output_dir}")


if __name__ == "__main__":
    main()
