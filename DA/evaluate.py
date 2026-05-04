#!/usr/bin/env python3
"""Evaluate a checkpoint on a labeled dataset and report accuracy/F1/mAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from DA.data_loader import CSVImageDataset, ImageFolderDataset, _default_image_loader
from DA.models.models import create_pretrained_model
from DA.train import _build_transforms, _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute accuracy and macro F1 for a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .ckpt from train.py.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory with labeled images.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns (subject, classname, img) like StateFarm.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.json path (defaults to <checkpoint_dir>/config.json).",
    )
    parser.add_argument("--device", default="auto", help="Device spec, e.g. 'cuda', 'cuda:1', 'cpu'.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional glob pattern relative to --data-dir for nested class folders (e.g., 'Tester*/**/*.jpg').",
    )
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        help="Skip the dataset vs checkpoint class-count check (use with care).",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with path.open() as handle:
        return json.load(handle)


def _build_val_transform(cfg: dict):
    args = SimpleNamespace(
        image_size=cfg.get("image_size", 224),
        resize_size=cfg.get("resize_size", 256),
        train_scale_min=cfg.get("train_scale_min", 0.8),
        train_scale_max=cfg.get("train_scale_max", 1.0),
    )
    _, val_transform = _build_transforms(args)
    return val_transform


class _GlobLabeledDataset(Dataset):
    """Dataset that infers class labels from the parent directory using a glob pattern."""

    def __init__(self, root: Path, pattern: str, transform):
        self.root = root
        self.transform = transform
        self.files = sorted(root.glob(pattern))
        if not self.files:
            raise RuntimeError(f"No files matching pattern '{pattern}' found under '{root}'.")
        self.classes = sorted({path.parent.name for path in self.files})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        image = _default_image_loader(path)
        if self.transform:
            image = self.transform(image)
        label_name = path.parent.name
        if label_name not in self.class_to_idx:
            raise KeyError(f"Label '{label_name}' not found in class mapping.")
        return image, self.class_to_idx[label_name]


def _compute_metrics(confusion: torch.Tensor) -> tuple[float, float]:
    eps = 1e-12
    tp = confusion.diag()
    precision = tp / (confusion.sum(dim=0) + eps)
    recall = tp / (confusion.sum(dim=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = f1.mean().item()
    accuracy = tp.sum().item() / confusion.sum().item()
    return accuracy, macro_f1


def _compute_map(probabilities: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Mean Average Precision treating each class as a one-vs-rest ranking problem."""
    eps = 1e-12
    aps = []
    for cls in range(num_classes):
        cls_targets = (targets == cls).int()
        positive = cls_targets.sum().item()
        if positive == 0:
            aps.append(0.0)
            continue
        scores = probabilities[:, cls]
        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_targets = cls_targets[indices]
        tp_cum = sorted_targets.cumsum(0).float()
        precision = tp_cum / (torch.arange(1, sorted_targets.numel() + 1, dtype=torch.float32) + eps)
        ap = (precision * sorted_targets.float()).sum() / positive
        aps.append(ap.item())
    return float(sum(aps) / len(aps))


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' not found.")

    config_path = args.config.expanduser().resolve() if args.config else checkpoint.parent / "config.json"
    config = _load_config(config_path)
    model_kwargs = config.get("model", {})
    num_classes = model_kwargs.get("num_classes")
    if num_classes is None:
        raise ValueError("num_classes missing in config; cannot build classifier head.")

    device = _resolve_device(args.device)
    model = create_pretrained_model(**model_kwargs).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = state["model_state"]
    if not model_state:
        raise RuntimeError(f"Checkpoint '{checkpoint}' is missing 'model_state'.")
    first_key = next(iter(model_state))
    if first_key.startswith("module."):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.eval()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    transform = _build_val_transform(config.get("args", {}))
    dataset_kwargs = dict(transform=transform)
    if args.labels_csv:
        dataset = CSVImageDataset(data_dir, args.labels_csv.expanduser().resolve(), **dataset_kwargs)
    elif args.glob:
        dataset = _GlobLabeledDataset(data_dir, args.glob, transform=transform)
    else:
        dataset = ImageFolderDataset(data_dir, **dataset_kwargs)

    dataset_classes = getattr(dataset, "classes", None)
    if dataset_classes is not None and len(dataset_classes) != num_classes:
        if not args.allow_class_mismatch:
            raise ValueError(
                f"Dataset exposes {len(dataset_classes)} classes but checkpoint expects {num_classes}. "
                "Ensure you're evaluating on compatible labels or pass --allow-class-mismatch to override."
            )
        else:
            print(
                f"Warning: dataset has {len(dataset_classes)} classes but checkpoint expects {num_classes}; "
                "continuing due to --allow-class-mismatch."
            )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total = 0
    correct = 0
    prob_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            prob_chunks.append(torch.softmax(outputs, dim=1).cpu())
            target_chunks.append(targets.cpu())
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    accuracy, macro_f1 = _compute_metrics(confusion)
    probabilities = torch.cat(prob_chunks) if prob_chunks else torch.empty(0, num_classes)
    target_tensor = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.long)
    mean_ap = _compute_map(probabilities, target_tensor, num_classes) if probabilities.numel() else 0.0
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"mAP: {mean_ap:.4f}")


if __name__ == "__main__":
    main()
