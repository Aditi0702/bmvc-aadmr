#!/usr/bin/env python3
"""Utility script for running inference on an unlabeled image folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from DA.data_loader import _default_image_loader
from DA.models.models import create_pretrained_model
from DA.train import _build_transforms, _resolve_device


class ImageFileDataset(Dataset):
    """Minimal dataset that applies transforms to a static list of image paths."""

    def __init__(self, files: Sequence[Path], transform):
        self.files = list(files)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        sample = _default_image_loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a training checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .ckpt produced by train.py.")
    parser.add_argument("--test-dir", type=Path, required=True, help="Directory containing unlabeled images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.json. Defaults to <checkpoint_dir>/config.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Destination CSV file (Kaggle submission style).",
    )
    parser.add_argument("--device", default="auto", help="Device spec, e.g. 'cuda', 'cuda:1', 'cpu', or 'auto'.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--glob", default="*.jpg", help="Glob pattern to match inside --test-dir.")
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places for the probability columns.",
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
        raise ValueError("Config does not specify 'num_classes'; cannot build the classifier head.")

    device = _resolve_device(args.device)
    model = create_pretrained_model(**model_kwargs).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    transform = _build_val_transform(config.get("args", {}))
    test_dir = args.test_dir.expanduser().resolve()
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory '{test_dir}' not found.")

    image_paths = sorted(test_dir.glob(args.glob))
    if not image_paths:
        raise RuntimeError(f"No files matching '{args.glob}' found in '{test_dir}'.")

    dataset = ImageFileDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    columns = ["img"] + [f"c{i}" for i in range(num_classes)]
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with torch.no_grad(), output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for batch, names in loader:
            batch = batch.to(device, non_blocking=True)
            probs = F.softmax(model(batch), dim=1).cpu()
            for name, row in zip(names, probs):
                formatted = [f"{p:.{args.precision}f}" for p in row.tolist()]
                writer.writerow([name] + formatted)
                total += 1

    print(f"Wrote predictions for {total} images to {output_path}")


if __name__ == "__main__":
    main()
