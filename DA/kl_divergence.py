#!/usr/bin/env python3
"""Compute KL divergence between two image datasets via pretrained features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from DA.data_loader import CSVImageDataset, ImageFolderDataset
from DA.train import IMAGENET_MEAN, IMAGENET_STD, _resolve_device


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KL divergence between two datasets (feature space).")
    parser.add_argument("--data-a", type=Path, required=True, help="Root for dataset A (folder with class subdirs).")
    parser.add_argument("--data-b", type=Path, required=True, help="Root for dataset B.")
    parser.add_argument("--labels-csv-a", type=Path, default=None, help="Optional CSV for dataset A (StateFarm style).")
    parser.add_argument("--labels-csv-b", type=Path, default=None, help="Optional CSV for dataset B.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--image-size", type=int, default=224, help="Center-crop size.")
    parser.add_argument("--resize-size", type=int, default=256, help="Pre-resize shorter side before crop.")
    parser.add_argument("--limit-a", type=int, default=None, help="Optional cap on samples from dataset A.")
    parser.add_argument("--limit-b", type=int, default=None, help="Optional cap on samples from dataset B.")
    parser.add_argument("--device", default="auto", help="Device spec, e.g. 'cuda:0', 'cpu', or 'auto'.")
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Report the symmetric KL = 0.5*(KL(A||B)+KL(B||A)) instead of directional KL(A||B) only.",
    )
    return parser.parse_args(argv)


def _build_dataset(root: Path, labels_csv: Optional[Path], limit: Optional[int], transform) -> Dataset:
    if labels_csv:
        dataset = CSVImageDataset(root, labels_csv, transform=transform)
    else:
        dataset = ImageFolderDataset(root, transform=transform)

    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = Subset(dataset, range(limit))
    return dataset


def _build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _build_transform(image_size: int, resize_size: int):
    return transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_feature_extractor(device: torch.device) -> nn.Module:
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    return backbone


@torch.no_grad()
def _dataset_stats(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
    total = 0
    sum_vec: torch.Tensor | None = None
    sum_outer: torch.Tensor | None = None

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        feats = model(images).to(torch.float64)

        if sum_vec is None:
            dim = feats.shape[1]
            sum_vec = torch.zeros(dim, device=device, dtype=torch.float64)
            sum_outer = torch.zeros((dim, dim), device=device, dtype=torch.float64)

        sum_vec += feats.sum(dim=0)
        sum_outer += feats.t() @ feats
        total += feats.shape[0]

    if total == 0 or sum_vec is None or sum_outer is None:
        raise RuntimeError("No samples were processed; check dataset paths and filters.")

    mean = sum_vec / total
    cov = sum_outer / total - torch.outer(mean, mean)
    return total, mean, cov


def _kl_gaussians(mean_p: torch.Tensor, cov_p: torch.Tensor, mean_q: torch.Tensor, cov_q: torch.Tensor) -> float:
    dim = mean_p.numel()
    eps = 1e-6
    eye = torch.eye(dim, device=mean_p.device, dtype=mean_p.dtype)
    cov_p = cov_p + eps * eye
    cov_q = cov_q + eps * eye

    sign_p, logdet_p = torch.linalg.slogdet(cov_p)
    sign_q, logdet_q = torch.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        raise RuntimeError("Covariance not positive definite; try larger epsilon or more samples.")

    inv_q = torch.linalg.inv(cov_q)
    trace_term = torch.trace(inv_q @ cov_p)
    diff = (mean_q - mean_p).unsqueeze(0)  # 1 x D
    mahal = (diff @ inv_q @ diff.t()).squeeze()

    kl = 0.5 * (logdet_q - logdet_p - dim + trace_term + mahal)
    return float(kl.item())


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)

    transform = _build_transform(args.image_size, args.resize_size)

    dataset_a = _build_dataset(args.data_a, args.labels_csv_a, args.limit_a, transform)
    dataset_b = _build_dataset(args.data_b, args.labels_csv_b, args.limit_b, transform)

    loader_a = _build_loader(dataset_a, args.batch_size, args.num_workers)
    loader_b = _build_loader(dataset_b, args.batch_size, args.num_workers)

    model = _load_feature_extractor(device)

    n_a, mean_a, cov_a = _dataset_stats(loader_a, model, device)
    n_b, mean_b, cov_b = _dataset_stats(loader_b, model, device)

    kl_ab = _kl_gaussians(mean_a, cov_a, mean_b, cov_b)
    print(f"Samples: A={n_a}, B={n_b}")
    print(f"KL(A || B): {kl_ab:.4f}")

    if args.symmetric:
        kl_ba = _kl_gaussians(mean_b, cov_b, mean_a, cov_a)
        print(f"KL(B || A): {kl_ba:.4f}")
        print(f"Symmetric KL: {0.5 * (kl_ab + kl_ba):.4f}")


if __name__ == "__main__":
    main()
