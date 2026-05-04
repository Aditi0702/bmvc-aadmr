"""
Utility helpers for building PyTorch DataLoaders from a simple folder layout.

The expected directory structure is the same as ``torchvision.datasets.ImageFolder``:

dataset_root/
├── class_a/
│   ├── img_000.jpg
│   └── ...
└── class_b/
    ├── img_042.jpg
    └── ...

This module intentionally keeps a small surface API to simplify experimentation
alongside the lightweight model helpers in ``DA/models``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, random_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = [
    "ImageFolderDataset",
    "CSVImageDataset",
    "DataLoaderConfig",
    "build_datasets",
    "build_dataloaders",
    "add_data_loader_args",
    "dataloader_config_from_args",
]


def _default_image_loader(path: Path) -> torch.Tensor:
    """Load an image from disk and convert it to a CHW float tensor in [0, 1]."""
    with Image.open(path) as img:
        image = img.convert("RGB")
    array = np.array(image, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return tensor


class ImageFolderDataset(Dataset):
    """
    A minimal replacement for torchvision.datasets.ImageFolder with zero dependencies.

    Parameters
    ----------
    root:
        Directory containing one sub-directory per class.
    transform:
        Callable applied to the PIL image before it is returned.
    target_transform:
        Callable applied to the integer class label.
    extensions:
        Allowed file extensions. Defaults to common image suffixes.
    loader:
        Callable that converts ``Path`` -> Tensor/PIL. Defaults to a lightweight RGB loader.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        loader: Optional[Callable[[Path], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")

        self.transform = transform
        self.target_transform = target_transform
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.loader = loader or _default_image_loader

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._find_samples()

    def _find_classes(self) -> Tuple[Sequence[str], dict]:
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not classes:
            raise RuntimeError(f"No class folders found in '{self.root}'.")
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    def _find_samples(self) -> Sequence[Tuple[Path, int]]:
        samples = []
        for cls in self.classes:
            class_dir = self.root / cls
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    samples.append((path, self.class_to_idx[cls]))
        if not samples:
            raise RuntimeError(f"No images with extensions {self.extensions} found in '{self.root}'.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)

        return sample, label


class CSVImageDataset(Dataset):
    """
    Dataset that reads labels from a CSV file containing (subject, classname, img) rows.

    Expected CSV columns: ``subject``, ``classname``, ``img``. Only ``classname`` and ``img`` are required.
    Images default to ``root/classname/img`` to match the StateFarm directory layout. When ``group_column``
    is provided, that CSV column is stored alongside each sample to enable group-aware train/validation
    splits (e.g., keep driver ``subject`` identities disjoint between splits).

    If one of the optional columns ``filepath``, ``path``, or ``relpath`` is present, each entry is
    resolved relative to ``root`` via that column instead. This supports datasets whose semantic class
    folders are not direct parents of the image files (e.g., Tester/Class/View/img.jpg).
    """

    def __init__(
        self,
        root: Path | str,
        labels_csv: Path | str,
        *,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        loader: Optional[Callable[[Path], torch.Tensor]] = None,
        group_column: Optional[str] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        csv_path = Path(labels_csv).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels CSV '{csv_path}' does not exist.")

        self.transform = transform
        self.target_transform = target_transform
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.loader = loader or _default_image_loader
        self.group_column = group_column

        self.samples, self.classes, self.class_to_idx, self.groups = self._parse_csv(csv_path)

    def _parse_csv(self, csv_path: Path):
        rows = []
        class_names = set()
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "classname" not in reader.fieldnames or "img" not in reader.fieldnames:
                raise ValueError("CSV must contain 'classname' and 'img' columns.")
            if self.group_column and self.group_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV must contain '{self.group_column}' when group_column is specified."
                )
            path_column = next(
                (col for col in ("filepath", "path", "relpath") if col in reader.fieldnames),
                None,
            )
            for entry in reader:
                classname = entry["classname"].strip()
                img_name = entry["img"].strip()
                if not classname or (not img_name and path_column is None):
                    continue
                rel_path = entry[path_column].strip() if path_column else None
                if path_column and not rel_path:
                    raise ValueError(f"Missing value for '{path_column}' in '{csv_path}'.")
                group_value = entry[self.group_column].strip() if self.group_column else None
                if self.group_column and not group_value:
                    raise ValueError(
                        f"Missing value for group column '{self.group_column}' in '{csv_path}'."
                    )
                class_names.add(classname)
                rows.append((classname, img_name, group_value, rel_path))

        if not rows:
            raise RuntimeError(f"No valid rows found in '{csv_path}'.")

        classes = sorted(class_names)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        samples = []
        groups = [] if self.group_column else None

        for classname, img_name, group_value, rel_path in rows:
            if rel_path:
                rel_path = Path(rel_path)
                path = rel_path if rel_path.is_absolute() else self.root / rel_path
            else:
                path = self.root / classname / img_name
            if path.suffix.lower() not in self.extensions:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Image '{path}' referenced in CSV was not found on disk.")
            samples.append((path, class_to_idx[classname]))
            if groups is not None:
                groups.append(group_value)

        if not samples:
            raise RuntimeError(f"No images resolved from '{csv_path}'.")

        return samples, classes, class_to_idx, groups

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


@dataclass
class DataLoaderConfig:
    """Configuration for ``build_datasets`` and ``build_dataloaders``."""

    data_dir: Path | str
    labels_csv: Path | None = None
    group_column: str | None = None
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    train_split: float = 0.8
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 < self.train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1 (exclusive).")


def build_datasets(
    config: DataLoaderConfig,
    *,
    transform: Optional[Callable] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Build train/validation datasets using a reproducible random split.

    Returns the tuple ``(train_dataset, val_dataset)``.
    """
    per_split_transforms = train_transform is not None or val_transform is not None

    dataset_kwargs = dict(
        transform=None if per_split_transforms else transform,
        target_transform=None if per_split_transforms else target_transform,
    )

    group_labels = None
    if config.labels_csv is not None:
        dataset = CSVImageDataset(
            config.data_dir,
            config.labels_csv,
            group_column=config.group_column,
            **dataset_kwargs,
        )
        group_labels = getattr(dataset, "groups", None)
    else:
        if config.group_column:
            raise ValueError("--group-column requires --labels-csv to be provided.")
        dataset = ImageFolderDataset(
            config.data_dir,
            **dataset_kwargs,
        )

    if config.group_column:
        if not group_labels:
            raise RuntimeError(
                "No group labels were parsed from the CSV; cannot create group-aware splits."
            )
        train_subset, val_subset = _split_dataset_by_group(
            dataset, group_labels, config.train_split, config.seed
        )
    else:
        generator = torch.Generator().manual_seed(config.seed)
        train_len = int(len(dataset) * config.train_split)
        val_len = len(dataset) - train_len
        if train_len == 0 or val_len == 0:
            raise ValueError(
                "train_split results in an empty train or validation set; adjust --train-split."
            )
        train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=generator)

    if per_split_transforms:
        train_subset = _SubsetWithTransform(
            train_subset,
            transform=train_transform or transform,
            target_transform=target_transform,
        )
        val_subset = _SubsetWithTransform(
            val_subset,
            transform=val_transform or transform,
            target_transform=target_transform,
        )

    return train_subset, val_subset


def build_dataloaders(
    config: DataLoaderConfig,
    *,
    transform: Optional[Callable] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience wrapper returning train/val dataloaders.

    Examples
    --------
    >>> from torchvision import transforms
    >>> cfg = DataLoaderConfig(data_dir="~/datasets/my_classification")
    >>> train_loader, val_loader = build_dataloaders(
    ...     cfg,
    ...     transform=transforms.Compose([
    ...         transforms.Resize((224, 224)),
    ...         transforms.ToTensor(),
    ...     ]),
    ... )
    """
    train_ds, val_ds = build_datasets(
        config,
        transform=transform,
        train_transform=train_transform,
        val_transform=val_transform,
        target_transform=target_transform,
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    train_loader = DataLoader(train_ds, shuffle=config.shuffle, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _split_dataset_by_group(
    dataset: Dataset,
    groups: Sequence[str],
    train_split: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    group_to_indices: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(idx)

    unique_groups = list(group_to_indices.keys())
    if len(unique_groups) < 2:
        raise ValueError("Need at least two distinct groups to perform a train/val split.")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique_groups), generator=generator).tolist()
    shuffled_groups = [unique_groups[i] for i in order]

    train_group_count = int(len(unique_groups) * train_split)
    train_group_count = max(1, min(len(unique_groups) - 1, train_group_count))

    train_group_names = set(shuffled_groups[:train_group_count])
    val_group_names = set(shuffled_groups[train_group_count:])

    train_indices: List[int] = []
    val_indices: List[int] = []
    for group_name, indices in group_to_indices.items():
        if group_name in train_group_names:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class _SubsetWithTransform(Dataset):
    """Apply per-split transforms while reusing a single indexed subset."""

    def __init__(
        self,
        subset: Dataset,
        *,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        sample, target = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target


def add_data_loader_args(parser: argparse.ArgumentParser) -> None:
    """Register common data-loading CLI arguments."""
    group = parser.add_argument_group("data")
    group.add_argument("--data-dir", type=Path, required=True, help="Root directory with class sub-folders.")
    group.add_argument("--batch-size", type=int, default=32, help="Batch size per step.")
    group.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    group.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of samples used for training (rest for validation).",
    )
    group.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    group.add_argument("--drop-last", action="store_true", help="Drop the last incomplete batch.")
    group.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling for the training DataLoader.",
    )
    group.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned memory buffers when transferring to CUDA.",
    )
    group.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns (subject, classname, img) to derive labels from.",
    )
    group.add_argument(
        "--group-column",
        type=str,
        default=None,
        help="Optional CSV column name used to keep samples from the same group in the same split "
        "(e.g., split StateFarm data by 'subject').",
    )


def dataloader_config_from_args(args: argparse.Namespace) -> DataLoaderConfig:
    """Create a ``DataLoaderConfig`` from parsed CLI arguments."""
    data_dir = getattr(args, "data_dir")
    if data_dir is None:
        raise ValueError("--data-dir is required.")

    shuffle = not getattr(args, "no_shuffle", False)
    pin_memory = not getattr(args, "no_pin_memory", False)
    labels_csv = getattr(args, "labels_csv", None)
    if labels_csv is not None:
        labels_csv = labels_csv.expanduser()
    group_column = getattr(args, "group_column", None)
    if group_column is not None and labels_csv is None:
        raise ValueError("--group-column can only be used when --labels-csv is provided.")
    return DataLoaderConfig(
        data_dir=Path(data_dir).expanduser(),
        labels_csv=labels_csv,
        group_column=group_column,
        batch_size=getattr(args, "batch_size", 32),
        num_workers=getattr(args, "num_workers", 4),
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=getattr(args, "drop_last", False),
        train_split=getattr(args, "train_split", 0.8),
        seed=getattr(args, "seed", 42),
    )


def _cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset statistics via DataLoaderConfig.")
    add_data_loader_args(parser)
    args = parser.parse_args(argv)
    config = dataloader_config_from_args(args)
    train_loader, val_loader = build_dataloaders(config)

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    total_samples = len(train_loader.dataset) + len(val_loader.dataset)

    print(f"Dataset root: {config.data_dir}")
    print(f"Total samples: {total_samples}")
    print(f"Train/val split: {config.train_split:.2f}/{1 - config.train_split:.2f}")
    print(f"Train batches: {train_batches}")
    print(f"Val batches: {val_batches}")
    print(f"Batch size: {config.batch_size}")
    print(f"Workers: {config.num_workers}")


if __name__ == "__main__":
    _cli()
