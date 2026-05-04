#!/usr/bin/env python3
"""Command-line entry point for training classification models with ResNet-50."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn, optim
from torchvision import transforms

from DA.data_loader import (
    add_data_loader_args,
    build_dataloaders,
    dataloader_config_from_args,
)
from DA.models.models import (
    add_model_args,
    create_pretrained_model,
    model_kwargs_from_args,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a torchvision backbone (e.g., ResNet-50).")
    add_data_loader_args(parser)
    add_model_args(parser)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    train_group.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    train_group.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay factor.")
    train_group.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help="Optimizer to use for training.",
    )
    train_group.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    train_group.add_argument(
        "--scheduler",
        choices=("none", "step", "cosine", "plateau"),
        default="none",
        help="Optional learning-rate scheduler.",
    )
    train_group.add_argument("--step-size", type=int, default=30, help="StepLR step size (epochs).")
    train_group.add_argument("--gamma", type=float, default=0.1, help="LR decay factor for StepLR.")
    train_group.add_argument("--plateau-patience", type=int, default=3, help="Epochs with no improvement before LR drop.")
    train_group.add_argument("--plateau-factor", type=float, default=0.1, help="LR multiplier when plateau scheduler steps.")
    train_group.add_argument("--plateau-min-lr", type=float, default=1e-6, help="Minimal LR for plateau scheduler.")
    train_group.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-4,
        help="Minimal change in metric to qualify as improvement for plateau scheduler.",
    )
    train_group.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing.")
    train_group.add_argument("--grad-clip", type=float, default=None, help="Clip gradients to this norm.")
    train_group.add_argument("--amp", action="store_true", help="Enable mixed-precision (CUDA only).")
    train_group.add_argument("--log-interval", type=int, default=20, help="Steps between train log prints.")
    train_group.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")

    io_group = parser.add_argument_group("io")
    io_group.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory for checkpoints/logs.")
    io_group.add_argument("--experiment-name", type=str, default=None, help="Optional sub-folder name.")
    io_group.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard event files to <output-dir>/<experiment>/tensorboard.",
    )

    aug_group = parser.add_argument_group("augmentation")
    aug_group.add_argument("--image-size", type=int, default=224, help="Final crop size fed into the network.")
    aug_group.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Shorter-side resize before center-crop for validation.",
    )
    aug_group.add_argument(
        "--train-scale-min",
        type=float,
        default=0.8,
        help="Lower bound for RandomResizedCrop scale.",
    )
    aug_group.add_argument(
        "--train-scale-max",
        type=float,
        default=1.0,
        help="Upper bound for RandomResizedCrop scale.",
    )
    aug_group.add_argument(
        "--augment-strong",
        action="store_true",
        help="Enable a stronger augmentation recipe (color jitter, grayscale, small rotation).",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--device",
        default="auto",
        help="Device spec, e.g. 'cuda', 'cuda:1', 'cpu'. 'auto' picks CUDA when available.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)

    exp_name = args.experiment_name or _default_experiment_name(args.model_name)
    output_dir = (args.output_dir / exp_name).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = dataloader_config_from_args(args)
    train_transform, val_transform = _build_transforms(args)
    train_loader, val_loader = build_dataloaders(
        config,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    num_classes = _infer_num_classes(train_loader)
    model_kwargs = model_kwargs_from_args(args)
    if model_kwargs["num_classes"] is None:
        model_kwargs["num_classes"] = num_classes

    model = create_pretrained_model(**model_kwargs)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = _build_optimizer(args, model)
    scheduler = _build_scheduler(args, optimizer)

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch, best_acc = _maybe_resume(args.resume, model, optimizer, scaler, device)

    metadata_path = output_dir / "config.json"
    if not metadata_path.exists():
        _write_metadata(metadata_path, args, model_kwargs)

    writer = _maybe_create_summary_writer(args.tensorboard, output_dir)

    try:
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_acc = _train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                args,
            )
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if writer is not None:
                step = epoch + 1
                writer.add_scalar("Loss/train", train_loss, step)
                writer.add_scalar("Acc/train", train_acc, step)
                writer.add_scalar("Loss/val", val_loss, step)
                writer.add_scalar("Acc/val", val_acc, step)
                if scheduler is not None:
                    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)
                writer.flush()

            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if amp_enabled else None,
                "val_acc": val_acc,
                "best_acc": best_acc,
                "args": vars(args),
            }

            _save_checkpoint(checkpoint, output_dir / "last.ckpt")
            if val_acc > best_acc:
                best_acc = val_acc
                _save_checkpoint(checkpoint, output_dir / "best.ckpt")
    finally:
        if writer is not None:
            writer.close()


def _build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def _build_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer):
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
            threshold=args.plateau_threshold,
        )
    return None


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = scaler.is_enabled()

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if args.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

        if step % args.log_interval == 0 or step == len(loader):
            avg_loss = running_loss / total
            avg_acc = correct / total
            print(
                f"Epoch {epoch + 1} Step {step}/{len(loader)} "
                f"| loss={avg_loss:.4f} acc={avg_acc:.4f}"
            )

    return running_loss / total, correct / total


def _maybe_create_summary_writer(enabled: bool, output_dir: Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requested but the 'tensorboard' package is not installed. "
            "Install it with 'pip install tensorboard' or rerun without --tensorboard."
        ) from exc

    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    return loss_sum / total, correct / total


def _build_transforms(args: argparse.Namespace):
    if args.augment_strong:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    args.image_size,
                    scale=(args.train_scale_min, args.train_scale_max),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    args.image_size,
                    scale=(args.train_scale_min, args.train_scale_max),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize(args.resize_size, antialias=True),
            transforms.CenterCrop(args.image_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _default_experiment_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_{timestamp}"


def _extract_base_dataset(dataset):
    if hasattr(dataset, "dataset"):
        return _extract_base_dataset(dataset.dataset)
    if hasattr(dataset, "subset"):
        return _extract_base_dataset(dataset.subset)
    return dataset


def _infer_num_classes(loader: torch.utils.data.DataLoader) -> int:
    dataset = _extract_base_dataset(loader.dataset)
    classes = getattr(dataset, "classes", None)
    if classes is None:
        raise RuntimeError("Unable to infer number of classes from dataset; pass --num-classes explicitly.")
    return len(classes)


def _maybe_resume(
    checkpoint_path: Optional[Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    if checkpoint_path is None:
        return 0, 0.0
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' not found.")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if "optimizer_state" in state and state["optimizer_state"] is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scaler.is_enabled() and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])

    start_epoch = state.get("epoch", 0)
    best_acc = state.get("best_acc", 0.0)
    print(f"Resumed from {path} at epoch {start_epoch} with best_acc={best_acc:.4f}")
    return start_epoch, best_acc


def _save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, path)


def _write_metadata(path: Path, args: argparse.Namespace, model_kwargs: dict) -> None:
    payload = {
        "args": {k: _serialize_arg(v) for k, v in vars(args).items()},
        "model": model_kwargs,
    }
    path.write_text(json.dumps(payload, indent=2))


def _serialize_arg(value):
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
