#!/usr/bin/env python3
"""Zero-shot evaluation on image classification datasets using Hugging Face CLIP checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from DA.data_loader import CSVImageDataset, ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot/linear-probe fine-tuning with CLIP.")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model id.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV describing (classname, img) pairs (StateFarm/SAM-DD style).",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON mapping class names to descriptive text prompts.",
    )
    parser.add_argument("--device", default="auto", help="Torch device spec.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of samples.")
    parser.add_argument(
        "--text-template",
        default="A photo of a driver doing {label}.",
        help="Template used to format prompts (label placeholder is '{label}').",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional JSONL file storing detailed predictions.",
    )
    parser.add_argument(
        "--mode",
        choices=("zero-shot", "finetune"),
        default="zero-shot",
        help="Select zero-shot evaluation or linear-probe fine-tuning.",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split for fine-tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for splitting.")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the classifier head.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for classifier training.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/clip"),
        help="Base directory for checkpoints/logs (each run gets an experiment subfolder).",
    )
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment subdirectory name (defaults to model_id + timestamp).",
    )
    parser.add_argument(
        "--load-visual-projection",
        type=Path,
        default=None,
        help="Optional path to a saved visual_projection state dict (e.g., from a fine-tuned CLIP run).",
    )
    parser.add_argument(
        "--train-vision-projection",
        action="store_true",
        help="Unfreeze CLIP's vision projection layer along with the classifier head.",
    )
    return parser.parse_args()


def _resolve_device(spec: str) -> torch.device:
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_dataset(data_dir: Path, labels_csv: Path | None):
    dataset_kwargs = dict(transform=None)
    if labels_csv:
        dataset = CSVImageDataset(data_dir, labels_csv.expanduser().resolve(), **dataset_kwargs)
    else:
        dataset = ImageFolderDataset(data_dir, **dataset_kwargs)
    return dataset


def _load_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Label map '{path}' not found.")
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Label map must be a JSON object mapping class -> text prompt segment.")
    return {str(k): str(v) for k, v in data.items()}


class ImagePathDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]]):
        self.samples = [(Path(path), int(label)) for path, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _make_collate_fn(include_paths: bool = False):
    def collate(batch):
        images: List[Image.Image] = []
        labels: List[int] = []
        paths: List[Path] = []
        for path, label in batch:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
            labels.append(label)
            if include_paths:
                paths.append(path)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        if include_paths:
            return paths, images, label_tensor
        return images, label_tensor

    return collate


def _default_experiment_name(model_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_id = model_id.replace("/", "_")
    return f"{safe_id}_{timestamp}"


def _experiment_dir(args) -> Path:
    base = args.output_dir.expanduser()
    name = args.experiment_name or _default_experiment_name(args.model_id)
    return base / name


def _maybe_create_writer(args, experiment_dir: Path):
    if not args.tensorboard:
        return None
    log_dir = experiment_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _split_samples(samples: Sequence[Tuple[Path, int]], train_split: float, seed: int):
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=rng).tolist()
    cutoff = int(len(samples) * train_split)
    train_idx = indices[:cutoff]
    val_idx = indices[cutoff:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def _format_prompts(classes: Sequence[str], label_map: dict[str, str], template: str) -> Sequence[str]:
    return [template.format(label=label_map.get(cls, cls)) for cls in classes]


def _compute_accuracy_f1(confusion: torch.Tensor) -> Tuple[float, float]:
    eps = 1e-12
    tp = confusion.diag()
    precision = tp / (confusion.sum(dim=0) + eps)
    recall = tp / (confusion.sum(dim=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = f1.mean().item()
    accuracy = tp.sum().item() / confusion.sum().item() if confusion.sum() else 0.0
    return accuracy, macro_f1


def _compute_map(probabilities: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    eps = 1e-12
    aps = []
    for cls in range(num_classes):
        cls_targets = (targets == cls).int()
        positive = cls_targets.sum().item()
        if positive == 0:
            aps.append(0.0)
            continue
        scores = probabilities[:, cls]
        _, indices = torch.sort(scores, descending=True)
        sorted_targets = cls_targets[indices]
        tp_cum = sorted_targets.cumsum(0).float()
        precision = tp_cum / (torch.arange(1, sorted_targets.numel() + 1, dtype=torch.float32) + eps)
        ap = (precision * sorted_targets.float()).sum() / max(positive, 1)
        aps.append(ap.item())
    return float(sum(aps) / len(aps)) if aps else 0.0


def _zero_shot_eval(args, device, processor, model, samples, classes, display_labels, writer=None):
    limit = args.limit if args.limit is not None else len(samples)
    samples = samples[:limit]

    text_inputs = processor(text=display_labels, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()

    dataset_wrapper = ImagePathDataset(samples)
    loader = DataLoader(
        dataset_wrapper,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_make_collate_fn(include_paths=True),
    )

    num_classes = len(classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    prob_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    predictions = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    topk = min(5, num_classes)

    for paths, images, labels in tqdm(loader, desc="CLIP zero-shot", unit="batch"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.t()) * logit_scale
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        prob_chunks.append(probs.cpu())
        target_chunks.append(labels.cpu())

        total += labels.size(0)
        top1_correct += (pred == labels).sum().item()
        top5 = logits.topk(topk, dim=1).indices
        top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        for path, pred_idx, label_idx in zip(paths, pred.cpu(), labels.cpu()):
            predictions.append(
                {
                    "image": str(path),
                    "prediction": classes[pred_idx.item()],
                    "prediction_text": display_labels[pred_idx.item()],
                    "target": classes[label_idx.item()],
                    "target_text": display_labels[label_idx.item()],
                    "correct": int(pred_idx.item() == label_idx.item()),
                }
            )
        for t, p in zip(labels.view(-1), pred.view(-1)):
            confusion[t.long(), p.long()] += 1

    probabilities = torch.cat(prob_chunks) if prob_chunks else torch.empty(0, num_classes)
    target_tensor = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.long)
    accuracy, macro_f1 = _compute_accuracy_f1(confusion)
    mean_ap = _compute_map(probabilities, target_tensor, num_classes) if probabilities.numel() else 0.0
    top5_acc = top5_correct / total if total else 0.0

    print(
        f"Zero-shot metrics on {total} samples | "
        f"top-1={accuracy:.4f} top-5={top5_acc:.4f} macro_f1={macro_f1:.4f} mAP={mean_ap:.4f}"
    )

    if writer is not None:
        writer.add_scalar("zero_shot/acc_top1", accuracy, 0)
        writer.add_scalar("zero_shot/acc_top5", top5_acc, 0)
        writer.add_scalar("zero_shot/macro_f1", macro_f1, 0)
        writer.add_scalar("zero_shot/mAP", mean_ap, 0)

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions.open("w") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        print(f"Saved predictions to '{args.save_predictions}'.")


def _evaluate_with_text_features(
    loader: DataLoader,
    processor,
    model: CLIPModel,
    text_features: torch.Tensor,
    logit_scale: float,
    device: torch.device,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    num_classes = text_features.size(0)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    prob_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    topk = min(5, num_classes)

    with torch.no_grad():
        for images, labels in loader:
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            labels = labels.to(device)
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.t()) * logit_scale
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)
            prob_chunks.append(probs.cpu())
            target_chunks.append(labels.cpu())
            total += labels.size(0)
            top1_correct += (preds == labels).sum().item()
            top5 = logits.topk(topk, dim=1).indices
            top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    probabilities = torch.cat(prob_chunks) if prob_chunks else torch.empty(0, num_classes)
    target_tensor = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.long)
    accuracy, macro_f1 = _compute_accuracy_f1(confusion)
    mean_ap = _compute_map(probabilities, target_tensor, num_classes) if probabilities.numel() else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    top5_acc = top5_correct / total if total else 0.0
    return accuracy, top5_acc, avg_loss, macro_f1, mean_ap


def _finetune(
    args,
    device,
    processor,
    base_model,
    samples,
    classes,
    display_labels,
    experiment_dir: Path,
    writer=None,
):
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be between 0 and 1.")

    limit = args.limit if args.limit is not None else len(samples)
    samples = samples[:limit]
    train_samples, val_samples = _split_samples(samples, args.train_split, args.seed)

    train_loader = DataLoader(
        ImagePathDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_make_collate_fn(),
    )
    val_loader = DataLoader(
        ImagePathDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_make_collate_fn(),
    )

    criterion = torch.nn.CrossEntropyLoss()

    text_inputs = processor(text=display_labels, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = base_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = base_model.logit_scale.exp().item()

    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.visual_projection.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        base_model.visual_projection.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        base_model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"CLIP finetune epoch {epoch}", unit="batch"):
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            image_features = base_model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.t()) * logit_scale
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_samples) if train_samples else 0.0
        val_acc, val_top5, val_loss, val_f1, val_map = _evaluate_with_text_features(
            val_loader,
            processor,
            base_model,
            text_features,
            logit_scale,
            device,
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
            f"val_macro_f1={val_f1:.4f} val_mAP={val_map:.4f}"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc_top1", val_acc, epoch)
            writer.add_scalar("val/acc_top5", val_top5, epoch)
            writer.add_scalar("val/macro_f1", val_f1, epoch)
            writer.add_scalar("val/mAP", val_map, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "visual_projection": base_model.visual_projection.state_dict(),
                "classes": classes,
                "config": {
                    "model_id": args.model_id,
                    "train_split": args.train_split,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "train_vision_projection": args.train_vision_projection,
                },
            }

    if best_state is not None:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = experiment_dir / "clip_visual_projection.pt"
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint (val_acc={best_acc:.4f}) to '{ckpt_path}'.")
    else:
        print("Fine-tuning finished, but no checkpoint was saved.")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    device = _resolve_device(args.device)
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    dataset = _load_dataset(data_dir, args.labels_csv)
    classes = getattr(dataset, "classes", None)
    if not classes:
        raise RuntimeError("Dataset must expose class names via .classes.")
    label_map = _load_label_map(args.label_map)
    display_labels = _format_prompts(classes, label_map, args.text_template)

    samples = list(dataset.samples)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id).to(device)
    if args.load_visual_projection:
        vp_path = args.load_visual_projection.expanduser().resolve()
        if not vp_path.exists():
            raise FileNotFoundError(f"visual_projection checkpoint '{vp_path}' not found.")
        state = torch.load(vp_path, map_location=device)
        if isinstance(state, dict) and "visual_projection" in state:
            state = state["visual_projection"]
        model.visual_projection.load_state_dict(state, strict=False)
        print(f"Loaded visual_projection weights from '{vp_path}'.")

    experiment_dir: Path | None = None
    if args.tensorboard or args.mode == "finetune":
        experiment_dir = _experiment_dir(args)
    writer = _maybe_create_writer(args, experiment_dir) if args.tensorboard else None

    try:
        if args.mode == "zero-shot":
            _zero_shot_eval(
                args,
                device,
                processor,
                model,
                samples,
                classes,
                display_labels,
                writer=writer,
            )
        else:
            assert experiment_dir is not None  # for type checkers
            _finetune(
                args,
                device,
                processor,
                model,
                samples,
                classes,
                display_labels,
                experiment_dir,
                writer=writer,
            )
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
