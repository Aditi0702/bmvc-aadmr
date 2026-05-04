#!/usr/bin/env python3
"""
Zero-shot evaluation and optional fine-tuning for SigLIP checkpoints.

Workflow:
1. Run zero-shot classification on a labeled dataset to gauge baseline accuracy.
2. If accuracy is lower than desired, optionally fine-tune a linear head (or the full model)
   using the same dataset split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, SiglipModel

from DA.data_loader import CSVImageDataset, ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot + fine-tuning pipeline for SigLIP.")
    parser.add_argument(
        "--model-id",
        default="google/siglip-base-patch16-224",
        help="Hugging Face model repository id.",
    )
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
        help="Optional JSON mapping from dataset class names to descriptive labels for prompts.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "zero-shot", "finetune"),
        default="auto",
        help="Mode selection. 'auto' runs zero-shot then fine-tunes if accuracy < threshold.",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=None,
        help="When set and mode=auto, triggers fine-tuning if zero-shot accuracy is below this value.",
    )
    parser.add_argument("--device", default="auto", help="Torch device spec (auto picks CUDA when available).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation/fine-tuning.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples evaluated during zero-shot.",
    )
    parser.add_argument(
        "--text-template",
        default="A photo of a driver exhibiting behavior {label}.",
        help="Template used to construct text prompts for zero-shot classification.",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split (finetune mode).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for fine-tuning.")
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="If set, unfreezes SigLIP vision backbone during fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/siglip"),
        help="Directory for saving fine-tuned checkpoints/metadata.",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional JSONL file storing zero-shot predictions.",
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


def _load_label_map(path: Path | None):
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Label map file '{path}' not found.")
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Label map JSON must be an object mapping class -> description.")
    return {str(k): str(v) for k, v in data.items()}


class ImagePathDataset(Dataset):
    """Dataset that returns (path, label) pairs without loading image tensors ahead of time."""

    def __init__(self, samples: Sequence[Tuple[Path, int]]):
        self.samples = [(Path(path), int(label)) for path, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _make_collate_fn(processor, include_paths: bool = False):
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
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        label_tensor = torch.tensor(labels, dtype=torch.long)
        if include_paths:
            return paths, pixel_values, label_tensor
        return pixel_values, label_tensor

    return collate


def _format_prompts(classes: Sequence[str], template: str) -> Sequence[str]:
    return [template.format(label=cls) for cls in classes]


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


def zero_shot_evaluate(
    model: SiglipModel,
    processor,
    device: torch.device,
    samples: Sequence[Tuple[Path, int]],
    classes: Sequence[str],
    display_labels: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    model.eval()
    limit = args.limit if args.limit is not None else len(samples)
    eval_samples = samples[:limit]
    dataset = ImagePathDataset(eval_samples)
    collate = _make_collate_fn(processor, include_paths=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    prompts = _format_prompts(display_labels, args.text_template)
    text_inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = torch.tensor(1.0, device=device)

    num_classes = len(classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    prob_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    predictions = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    topk = min(5, num_classes)

    for paths, pixel_values, labels in tqdm(loader, desc="Zero-shot", unit="batch"):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        logits = (image_embeds @ text_embeds.t()) * scale
        preds = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=1)
        prob_chunks.append(probs.cpu())
        target_chunks.append(labels.cpu())
        total += labels.size(0)
        top1_correct += (preds == labels).sum().item()
        top5 = logits.topk(topk, dim=1).indices
        top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        for path, pred_idx, label_idx in zip(paths, preds.cpu(), labels.cpu()):
            pred_label = classes[pred_idx.item()]
            target_label = classes[label_idx.item()]
            predictions.append(
                {
                    "image": str(path),
                    "prediction": pred_label,
                    "prediction_text": display_labels[pred_idx.item()],
                    "target": target_label,
                    "target_text": display_labels[label_idx.item()],
                    "correct": int(pred_idx.item() == label_idx.item()),
                }
            )
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion[t.long(), p.long()] += 1

    probabilities = torch.cat(prob_chunks) if prob_chunks else torch.empty(0, num_classes)
    target_tensor = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.long)
    accuracy, macro_f1 = _compute_accuracy_f1(confusion)
    mean_ap = _compute_map(probabilities, target_tensor, num_classes) if probabilities.numel() else 0.0
    total = len(target_tensor)
    top5_acc = top5_correct / total if total else 0.0
    print(
        f"Zero-shot metrics on {total} samples | "
        f"top-1={accuracy:.4f} top-5={top5_acc:.4f} macro_f1={macro_f1:.4f} mAP={mean_ap:.4f}"
    )

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions.open("w") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        print(f"Saved zero-shot predictions to '{args.save_predictions}'.")

    return accuracy, top5_acc, macro_f1, mean_ap


class SiglipClassifier(nn.Module):
    def __init__(self, base_model: SiglipModel, num_classes: int, train_backbone: bool = False):
        super().__init__()
        self.base = base_model
        self.train_backbone = train_backbone
        embedding_dim = base_model.config.projection_dim
        self.classifier = nn.Linear(embedding_dim, num_classes)
        if not train_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.base.get_image_features(pixel_values=pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features)


def _split_samples(samples: Sequence[Tuple[Path, int]], train_split: float, seed: int):
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=rng).tolist()
    cutoff = int(len(samples) * train_split)
    train_idx = indices[:cutoff]
    val_idx = indices[cutoff:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def _evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    num_classes = model.classifier.out_features
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    prob_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    topk = min(5, num_classes)
    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values)
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


def fine_tune(
    base_model: SiglipModel,
    processor,
    device: torch.device,
    samples: Sequence[Tuple[Path, int]],
    classes: Sequence[str],
    args: argparse.Namespace,
) -> None:
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be between 0 and 1.")

    train_samples, val_samples = _split_samples(samples, args.train_split, args.seed)
    train_dataset = ImagePathDataset(train_samples)
    val_dataset = ImagePathDataset(val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_make_collate_fn(processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_make_collate_fn(processor),
    )

    classifier = SiglipClassifier(base_model, num_classes=len(classes), train_backbone=args.train_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (param for param in classifier.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        running_loss = 0.0
        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = classifier(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset) if len(train_dataset) else 0.0
        val_acc, val_top5, val_loss, val_f1, val_map = _evaluate_classifier(classifier, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
            f"val_macro_f1={val_f1:.4f} val_mAP={val_map:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model_state": classifier.state_dict(),
                "classes": classes,
                "config": {
                    "model_id": args.model_id,
                    "train_split": args.train_split,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "train_backbone": args.train_backbone,
                },
            }

    if best_state is not None:
        output_dir = args.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / "siglip_finetuned.pt"
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint (val_acc={best_acc:.4f}) to '{ckpt_path}'.")
    else:
        print("Fine-tuning finished, but no checkpoint was saved (no validation samples?).")


def main() -> None:
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
    display_labels = [label_map.get(cls, cls) for cls in classes]

    samples = list(dataset.samples)
    processor = AutoProcessor.from_pretrained(args.model_id)
    base_model = SiglipModel.from_pretrained(args.model_id).to(device)

    def run_zero_shot():
        accuracy, _, _, _ = zero_shot_evaluate(
            base_model,
            processor,
            device,
            samples,
            classes,
            display_labels,
            args,
        )
        return accuracy

    if args.mode == "zero-shot":
        run_zero_shot()
        return

    if args.mode == "finetune":
        fine_tune(base_model, processor, device, samples, classes, args)
        return

    # Auto mode: zero-shot first, then fine-tune if threshold not met
    accuracy = run_zero_shot()
    if args.auto_threshold is None:
        print("Auto mode complete (no threshold specified, skipping fine-tune).")
        return

    if accuracy >= args.auto_threshold:
        print(
            f"Zero-shot accuracy {accuracy:.4f} >= threshold {args.auto_threshold:.4f}; "
            "skipping fine-tune."
        )
        return

    print(
        f"Zero-shot accuracy {accuracy:.4f} < threshold {args.auto_threshold:.4f}; "
        "starting fine-tune stage."
    )
    fine_tune(base_model, processor, device, samples, classes, args)


if __name__ == "__main__":
    main()
