#!/usr/bin/env python3
"""CLIP zero-shot or LoRA fine-tuning on image classification datasets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from DA.data_loader import CSVImageDataset, ImageFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP zero-shot or LoRA fine-tuning (vision side only).")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model id.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root.")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional CSV (classname,img) layout.")
    parser.add_argument("--label-map", type=Path, default=None, help="Optional JSON class -> prompt text.")
    parser.add_argument("--mode", choices=("zero-shot", "finetune"), default="zero-shot", help="Select run mode.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples.")
    parser.add_argument("--text-template", default="A photo of a driver doing {label}.", help="Prompt template.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split for finetune.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--epochs", type=int, default=5, help="Finetune epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--train-vision-projection", action="store_true", help="Unfreeze visual_projection too.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/clip_lora"), help="Base run directory.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name.")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--save-predictions", type=Path, default=None, help="Optional JSONL predictions.")
    parser.add_argument("--load-visual-projection", type=Path, default=None, help="Optional projection checkpoint.")
    parser.add_argument("--load-adapter", type=Path, default=None, help="Optional existing LoRA adapter path.")
    parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Skip wrapping the model with LoRA; only finetune visual_projection (requires --train-vision-projection).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop finetuning if validation top-1 does not improve for this many epochs.",
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
    with path.open() as handle:
        data = json.load(handle)
    return {str(k): str(v) for k, v in data.items()}


class ImagePathDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]]):
        self.samples = [(Path(p), int(l)) for p, l in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


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
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_id.replace('/','_')}_{ts}"


def _experiment_dir(args) -> Path:
    name = args.experiment_name or _default_experiment_name(args.model_id)
    return args.output_dir.expanduser() / name


def _maybe_writer(args, exp_dir: Path):
    if not args.tensorboard:
        return None
    log_dir = exp_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _format_prompts(classes: Sequence[str], label_map: dict[str, str], template: str) -> Sequence[str]:
    return [template.format(label=label_map.get(cls, cls)) for cls in classes]


def _compute_metrics(confusion: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor):
    eps = 1e-12
    tp = confusion.diag()
    precision = tp / (confusion.sum(dim=0) + eps)
    recall = tp / (confusion.sum(dim=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = f1.mean().item()
    acc = tp.sum().item() / confusion.sum().item() if confusion.sum() else 0.0
    # mAP
    aps = []
    num_classes = probs.size(1)
    for cls in range(num_classes):
        cls_targets = (targets == cls).int()
        pos = cls_targets.sum().item()
        if pos == 0:
            aps.append(0.0)
            continue
        scores = probs[:, cls]
        _, idx = torch.sort(scores, descending=True)
        sorted_targets = cls_targets[idx]
        tp_cum = sorted_targets.cumsum(0).float()
        precision_cls = tp_cum / (torch.arange(1, sorted_targets.numel() + 1, dtype=torch.float32) + eps)
        ap = (precision_cls * sorted_targets.float()).sum() / max(pos, 1)
        aps.append(ap.item())
    mean_ap = float(sum(aps) / len(aps)) if aps else 0.0
    return acc, macro_f1, mean_ap


def _zero_shot(args, device, processor, model, samples, classes, prompts, writer=None):
    limit = args.limit if args.limit is not None else len(samples)
    samples = samples[:limit]

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.detach()  # freeze text side graph
    logit_scale = model.logit_scale.exp().detach()

    loader = DataLoader(
        ImagePathDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_make_collate_fn(include_paths=True),
    )

    num_classes = len(classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    prob_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    topk = min(5, num_classes)
    predictions = []

    for paths, images, labels in tqdm(loader, desc="CLIP zero-shot", unit="batch"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        labels = labels.to(device)
        with torch.no_grad():
            img_feat = model.get_image_features(pixel_values=pixel_values)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ text_features.t()) * logit_scale
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        prob_chunks.append(probs.cpu())
        target_chunks.append(labels.cpu())
        total += labels.size(0)
        top1_correct += (preds == labels).sum().item()
        top5 = logits.topk(topk, dim=1).indices
        top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion[t.long(), p.long()] += 1
        for path, pred_idx, label_idx in zip(paths, preds.cpu(), labels.cpu()):
            predictions.append(
                {
                    "image": str(path),
                    "prediction": classes[pred_idx.item()],
                    "target": classes[label_idx.item()],
                    "correct": int(pred_idx.item() == label_idx.item()),
                }
            )

    prob_tensor = torch.cat(prob_chunks) if prob_chunks else torch.empty(0, num_classes)
    target_tensor = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.long)
    acc, macro_f1, mean_ap = _compute_metrics(confusion, prob_tensor, target_tensor)
    top5_acc = top5_correct / total if total else 0.0
    print(
        f"Zero-shot metrics on {total} samples | top-1={acc:.4f} top-5={top5_acc:.4f} "
        f"macro_f1={macro_f1:.4f} mAP={mean_ap:.4f}"
    )
    if writer:
        writer.add_scalar("zero_shot/acc_top1", acc, 0)
        writer.add_scalar("zero_shot/acc_top5", top5_acc, 0)
        writer.add_scalar("zero_shot/macro_f1", macro_f1, 0)
        writer.add_scalar("zero_shot/mAP", mean_ap, 0)
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions.open("w") as f:
            for row in predictions:
                f.write(json.dumps(row) + "\n")
        print(f"Saved predictions to '{args.save_predictions}'.")


def _split(samples: Sequence[Tuple[Path, int]], split: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(samples), generator=g).tolist()
    cut = int(len(samples) * split)
    return [samples[i] for i in idx[:cut]], [samples[i] for i in idx[cut:]]


def _finetune(args, device, processor, model, samples, classes, prompts, exp_dir: Path, writer=None):
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be between 0 and 1.")
    limit = args.limit if args.limit is not None else len(samples)
    samples = samples[:limit]
    train_samples, val_samples = _split(samples, args.train_split, args.seed)

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

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).detach()
    logit_scale = model.logit_scale.exp().detach()

    if args.disable_lora and not args.train_vision_projection:
        raise ValueError("--disable-lora requires --train-vision-projection so there are trainable parameters.")

    if not args.disable_lora:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        model = get_peft_model(model, lora_config)

        for name, param in model.named_parameters():
            if "visual_projection" in name and args.train_vision_projection:
                param.requires_grad = True
            elif "lora_" in name:
                param.requires_grad = True
            else:
                # freeze everything else (including text tower)
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False
        if args.train_vision_projection:
            for name, param in model.named_parameters():
                if "visual_projection" in name:
                    param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    patience = args.early_stopping_patience if args.early_stopping_patience and args.early_stopping_patience > 0 else None
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"CLIP LoRA finetune {epoch}", unit="batch"):
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            img_feat = model.get_image_features(pixel_values=pixel_values)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ text_features.t()) * logit_scale
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_samples) if train_samples else 0.0
        val_acc, val_top5, val_loss, val_f1, val_map = _evaluate_with_text_features(
            val_loader, processor, model, text_features, logit_scale, device
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
            f"val_macro_f1={val_f1:.4f} val_mAP={val_map:.4f}"
        )
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc_top1", val_acc, epoch)
            writer.add_scalar("val/acc_top5", val_top5, epoch)
            writer.add_scalar("val/macro_f1", val_f1, epoch)
            writer.add_scalar("val/mAP", val_map, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "classes": classes,
                "config": vars(args),
            }
            if args.disable_lora:
                best_state["adapter"] = None
                best_state["visual_projection"] = (
                    model.visual_projection.state_dict() if args.train_vision_projection else None
                )
            else:
                best_state["adapter"] = model.state_dict()
                best_state["visual_projection"] = (
                    model.visual_projection.state_dict() if args.train_vision_projection else None
                )
            epochs_without_improve = 0
        else:
            if patience is not None:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no val improvement for {patience} consecutive epochs)."
                    )
                    break

    if best_state:
        exp_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, exp_dir / "clip_lora_best.pt")
        if args.disable_lora:
            if best_state["visual_projection"] is not None:
                torch.save(best_state["visual_projection"], exp_dir / "visual_projection.pt")
                print(f"Saved best visual_projection weights to '{exp_dir}'.")
            else:
                print(f"Saved projector-only run metadata to '{exp_dir}'.")
        else:
            (exp_dir / "adapter").mkdir(parents=True, exist_ok=True)
            # save LoRA adapter weights
            model.save_pretrained(exp_dir / "adapter")
            if args.train_vision_projection and best_state["visual_projection"] is not None:
                torch.save(best_state["visual_projection"], exp_dir / "visual_projection.pt")
            print(f"Saved best adapter to '{exp_dir}'.")
    else:
        print("Finished finetune with no validation samples; nothing saved.")


def _evaluate_with_text_features(loader, processor, model, text_features, logit_scale, device):
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
            img_feat = model.get_image_features(pixel_values=pixel_values)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ text_features.t()) * logit_scale
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
    prob_tensor = torch.cat(prob_chunks) if prob_chunks else torch.empty(0, num_classes)
    target_tensor = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.long)
    acc, macro_f1, mean_ap = _compute_metrics(confusion, prob_tensor, target_tensor)
    top5_acc = top5_correct / total if total else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return acc, top5_acc, avg_loss, macro_f1, mean_ap


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
    prompts = _format_prompts(classes, label_map, args.text_template)

    samples = list(dataset.samples)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id).to(device)
    if args.load_visual_projection:
        vp = args.load_visual_projection.expanduser().resolve()
        if not vp.exists():
            raise FileNotFoundError(f"visual_projection checkpoint '{vp}' not found.")
        state = torch.load(vp, map_location=device)
        if isinstance(state, dict) and "visual_projection" in state:
            state = state["visual_projection"]
        model.visual_projection.load_state_dict(state, strict=False)
        print(f"Loaded visual_projection from '{vp}'")
    if args.load_adapter:
        adapter_path = args.load_adapter.expanduser().resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter path '{adapter_path}' not found.")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        print(f"Loaded LoRA adapter from '{adapter_path}'")

    exp_dir = _experiment_dir(args)
    writer = _maybe_writer(args, exp_dir)
    if args.save_predictions and args.mode == "zero-shot":
        args.save_predictions = args.save_predictions.expanduser()

    try:
        if args.mode == "zero-shot":
            _zero_shot(args, device, processor, model, samples, classes, prompts, writer=writer)
        else:
            _finetune(args, device, processor, model, samples, classes, prompts, exp_dir, writer=writer)
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
