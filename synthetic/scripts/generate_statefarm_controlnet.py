#!/usr/bin/env python3
"""
Generate synthetic StateFarm driver activity images using ControlNet + Stable Diffusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover - OpenCV is a runtime dependency
    raise SystemExit("OpenCV (cv2) is required. Install via `pip install opencv-python`.") from exc

try:
    from controlnet_aux import OpenposeDetector
except ImportError as exc:  # pragma: no cover - controlnet_aux is optional until runtime
    raise SystemExit("controlnet-aux is required. Install via `pip install controlnet-aux`.") from exc

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "diffusers is required. Install via `pip install diffusers transformers accelerate safetensors`."
    ) from exc


PROMPTS_10: Dict[str, str] = {

    "c0": "photo inside a moving car, driver holding the steering wheel with both hands, eyes open, focused on the road, daytime lighting",
    "c1": "inside car photo of a distracted driver texting on a smartphone with their right hand, eyes open, casual clothing, realistic interior",
    "c2": "photo inside a car of a driver talking on a phone with the right hand near their ear while steering, eyes open, natural lighting",
    "c3": "inside car photo of a driver texting with their left hand while the other hand is on the steering wheel, eyes open, realistic style",
    "c4": "driver talking on a phone with the left hand, sitting in the driver seat of a regular sedan, eyes open, natural daylight",
    "c5": "driver reaching toward the car radio or central console, adjusting controls, eyes open, interior photo",
    "c6": "driver taking a sip from a water bottle or travel cup while driving, eyes open, realistic car interior photo",
    "c7": "driver reaching behind the seat to grab an item, torso twisted, eyes open, car interior photo",
    "c8": "driver applying makeup with a compact mirror while seated behind the wheel, eyes open, realistic lighting",
    "c9": "driver looking and talking to a passenger on the right side, expressive gesture, eyes open, realistic interior photo"
}

# PROMPTS_10: Dict[str, str] = {
#     "c0": "photo inside a car, driver holding the steering wheel with both hands, eyes open, focused on the road",
#     "c1": "inside car photo of a distracted driver texting on a smartphone with their right hand, casual clothing, realistic interior",
#     "c2": "photo inside a car of a driver talking on a phone with the right hand near their ear while steering, natural lighting",
#     "c3": "inside car photo of a driver texting with their left hand while the other hand is on the steering wheel, realistic style",
#     "c4": "driver talking on a phone with the left hand, sitting in the driver seat of a regular sedan, natural daylight",
#     "c5": "driver reaching toward the car radio or central console, adjusting controls, interior photo",
#     "c6": "driver taking a sip from a water bottle or travel cup while driving, realistic car interior photo",
#     "c7": "driver reaching behind the seat to grab an item, torso twisted, car interior photo",
#     "c8": "driver applying makeup with a compact mirror while seated behind the wheel, realistic lighting",
#     "c9": "driver looking and talking to a passenger on the right side, expressive gesture, realistic interior photo",
# }

# Shared-7 subset (maps to c0,c6,c4,c2,c3,c1,c9 prompts)
PROMPTS_SHARED7: Dict[str, str] = {
    key: PROMPTS_10[key] for key in ("c0", "c6", "c4", "c2", "c3", "c1", "c9")
}
NEG_PROMPT = (
    "blurry, low resolution, distorted hands, extra limbs, deformed face, watermark, cartoon, CGI, text overlay, outdoor scene, view of street, road, highway, exterior shot, outside of car"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="StateFarm image root (imgs/train).")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="CSV with columns (subject, classname, img) describing the source frames.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write synthetic images.")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON for class -> natural language override. Otherwise built-in prompts are used.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of classes to synthesize (e.g., c0 c1 c2). Defaults to all present in the CSV.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many source frames.")
    parser.add_argument("--num-augments", type=int, default=2, help="Synthetic variants per source frame.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for ControlNet hints and generated images (width = height).",
    )
    parser.add_argument(
        "--base-model",
        default="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        help="Stable Diffusion checkpoint for photorealistic cabin renders.",
    )
    parser.add_argument(
        "--openpose-controlnet",
        default="lllyasviel/control_v11p_sd15_openpose",
        help="ControlNet id for pose conditioning.",
    )
    parser.add_argument(
        "--canny-controlnet",
        default="lllyasviel/control_v11p_sd15_canny",
        help="ControlNet id for edge conditioning.",
    )
    parser.add_argument(
        "--openpose-detector",
        default="lllyasviel/ControlNet",
        help="controlnet-aux OpenPose detector repo id.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device (cuda/cuda:1/cpu).")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--inference-steps", type=int, default=28, help="Diffusion steps.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    parser.add_argument(
        "--prompt-json",
        type=Path,
        default=None,
        help="Optional JSON overrides for prompts {classname: text}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Iterate through metadata without generating images.")

    # <<< NEW: how many synthetic images TOTAL per class
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=5,
        help="Target number of synthetic images to generate per class.",
    )
    # >>> END NEW

    return parser.parse_args()


def _load_prompts(args, classes: Sequence[str]) -> Dict[str, str]:
    if args.prompt_json:
        data = json.loads(args.prompt_json.read_text())
        return {cls: data.get(cls, PROMPTS_10.get(cls, cls)) for cls in classes}
    if args.label_map:
        data = json.loads(args.label_map.read_text())
        return {cls: data.get(cls, PROMPTS_10.get(cls, cls)) for cls in classes}
    # default to built-in prompts (choose shared subset if classes matches)
    if set(classes).issubset(PROMPTS_SHARED7.keys()):
        base = PROMPTS_SHARED7
    else:
        base = PROMPTS_10
    return {cls: base.get(cls, f"driver performing action {cls}") for cls in classes}


def _read_rows(csv_path: Path, allowed_classes: set[str] | None, limit: int | None):
    rows = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        path_column = next((col for col in ("filepath", "path", "relpath") if col in reader.fieldnames), None)
        for row in reader:
            cls = row["classname"].strip()
            if allowed_classes and cls not in allowed_classes:
                continue
            img_name = row.get("img", "").strip()
            rel_path_raw = row.get(path_column, "").strip() if path_column else ""
            if path_column and not rel_path_raw:
                continue
            if rel_path_raw:
                rel_path = Path(rel_path_raw.strip(" '\""))
            else:
                rel_path = Path(cls) / img_name
            rows.append((rel_path, cls))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("No rows matched the provided filters.")
    return rows


def _load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _resize_hint(img: Image.Image, resolution: int) -> Image.Image:
    return img.resize((resolution, resolution), resample=Image.BILINEAR)


def _make_canny(img: Image.Image, resolution: int) -> Image.Image:
    arr = np.array(img)
    edges = cv2.Canny(arr, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return _resize_hint(Image.fromarray(edges), resolution)


def build_pipeline(args) -> StableDiffusionControlNetPipeline:
    dtype = torch.float16 if "cuda" in args.device and torch.cuda.is_available() else torch.float32
    control_nets = [
        ControlNetModel.from_pretrained(args.openpose_controlnet, torch_dtype=dtype),
        ControlNetModel.from_pretrained(args.canny_controlnet, torch_dtype=dtype),
    ]
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=control_nets,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    if dtype == torch.float16 and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xFormers optional
    return pipe


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    csv_path = args.labels_csv.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_classes = set(args.classes) if args.classes else None
    rows = _read_rows(csv_path, allowed_classes, args.limit)
    classes = sorted(set(cls for _, cls in rows))
    prompts = _load_prompts(args, classes)

    # <<< NEW: counter of generated images per class
    synth_counts: Dict[str, int] = {cls: 0 for cls in classes}
    target_per_class = args.images_per_class
    # >>> END NEW

    detector = OpenposeDetector.from_pretrained(args.openpose_detector)
    pipe = None if args.dry_run else build_pipeline(args)

    generator = torch.Generator(device=args.device if "cuda" in args.device else "cpu")
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    for rel_path, cls in tqdm(rows, desc="Generating synthetic frames"):
        # <<< NEW: if we already have enough for this class, skip
        if synth_counts[cls] >= target_per_class:
            continue
        # >>> END NEW

        src_path = data_dir / rel_path
        if not src_path.exists():
            raise FileNotFoundError(f"Source image '{src_path}' not found.")
        base_name = src_path.stem
        class_out = out_dir / cls
        class_out.mkdir(parents=True, exist_ok=True)

        pose_hint_path = class_out / f"{base_name}_pose.png"
        canny_hint_path = class_out / f"{base_name}_canny.png"

        original = _load_image(src_path)
        pose_hint = detector(original)
        pose_hint = _resize_hint(pose_hint, args.resolution)
        canny_hint = _make_canny(original, args.resolution)

        pose_hint.save(pose_hint_path)
        canny_hint.save(canny_hint_path)

        if args.dry_run:
            continue

        for aug_idx in range(args.num_augments):
            # <<< NEW: stop if we've hit the per-class quota
            if synth_counts[cls] >= target_per_class:
                break
            # >>> END NEW

            seed = random.randint(0, 1_000_000)
            local_generator = generator.manual_seed(seed)
            images = pipe(
                prompt=prompts[cls],
                negative_prompt=NEG_PROMPT,
                image=[pose_hint, canny_hint],
                width=args.resolution,
                height=args.resolution,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                generator=local_generator,
            ).images
            for img_idx, image in enumerate(images):
                # <<< NEW: guard again inside inner loop
                if synth_counts[cls] >= target_per_class:
                    break
                # >>> END NEW

                out_path = class_out / f"{base_name}_synthetic_{aug_idx}_{img_idx}.png"
                if out_path.exists() and args.skip_existing:
                    continue
                image.save(out_path)
                synth_counts[cls] += 1  # <<< NEW: increment count for this class

        # <<< NEW: optional micro-optimization: if ALL classes reached target, stop early
        if all(count >= target_per_class for count in synth_counts.values()):
            break
        # >>> END NEW


if __name__ == "__main__":
    main()
