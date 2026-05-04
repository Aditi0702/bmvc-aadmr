#!/usr/bin/env python3
"""
Generate synthetic StateFarm driver activity images using ControlNet + Stable Diffusion
with OpenPose + Canny + Depth.

- Uses side-view prompts (camera from passenger seat) for all classes.
- Generates up to --images-per-class synthetic images per class.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV (cv2) is required. Install via `pip install opencv-python`.") from exc

try:
    from controlnet_aux import OpenposeDetector, MidasDetector
except ImportError as exc:
    raise SystemExit(
        "controlnet-aux is required. Install via `pip install controlnet-aux`."
    ) from exc

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
except ImportError as exc:
    raise SystemExit(
        "diffusers is required. Install via `pip install diffusers transformers accelerate safetensors`."
    ) from exc


# ---------------------- SIDE-VIEW PROMPTS (c0–c9) ---------------------- #

PROMPTS_10: Dict[str, str] = {
    "c0": (
        "side view of a driver inside a car, photographed from the front passenger seat angle, "
        "both hands on the steering wheel, eyes on the road, safe driving, "
        "natural daylight, high-resolution realistic photo"
    ),
    "c1": (
        "side view of a driver inside a car, camera at the front passenger seat, "
        "driver texting on a phone with the right hand while driving, distracted driving, "
        "realistic car interior photo"
    ),
    "c2": (
        "side view of a driver inside a car, camera from passenger seat, "
        "driver talking on a phone with the right hand near the ear while driving, "
        "distracted driving, realistic photo"
    ),
    "c3": (
        "side view of a driver inside a car, photographed from passenger seat, "
        "driver texting on a phone with the left hand while driving, "
        "distracted driving, realistic car interior"
    ),
    "c4": (
        "side view of a driver inside a car, passenger-seat camera angle, "
        "driver talking on a phone with the left hand near the ear while driving, "
        "distracted driving, realistic car interior photo"
    ),
    "c5": (
        "side view from the front passenger seat of a driver inside a car, "
        "driver reaching forward to adjust the radio or dashboard controls, "
        "distracted driving, realistic lighting"
    ),
    "c6": (
        "side passenger view of a driver inside a car, "
        "driver holding a drink or bottle while driving, "
        "distracted behavior, cinematic yet realistic interior photo"
    ),
    "c7": (
        "side view of a driver inside a car from the front passenger seat, "
        "driver turning the upper body and reaching behind the seat, "
        "distracted driving, realistic car interior"
    ),
    "c8": (
        "side passenger-seat view of a driver inside a car, "
        "driver using one hand to fix hair or makeup while driving, "
        "distracted behavior, ultra-realistic interior lighting"
    ),
    "c9": (
        "side view of a driver inside a car from the front passenger seat, "
        "driver slightly turned toward a passenger and talking, "
        "distracted driving, realistic car interior photo"
    ),
}

PROMPTS_SHARED7: Dict[str, str] = {
    key: PROMPTS_10[key] for key in ("c0", "c6", "c4", "c2", "c3", "c1", "c9")
}

NEG_PROMPT = (
    "blurry, low resolution, distorted hands, extra limbs, deformed face, watermark, cartoon, CGI, text overlay, "
    "outdoor scene, view of street, road, highway, exterior shot, outside of car"
)


# ---------------------- ARG PARSING ---------------------- #

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

    # *** NO explicit model IDs here – all are passed via CLI ***
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path or repo id for the base Stable Diffusion model (e.g. SD 1.5).",
    )
    parser.add_argument(
        "--openpose-controlnet",
        type=str,
        required=True,
        help="Path or repo id for the OpenPose ControlNet (SD1.5).",
    )
    parser.add_argument(
        "--canny-controlnet",
        type=str,
        required=True,
        help="Path or repo id for the Canny ControlNet (SD1.5).",
    )
    parser.add_argument(
        "--depth-controlnet",
        type=str,
        required=True,
        help="Path or repo id for the Depth ControlNet (SD1.5).",
    )
    parser.add_argument(
        "--openpose-detector",
        type=str,
        default="lllyasviel/ControlNet",
        help="Path or repo id for controlnet-aux OpenPose detector.",
    )
    parser.add_argument(
        "--depth-detector",
        type=str,
        default="lllyasviel/Annotators",
        help="Path or repo id for controlnet-aux depth detector (Midas).",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device (e.g., cuda, cuda:0, cuda:3, cpu).",
    )
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

    parser.add_argument(
        "--images-per-class",
        type=int,
        default=5,
        help="Target number of synthetic images to generate per class.",
    )

    return parser.parse_args()


# ---------------------- HELPERS ---------------------- #

def _load_prompts(args, classes: Sequence[str]) -> Dict[str, str]:
    if args.prompt_json:
        data = json.loads(args.prompt_json.read_text())
        return {cls: data.get(cls, PROMPTS_10.get(cls, cls)) for cls in classes}
    if args.label_map:
        data = json.loads(args.label_map.read_text())
        return {cls: data.get(cls, PROMPTS_10.get(cls, cls)) for cls in classes}
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
    return Image.open(path).convert("RGB")


def _resize_hint(img: Image.Image, resolution: int) -> Image.Image:
    return img.resize((resolution, resolution), resample=Image.BILINEAR)


def _make_canny(img: Image.Image, resolution: int) -> Image.Image:
    arr = np.array(img)
    edges = cv2.Canny(arr, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return _resize_hint(Image.fromarray(edges), resolution)


def _make_depth(img: Image.Image, depth_detector: MidasDetector, resolution: int) -> Image.Image:
    depth_img = depth_detector(img)
    depth_img = _resize_hint(depth_img, resolution)
    return depth_img


# ---------------------- PIPELINE BUILD ---------------------- #

def build_pipeline(args) -> StableDiffusionControlNetPipeline:
    dtype = torch.float16 if ("cuda" in args.device and torch.cuda.is_available()) else torch.float32

    control_nets = [
        ControlNetModel.from_pretrained(args.openpose_controlnet, torch_dtype=dtype),
        ControlNetModel.from_pretrained(args.canny_controlnet, torch_dtype=dtype),
        ControlNetModel.from_pretrained(args.depth_controlnet, torch_dtype=dtype),
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


# ---------------------- MAIN ---------------------- #

def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    csv_path = args.labels_csv.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # read all rows from CSV
    rows = _read_rows(csv_path, set(args.classes) if args.classes else None, args.limit)
    classes = sorted(set(cls for _, cls in rows))
    prompts = _load_prompts(args, classes)

    # group rows per class (like SDXL script)
    rows_by_class: Dict[str, List[Path]] = {cls: [] for cls in classes}
    for rel_path, cls in rows:
        rows_by_class[cls].append(rel_path)

    synth_counts: Dict[str, int] = {cls: 0 for cls in classes}
    target_per_class = args.images_per_class

    print("Loading OpenPose + depth annotators...")
    pose_detector = OpenposeDetector.from_pretrained(args.openpose_detector)
    depth_detector = MidasDetector.from_pretrained(args.depth_detector)

    pipe = None if args.dry_run else build_pipeline(args)

    generator = torch.Generator(device=args.device if "cuda" in args.device else "cpu")
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    # -------- per-class loop, SDXL-style --------
    for cls in classes:
        target = target_per_class
        if target <= 0:
            continue

        cls_rows = rows_by_class[cls]
        if not cls_rows:
            print(f"[WARN] No rows for class {cls}, skipping.")
            continue

        class_out = out_dir / cls
        class_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Generating for class {cls} (target {target}) ===")
        pbar = tqdm(total=target, desc=f"class {cls}")

        # iterate over frames for this class
        for rel_path in cls_rows:
            if synth_counts[cls] >= target:
                break

            src_path = data_dir / rel_path
            if not src_path.exists():
                print(f"[WARN] Source image '{src_path}' not found, skipping.")
                continue

            base_name = src_path.stem

            pose_hint_path = class_out / f"{base_name}_pose.png"
            canny_hint_path = class_out / f"{base_name}_canny.png"
            depth_hint_path = class_out / f"{base_name}_depth.png"

            original = _load_image(src_path)

            # compute hints
            pose_hint = pose_detector(original)
            pose_hint = _resize_hint(pose_hint, args.resolution)
            canny_hint = _make_canny(original, args.resolution)
            depth_hint = _make_depth(original, depth_detector, args.resolution)

            # save hint images
            pose_hint.save(pose_hint_path)
            canny_hint.save(canny_hint_path)
            depth_hint.save(depth_hint_path)

            if args.dry_run:
                continue

            # generate up to num_augments per source frame
            for aug_idx in range(args.num_augments):
                if synth_counts[cls] >= target:
                    break

                seed = random.randint(0, 1_000_000)
                local_generator = generator.manual_seed(seed)

                images = pipe(
                    prompt=prompts[cls],
                    negative_prompt=NEG_PROMPT,
                    image=[pose_hint, canny_hint, depth_hint],
                    width=args.resolution,
                    height=args.resolution,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    generator=local_generator,
                ).images

                for img_idx, image in enumerate(images):
                    if synth_counts[cls] >= target:
                        break

                    out_path = class_out / f"{base_name}_synthetic_{aug_idx}_{img_idx}.png"
                    if out_path.exists() and args.skip_existing:
                        continue

                    image.save(out_path)
                    synth_counts[cls] += 1
                    pbar.update(1)

        pbar.close()

        if synth_counts[cls] < target:
            print(
                f"[INFO] Only generated {synth_counts[cls]} images for {cls} "
                f"(not enough frames or early stop)."
            )

    print("\nDone. Generated per class:")
    for cls in classes:
        print(f"  {cls}: {synth_counts[cls]} images")



if __name__ == "__main__":
    main()
