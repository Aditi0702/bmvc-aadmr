#!/usr/bin/env python3
"""
Generate photorealistic driver-action augmentations using Flux + ControlNet.

This script mirrors ``generate_statefarm_controlnet.py`` but swaps the Stable Diffusion
backbone for the Flux family plus Flux-specific ControlNets (depth + canny, optional pose).
It extracts structural hints (Canny edges, MiDaS depth, DW-Pose) from existing StateFarm
frames and drives the Flux pipeline to create synthetic variations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
    import torch
except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyTorch is required. Install via `pip install torch torchvision`.") from exc

try:
    from diffusers import (
        FluxControlNetModel,
        FluxControlNetPipeline,
        FluxMultiControlNetModel,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "diffusers with Flux support is required. Install >=0.30 via "
        "`pip install diffusers transformers accelerate safetensors`."
    ) from exc

try:
    from controlnet_aux import CannyDetector, MidasDetector, DWposeDetector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "controlnet-aux is required for generating control hints. "
        "Install via `pip install controlnet-aux opencv-python`."
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
    "c9": "driver looking and talking to a passenger on the right side, expressive gesture, eyes open, realistic interior photo",
}

PROMPTS_SHARED7 = {cls: PROMPTS_10[cls] for cls in ("c0", "c6", "c4", "c2", "c3", "c1", "c9")}
NEG_PROMPT = (
    "blurry, low resolution, distorted hands, extra limbs, deformed face, watermark, cartoon, CGI, text overlay, outdoor scene,"
    " view of street, road, highway, exterior shot, outside of car"
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
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for synthetic renders.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of classes to synthesize (default: all classes present in CSV).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many source frames.")
    parser.add_argument("--num-augments", type=int, default=2, help="Synthetic variants per source frame.")
    parser.add_argument("--resolution", type=int, default=768, help="Square output resolution.")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    parser.add_argument(
        "--per-class-samples",
        type=int,
        default=None,
        help="Limit total synthetic outputs per class to this number.",
    )
    parser.add_argument(
        "--base-model",
        default="black-forest-labs/FLUX.1-dev",
        help="Flux checkpoint ID (e.g., FLUX.1-dev, FLUX.1-schnell, Flux1.1 Pro open weights).",
    )
    parser.add_argument(
        "--depth-controlnet",
        default="XLabs-AI/flux-controlnet-depth-diffusers",
        help="Flux ControlNet repo for depth conditioning.",
    )
    parser.add_argument(
        "--canny-controlnet",
        default="XLabs-AI/flux-controlnet-canny-diffusers",
        help="Flux ControlNet repo for canny conditioning.",
    )
    parser.add_argument(
        "--pose-controlnet",
        default=None,
        help="Optional Flux ControlNet repo for pose conditioning.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Classifier-free guidance scale.")
    parser.add_argument("--inference-steps", type=int, default=28, help="Number of diffusion steps.")
    parser.add_argument("--device", default="cuda", help="Device spec, e.g. 'cuda:0', 'cuda', or 'cpu'.")
    parser.add_argument("--prompt-json", type=Path, default=None, help="Optional JSON {classname: prompt}.")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON mapping for natural-language descriptions per class.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=None,
        help="Control strength per hint (depth, canny, pose). Defaults to [0.8, 0.5, 0.4].",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip renders that already exist on disk.")
    parser.add_argument("--dry-run", action="store_true", help="Walk through metadata without invoking Flux.")
    return parser.parse_args()


def _load_prompts(args: argparse.Namespace, classes: Sequence[str]) -> Dict[str, str]:
    if args.prompt_json:
        data = json.loads(args.prompt_json.read_text())
        return {cls: data.get(cls, PROMPTS_10.get(cls, cls)) for cls in classes}
    if args.label_map:
        data = json.loads(args.label_map.read_text())
        return {cls: data.get(cls, PROMPTS_10.get(cls, cls)) for cls in classes}
    return {cls: PROMPTS_10.get(cls, f"driver performing action {cls}") for cls in classes}


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
            rel_path_raw = row.get(path_column, "").strip() if path_column else None
            if rel_path_raw:
                rel_path = Path(rel_path_raw.strip(" '\""))
            elif img_name:
                rel_path = Path(cls) / img_name
            else:
                continue
            rows.append((rel_path, cls))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("No rows matched the provided filters.")
    return rows


#def _load_hint_generators(device: str, use_depth: bool, use_canny: bool, use_pose: bool):
def _load_hint_generators(main_device: str, use_depth: bool, use_canny: bool, use_pose: bool):
    """Load hint generators, distributing them across available GPUs if possible."""
    canny = CannyDetector() if use_canny else None
    depth = MidasDetector.from_pretrained("lllyasviel/ControlNet") if use_depth else None
    pose = DWposeDetector(device=main_device) if use_pose else None  # fixed

    # Distribute detectors to other GPUs to save VRAM on the main device
    depth_device = "cuda:1" if torch.cuda.device_count() > 1 else main_device
    pose_device = "cuda:2" if torch.cuda.device_count() > 2 else main_device

    depth = MidasDetector.from_pretrained("lllyasviel/ControlNet").to(depth_device) if use_depth else None
    pose = DWposeDetector(device=pose_device) if use_pose else None
    return canny, depth, pose



def _prepare_control_images(
    img: Image.Image,
    resolution: int,
    canny,
    depth,
    pose,
):
    resized = img.resize((resolution, resolution), Image.BILINEAR)
    canny_img = canny(resized) if canny else None
    depth_img = depth(resized) if depth else None
    pose_img = pose(resized) if pose else None
    return canny_img, depth_img, pose_img


def _build_controlnet(args: argparse.Namespace):
    control_modules: List[FluxControlNetModel] = []
    if args.depth_controlnet:
        control_modules.append(
            FluxControlNetModel.from_pretrained(args.depth_controlnet, torch_dtype=torch.float16)
        )
    if args.canny_controlnet:
        control_modules.append(
            FluxControlNetModel.from_pretrained(args.canny_controlnet, torch_dtype=torch.float16)
        )
    if args.pose_controlnet:
        control_modules.append(
            FluxControlNetModel.from_pretrained(args.pose_controlnet, torch_dtype=torch.float16)
        )
    if not control_modules:
        raise ValueError("At least one ControlNet (depth/canny/pose) must be provided.")
    if len(control_modules) == 1:
        return control_modules[0]
    return FluxMultiControlNetModel(control_modules)


def main() -> None:
    args = parse_args()

    allowed_classes = set(args.classes) if args.classes else None
    samples = _read_rows(args.labels_csv, allowed_classes, args.limit)
    prompts = _load_prompts(args, sorted({cls for _, cls in samples}))

    if args.scales is None:
        default_scales = [0.8, 0.5, 0.4]
    else:
        default_scales = args.scales

    controlnet = _build_controlnet(args)

    # --- Multi-GPU loading ---
    pipe = FluxControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        device_map="balanced",  # automatically distributes model across all GPUs
    )
    pipe.enable_attention_slicing()  # reduces peak memory usage

    # Prepare hint detectors
    canny_detector, depth_detector, pose_detector = _load_hint_generators(
        main_device="cuda",  # base device
        use_depth=bool(args.depth_controlnet),
        use_canny=bool(args.canny_controlnet),
        use_pose=bool(args.pose_controlnet),
    )

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else torch.seed()
    generator = torch.Generator(device="cuda")
    generated_counts = defaultdict(int)
    target_classes = sorted({cls for _, cls in samples})

    def class_done(name: str) -> bool:
        return args.per_class_samples is not None and generated_counts[name] >= args.per_class_samples

    def all_done() -> bool:
        return args.per_class_samples is not None and all(class_done(cls) for cls in target_classes)

    for rel_path, classname in tqdm(samples, desc="Generating"):
        if all_done():
            break
        if class_done(classname):
            continue
        src_path = (args.data_dir / rel_path).expanduser()
        if not src_path.exists():
            continue
        raw = Image.open(src_path).convert("RGB")
        canny_img, depth_img, pose_img = _prepare_control_images(
            raw,
            args.resolution,
            canny_detector,
            depth_detector,
            pose_detector,
        )

        control_images: List[Image.Image] = []
        if args.depth_controlnet and depth_img is not None:
            control_images.append(depth_img)
        if args.canny_controlnet and canny_img is not None:
            control_images.append(canny_img)
        if args.pose_controlnet and pose_img is not None:
            control_images.append(pose_img)

        scales = default_scales[: len(control_images)]
        class_out_dir = output_root / classname
        class_out_dir.mkdir(parents=True, exist_ok=True)
        prompt = prompts.get(classname, classname)

        for aug_idx in range(args.num_augments):
            if class_done(classname):
                break
            seed = base_seed + aug_idx
            generator.manual_seed(int(seed))
            filename = f"{src_path.stem}_synthetic_flux_{aug_idx}.png"
            dest = class_out_dir / filename
            if args.skip_existing and dest.exists():
                continue
            if args.dry_run:
                generated_counts[classname] += 1
                continue

            images = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                control_image=control_images,
                controlnet_conditioning_scale=scales,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                width=args.resolution,
                height=args.resolution,
            ).images

            images[0].save(dest)
            generated_counts[classname] += 1


if __name__ == "__main__":
    main()
