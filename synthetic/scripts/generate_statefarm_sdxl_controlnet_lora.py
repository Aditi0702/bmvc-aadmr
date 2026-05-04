#!/usr/bin/env python3
"""
Generate synthetic StateFarm driver activity images using SDXL + depth & openpose
ControlNets + optional LoRA.

- Uses side-view prompts (camera from passenger seat) for all classes.
- Generates N images per class (default 5).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from controlnet_aux import OpenposeDetector, MidasDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
)

# ---------------------- PROMPTS (SIDE VIEW) ---------------------- #

STATEFARM_PROMPTS: Dict[str, str] = {
    "c0": (
        "side view of a driver inside a car, photographed from the front passenger seat angle, "
        "both hands on the steering wheel, eyes on the road, safe driving, "
        "natural daylight, high-resolution realistic DSLR photo"
    ),
    "c1": (
        "side view of a driver inside a car, camera at the front passenger seat, "
        "driver texting on a phone with the right hand while driving, distracted driving, "
        "realistic car interior photo"
    ),
    "c2": (
        "side view of a driver inside a car, camera from passenger seat, "
        "driver talking on a phone with the right hand near the ear while driving, "
        "distracted driving, realistic DSLR photo"
    ),
    "c3": (
        "side view of a driver inside a car, photographed from passenger seat, "
        "driver texting on a phone with the left hand while driving, "
        "distracted driving, realistic car interior"
    ),
    "c4": (
        "side view of a driver inside a car, passenger-seat camera angle, "
        "driver talking on a phone with the left hand near the ear while driving, "
        "distracted driving, realistic photo"
    ),
    "c5": (
        "side view from the front passenger seat of a driver inside a car, "
        "driver reaching forward to adjust the radio or dashboard controls, "
        "distracted driving, realistic lighting"
    ),
    "c6": (
        "side passenger view of a driver inside a car, "
        "driver holding a drink or bottle while driving, "
        "distracted behavior, cinematic yet realistic photo"
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
        "distracted driving, realistic photo"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, extra limbs, deformed hands, missing fingers, "
    "multiple faces, cartoon, painting, drawing, watermark, text, logo"
)

# ---------------------- ARG PARSING ---------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SDXL + ControlNet + LoRA synthetic images for StateFarm classes."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of real images, one subfolder per class (e.g. c0, c1, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to save generated images.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to SDXL base model (Diffusers).",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        required=True,
        help="Path to SDXL depth ControlNet (Diffusers).",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        required=True,
        help="Path to SDXL openpose ControlNet (Diffusers).",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Optional path to SDXL VAE (e.g. sdxl-vae-fp16-fix).",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Optional path to LoRA weights directory or .safetensors file.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.8,
        help="LoRA scale when fusing into pipeline.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Number of synthetic images to generate per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda | mps | cpu",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion steps.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=6.5,
        help="Guidance scale.",
    )
    return parser.parse_args()

# ---------------------- UTILS ---------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_classes(data_dir: Path) -> list[str]:
    classes = [p.name for p in data_dir.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def build_pipeline(args: argparse.Namespace) -> StableDiffusionXLControlNetPipeline:
    print("Loading ControlNet models...")
    depth_cn = ControlNetModel.from_pretrained(
        args.depth_model,
        torch_dtype=torch.float16,
    )
    pose_cn = ControlNetModel.from_pretrained(
        args.pose_model,
        torch_dtype=torch.float16,
    )

    controlnets = [depth_cn, pose_cn]

    vae = None
    if args.vae is not None:
        print(f"Loading VAE from {args.vae}")
        vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16)

    print("Loading SDXL base model...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnets,
        vae=vae,
        torch_dtype=torch.float16,
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # --- move pipeline to device properly ---
    # supports "cuda", "cuda:3", "mps", "cpu"
    if args.device.startswith("cuda"):
        print(f"Moving pipeline to device: {args.device}")
        pipe.to(args.device)
    elif args.device == "mps":
        print("Moving pipeline to Apple MPS")
        pipe.to("mps")
    else:
        print("Using CPU (this will be very slow).")
        pipe.to("cpu")

    # --- memory saving features ---
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Using xFormers memory efficient attention.")
    except Exception:
        print("xFormers not available, continuing without it.")

    # --- optional LoRA ---
    if args.lora:
        print(f"Loading LoRA weights from {args.lora}")
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)
        print(f"Fused LoRA with scale={args.lora_scale}")
    else:
        print("No LoRA provided — running plain SDXL + ControlNet.")

    return pipe


def build_annotators(device: str):
    print("Loading annotators (depth + openpose)...")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    depth = MidasDetector.from_pretrained("lllyasviel/Annotators")
    openpose.to(device)
    depth.to(device)
    return depth, openpose

# ---------------------- MAIN ---------------------- #

def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(data_dir)
    print("Found classes:", classes)

    pipe = build_pipeline(args)
    depth_annotator, pose_annotator = build_annotators(args.device)

    generated_counts: Dict[str, int] = {cls: 0 for cls in classes}

    # collect files per class
    per_class_files: Dict[str, List[Path]] = {}
    for cls in classes:
        cls_dir = data_dir / cls
        per_class_files[cls] = sorted(
            [p for p in cls_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

    # one class at a time (slow, clear)
    for cls in classes:
        target = args.per_class
        if target <= 0:
            continue

        files = per_class_files[cls]
        if not files:
            print(f"[WARN] No images for class {cls}, skipping.")
            continue

        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Generating for class {cls} (target {target}) ===")
        pbar = tqdm(total=target, desc=f"class {cls}")

        idx = 0
        while generated_counts[cls] < target and idx < len(files):
            real_path = files[idx]
            idx += 1

            try:
                real_img = Image.open(real_path).convert("RGB")
            except Exception as exc:
                print(f"[WARN] Failed to open {real_path}: {exc}")
                continue

            # annotators from real image
            with torch.no_grad():
                depth_img = depth_annotator(real_img)
                pose_img = pose_annotator(real_img)

            prompt = STATEFARM_PROMPTS.get(
                cls,
                "side view of a driver inside a car, realistic photo from passenger seat",
            )

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=[depth_img, pose_img],  # depth, pose
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    height=args.height,
                    width=args.width,
                )

            gen_img = result.images[0]
            out_name = f"{cls}_synthetic_{generated_counts[cls]:03d}.png"
            gen_img.save(out_dir / out_name)

            generated_counts[cls] += 1
            pbar.update(1)

        pbar.close()

        if generated_counts[cls] < target:
            print(
                f"[INFO] Only generated {generated_counts[cls]} images for {cls} "
                f"(ran out of real images or had errors)."
            )

    print("\nDone. Generated per class:")
    for cls in classes:
        print(f"  {cls}: {generated_counts[cls]} images")


if __name__ == "__main__":
    main()
