# Synthetic StateFarm Generation

This is the main folder for synthetic data generation and outputs.

## Dataset outputs and source scripts
Typical output folders live under `synthetic/dataset/`. The mapping below shows
which script generates each folder.

- `synthetic/dataset/statefarm_controlnet_shared7/`
  - Script: `synthetic/scripts/generate_statefarm_controlnet.py`
  - Notes: ControlNet (pose + canny) SDXL pipeline.
- `synthetic/dataset/sd15_controlnet_depth_pose_canny/`
  - Script: `synthetic/scripts/generate_statefarm_controlnet_depth_pose_canny.py`
  - Notes: SD1.5 ControlNet (depth + pose + canny).
- `synthetic/dataset/sdxl_outputs/`
  - Script: `synthetic/scripts/generate_statefarm_sdxl_controlnet_lora.py`
  - Notes: SDXL ControlNet + LoRA pipeline. SDXL model weights come from an earlier company project.
    Current `sdxl_outputs` are NIR-style images (not RGB). Future work: finetune
    SDXL on StateFarm RGB data, then regenerate RGB outputs.
- `synthetic/dataset/debug_shared7/`
  - Script: `synthetic/scripts/generate_statefarm_controlnet.py`
  - Notes: Smaller debug run for shared-7 classes.

Utility scripts for producing photorealistic driver-action augmentations using
Stable Diffusion + ControlNet.

## Requirements

```bash
pip install diffusers transformers accelerate safetensors controlnet-aux opencv-python pillow
pip install xformers     # optional but speeds up inference
```

Log in to Hugging Face once if the chosen base model is gated:

```bash
huggingface-cli login
```

## ControlNet Workflow

`generate_statefarm_controlnet.py` reads the original StateFarm CSV plus images,
derives OpenPose keypoints and Canny edge maps, then feeds them into
`SG161222/Realistic_Vision_V6.0_B2_noVAE` (previous runs used the B1 checkpoint) 
replaced with 'stabilityai/stable-diffusion-xl-base-1.0' 

with the SDXL ControlNet adapters (`diffusers/controlnet-openpose-sdxl-1.0` +
`diffusers/controlnet-canny-sdxl-1.0`). B2 is slightly more demographically
balanced; if you want to reproduce the older B1 + SD1.5 ControlNet results, pass
`--base-model SG161222/Realistic_Vision_V6.0_B1_noVAE --openpose-controlnet lllyasviel/control_v11p_sd15_openpose --canny-controlnet lllyasviel/control_v11p_sd15_canny`.

Example (shared-7 classes):

```bash
python synthetic/generate_statefarm_controlnet.py \
  --data-dir Dataset/statefarm/imgs/train \
  --labels-csv Dataset/statefarm/driver_imgs_list_shared7.csv \
  --output-dir synthetic/statefarm_controlnet_shared7 \
  --classes c0 c6 c4 c2 c3 c1 c9 \
  --num-augments 4 \
  --resolution 512 \
  --device cuda:0
```

For the full 10-class dataset just point `--labels-csv` to
`Dataset/statefarm/driver_imgs_list.csv` and omit `--classes` so every class is
included. The script saves intermediate hints (`*_pose.png`, `*_canny.png`) and
synthesized frames (`*_synthetic_*.png`) under the requested output directory,
grouped per class. These images can then be merged into a new CSV and used for
LoRA/projector finetuning.

## Flux workflow
`synthetic/scripts/generate_statefarm_flux.py` runs the Flux model. It requires
~40 GB of GPU VRAM to generate images, so plan runs on high-memory GPUs.



## depth + pose finetune lora
```bash
python synthetic/generate_statefarm_sdxl_controlnet_lora.py   --data-dir Dataset/statefarm/imgs/train   --output-dir synthetic/sdxl_outputs   --base-model incabingenai/models/stable-diffusion-xl-base-1.0   --depth-model incabingenai/models/controlnet-depth-sdxl-1.0   --pose-model incabingenai/models/controlnet-openpose-sdxl-1.0   --vae incabingenai/models/sdxl-vae-fp16-fix   --lora incabingenai/models/lora   --lora-scale 0.8   --per-class 5   --device cuda:6'
```

#synthetic/sdxl_outputs 




# depth +pose+canny  FROM BASE MODEL  models--SG161222--Realistic_Vision_V6.0_B1_noVAE
""""

```bash
python synthetic/generate_statefarm_controlnet_depth_pose_canny.py   --data-dir Dataset/statefarm/imgs/train   --labels-csv Dataset/statefarm/driver_imgs_list.csv   --output-dir synthetic/sd15_controlnet_depth_pose_canny   --base-model SG161222/Realistic_Vision_V6.0_B1_noVAE   --openpose-controlnet lllyasviel/control_v11p_sd15_openpose   --canny-controlnet lllyasviel/control_v11p_sd15_canny   --depth-controlnet lllyasviel/control_v11f1p_sd15_depth   --openpose-detector lllyasviel/ControlNet   --device cuda:6   --resolution 512   --inference-steps 28   --images-per-class 5
```
output
synthetic/sd15_controlnet_depth_pose_canny

"""








##1.SDXL ControlNet + LoRA Pipeline
Base Model (SDXL)

incabingenai/models/stable-diffusion-xl-base-1.0

VAE

incabingenai/models/sdxl-vae-fp16-fix

ControlNets (SDXL versions)

Depth ControlNet (SDXL)

incabingenai/models/controlnet-depth-sdxl-1.0

OpenPose ControlNet (SDXL)

incabingenai/models/controlnet-openpose-sdxl-1.0

LoRA Adapter

incabingenai/models/lora




##2. SD15 Pipeline (Depth + Pose + Canny)
Base Model (SD 1.5)

SG161222/Realistic_Vision_V6.0_B1_noVAE

ControlNets (SD1.5 versions)

OpenPose ControlNet

lllyasviel/control_v11p_sd15_openpose

Canny ControlNet

lllyasviel/control_v11p_sd15_canny

Depth ControlNet

lllyasviel/control_v11f1p_sd15_depth

Annotators (controlnet_aux)

OpenPose detector

lllyasviel/ControlNet

Depth detector (MiDaS)

lllyasviel/Annotators ← Midas model auto-resolved inside package



