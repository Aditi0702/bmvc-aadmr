# Vision-Language Model (VLM) Experiments

The `VLM` module lets you run PaLI/Gemma, LLaVA/LLaMA and other Hugging Face
vision-language checkpoints on SAM-DD (or any ImageFolder-style dataset) without
changing the original data layout. It reuses the CSV exported by
`DA/tools/build_samdd_labels.py`, so there is no need to copy or flatten images
again.

> ⚠️ VLM checkpoints are huge (several GB) and typically require a modern GPU
> with >24 GB VRAM. Make sure the `transformers`, `accelerate`, and
> `safetensors` packages are installed in an environment that can download the
> weights from Hugging Face.

## Quick start

Generate the label CSV once (if you have not already):

```bash
python -m DA.tools.build_samdd_labels \
  --data-dir Dataset/'SAM-DD(RGB)' \
  --output Dataset/samdd_labels.csv
```

Run a zero-shot evaluation with PaLI-Gemma:

```bash
python -m VLM.predict \
  --model-id google/paligemma-3b-pt-224 \
  --data-dir Dataset/'SAM-DD(RGB)' \
  --labels-csv Dataset/samdd_labels.csv \
  --device auto \
  --torch-dtype bf16 \
  --limit 128 \
  --save-predictions runs/paligemma_zero_shot.jsonl
```

To try a LLaVA/LLaMA style checkpoint, swap `--model-id`:

```bash
python -m VLM.predict \
  --model-id llava-hf/llava-1.5-7b-hf \
  --data-dir Dataset/'SAM-DD(RGB)' \
  --labels-csv Dataset/samdd_labels.csv
```

The script prints per-sample outputs and a running accuracy summary. JSONL
predictions include the raw text response so you can inspect how the VLM reasons
about each driver pose.

## How it works

* `VLM/models/vlm.py` wraps `transformers.AutoProcessor` plus
  `AutoModelForVisionText2Text` (or `PaliGemmaForConditionalGeneration` for PaLI)
  into a single helper that handles dtype/device placement.
* `VLM/predict.py` reuses the datasets from `DA.data_loader` to get class names
  and file paths, builds a natural-language prompt listing all 10 SAM-DD labels,
  and asks the VLM to respond with the label id.
* Predictions are matched back to the closest label string/number so we can
  compute accuracy without a custom head.

## Next steps

This folder is intentionally lightweight so you can iterate quickly:

1. Add fine-tuning or LoRA adapters using `peft` if zero-shot accuracy is not
   sufficient.
2. Expand the prompt template (or make it sample-specific) to inject more
   context from metadata such as camera view.
3. Connect TensorBoard or Weights & Biases logging if you begin training runs.

File an issue in your notes if you need the VLM loader to support distributed
`device_map` setups or other advanced Hugging Face options. For now it assumes a
single-device load, which keeps the dependency surface small.

