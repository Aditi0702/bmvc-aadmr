#!/usr/bin/env python3
"""Utility wrapper around Hugging Face VLM checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVisionText2Text
except ImportError as exc:  # pragma: no cover - transformers version mismatch
    AutoModelForVisionText2Text = None  # type: ignore[assignment]
    _VISION_IMPORT_ERROR = exc
else:
    _VISION_IMPORT_ERROR = None

try:
    from transformers import PaliGemmaForConditionalGeneration
except ImportError:  # pragma: no cover - optional extra
    PaliGemmaForConditionalGeneration = None  # type: ignore[misc]


def _ensure_vision_text_support() -> None:
    if AutoModelForVisionText2Text is None:
        raise ImportError(
            "transformers>=4.39 is required for AutoModelForVisionText2Text "
            f"(original error: {_VISION_IMPORT_ERROR})"
        )


def _resolve_device(spec: str | None) -> torch.device:
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _resolve_dtype(spec: str | None):
    if spec is None or spec.lower() == "auto":
        return None
    spec = spec.lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if spec not in mapping:
        raise ValueError(f"Unsupported dtype '{spec}'. Choose from {sorted(mapping)} or 'auto'.")
    return mapping[spec]


@dataclass
class VisionLanguageModel:
    """Thin wrapper that bundles a processor and model for text generation."""

    model_id: str
    processor: Any
    model: Any
    device: torch.device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        torch_dtype: str | None = "bf16",
        device: str | None = "auto",
        **model_kwargs,
    ) -> "VisionLanguageModel":
        """
        Build the processor/model pair from a Hugging Face repository.

        Parameters
        ----------
        model_id:
            Hugging Face repository id, e.g. ``google/paligemma-3b-pt-224`` or ``llava-hf/llava-1.5-7b-hf``.
        torch_dtype:
            Optional dtype override, e.g. ``bf16``, ``fp16``, ``fp32``. ``auto`` defers to the checkpoint default.
        device:
            Torch device string. ``auto`` selects CUDA when available, otherwise CPU.
        model_kwargs:
            Extra keyword arguments forwarded to ``from_pretrained`` (e.g., ``revision``, ``cache_dir``).
        """

        processor = AutoProcessor.from_pretrained(model_id)
        dtype = _resolve_dtype(torch_dtype)
        target_device = _resolve_device(device)

        model = _load_hf_model(model_id, torch_dtype=dtype, **model_kwargs)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(target_device)

        return cls(model_id=model_id, processor=processor, model=model, device=target_device)

    def generate(
        self,
        image,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        additional_generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run the VLM on a single image + text prompt and return the decoded string response.
        """

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        if additional_generate_kwargs:
            generate_kwargs.update(additional_generate_kwargs)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)

        return self._decode(output[0]).strip()

    def _decode(self, sequence) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Processor does not expose a tokenizer; cannot decode generation. "
                "Update transformers or provide a custom decode hook."
            )
        return tokenizer.decode(sequence, skip_special_tokens=True)


def _load_hf_model(model_id: str, *, torch_dtype=None, **model_kwargs):
    lower_id = model_id.lower()
    if "paligemma" in lower_id:
        if PaliGemmaForConditionalGeneration is None:
            raise ImportError(
                "PaliGemma checkpoints require transformers>=4.40 with the paligemma extra installed."
            )
        return PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, **model_kwargs
        )

    _ensure_vision_text_support()
    return AutoModelForVisionText2Text.from_pretrained(model_id, torch_dtype=torch_dtype, **model_kwargs)

