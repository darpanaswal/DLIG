# models/backends/__init__.py
"""
Backend factory. Selects the ModelBackend implementation for a loaded model.

Usage:
    backend = build_backend(model, tokenizer, family="auto")  # or "dream"/"diffugpt"
"""

from .base import ModelBackend
from .dream import DreamBackend
from .diffugpt import DiffuGPTBackend


def detect_family(model) -> str:
    """
    Best-effort family detection from module structure.
      - DiffuGPT wrapper:   has .denoise_model + .embed_tokens
      - plain GPT-2:        has .transformer with .wte/.h
      - Dream:              has .model with .layers (+ .rotary_emb)
    """
    if hasattr(model, "denoise_model") and hasattr(model, "embed_tokens"):
        return "diffugpt"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "diffugpt"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "dream"
    raise ValueError(
        "Could not auto-detect model family. Pass family= explicitly "
        "('dream' or 'diffugpt')."
    )


def build_backend(model, tokenizer, family: str = "auto", **kwargs) -> ModelBackend:
    """
    family: 'auto' | 'dream' | 'diffugpt'.
    kwargs forwarded to the backend (e.g. shift=, logits_temp=, topp_temp= for DiffuGPT).
    """
    if family == "auto":
        family = detect_family(model)

    if family == "dream":
        return DreamBackend(model, tokenizer)
    if family == "diffugpt":
        return DiffuGPTBackend(model, tokenizer, **kwargs)
    raise ValueError(f"Unknown backend family: {family!r}")


__all__ = ["ModelBackend", "DreamBackend", "DiffuGPTBackend",
           "build_backend", "detect_family"]