# models/backends/base.py
"""
ModelBackend: abstraction isolating every model-family-specific assumption that
DLIG depends on, so the attribution math is written once and runs on any masked
diffusion LM (Dream, DiffuGPT, ...).

WHY THIS EXISTS
---------------
DLIG's correctness rests on a handful of model internals that differ by family:

  1. decoder stack location        Dream: model.model.layers
                                    DiffuGPT(GPT-2): model.transformer.h
  2. positional scheme             Dream: RoPE (per-layer position_embeddings)
                                    DiffuGPT: learned absolute (wpe, baked at embed)
  3. final norm + lm_head          Dream: model.model.norm + model.lm_head
                                    DiffuGPT: transformer.ln_f + lm_head
  4. readout shift convention      whether logits at position i predict token i
                                    (in-place) or token i+1 (shifted)
  5. mask token id                 Dream has one natively; GPT-2 needs one designated
  6. trajectory sampler            Dream: HF diffusion_generate / _sample
                                    DiffuGPT: custom random-reveal loop (HKUNLP)

A backend supplies all six. DLIG, verify_completeness, and contrastive_attribution
call ONLY the backend for these, never the raw model.

CONTRACT FOR suffix_forward (the load-bearing method)
-----------------------------------------------------
Given hidden states that are the OUTPUT of decoder layer `start_layer_idx - 1`
(or the embeddings when start_layer_idx == 0), replay
    layers[start_layer_idx:] -> final_norm -> lm_head
and return logits [B, S, |V|] over ALL positions, full (non-causal) attention,
no KV cache. This must match what a full forward would produce at those positions,
which is exactly what the completeness check (A: hook transparency) verifies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
import torch


class ModelBackend(ABC):
    """Family-specific adapter. One instance wraps one loaded model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # -- identity ---------------------------------------------------------- #
    @property
    @abstractmethod
    def family(self) -> str:
        """Short tag, e.g. 'dream' or 'diffugpt'. Used in logs/output records."""

    @property
    @abstractmethod
    def predicts_shifted(self) -> bool:
        """
        True  if logits at position i predict token i+1 (Qwen/Dream, shift=True).
        False if logits at position i predict token i   (in-place).

        Drives the self-generated scorer's index alignment. Completeness is
        shift-agnostic, so this is the ONE thing the completeness harness cannot
        catch -- it must be set from the checkpoint's training config and
        confirmed with a token-level decode (see verify_shift()).
        """

    # -- structure --------------------------------------------------------- #
    @abstractmethod
    def num_layers(self) -> int:
        """Number of decoder layers in the stack."""

    @abstractmethod
    def get_layer_module(self, layer_spec: str) -> torch.nn.Module:
        """
        Resolve a layer spec to the nn.Module whose OUTPUT is the activation we
        attribute. Specs: a digit string '0'..'L-1', or 'embed_tokens'.

        For a digit spec, this is the decoder block; the forward hook fires after
        it, so its output is the layer activation a^(l).
        For 'embed_tokens', this is the token embedding module; its output is the
        embedding activation (suffix then starts at layer 0).
        """

    def mask_token_id(self) -> int:
        """Mask token id for the prompt-masked null baseline a'."""
        mid = self.tokenizer.mask_token_id
        if mid is None:
            mid = self.tokenizer.pad_token_id
        if mid is None:
            mid = 0
        return int(mid)

    # -- partial forward (suffix replay) ----------------------------------- #
    @abstractmethod
    def suffix_forward(
        self,
        hidden_states: torch.Tensor,   # [B, S, H] = output of layer start-1 (or embeds)
        start_layer_idx: int,
    ) -> torch.Tensor:                 # [B, S, |V|] logits, all positions
        """Replay layers[start_layer_idx:] -> final_norm -> lm_head. See contract above."""

    def resolve_suffix_start(self, layer_spec: str) -> int:
        """
        Index of the first decoder layer that must re-run on interpolated acts.
        Hook fires AFTER the hooked module:
          digit l       -> interpolated act is layers[l] output -> suffix at l+1
          'embed_tokens'-> interpolated act is embeddings        -> suffix at 0
        """
        if layer_spec == "embed_tokens":
            return 0
        if isinstance(layer_spec, str) and layer_spec.isdigit():
            return int(layer_spec) + 1
        if isinstance(layer_spec, int):
            return layer_spec + 1
        raise ValueError(f"Cannot resolve suffix start for layer spec: {layer_spec!r}")

    # -- full forward (for activation capture / plain scoring) ------------- #
    @abstractmethod
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward over input_ids, full (non-causal) attention, no KV cache.
        Returns logits [B, S, |V|]. Used by the plain-score path and as the
        forward whose intermediate activations the hooks capture.
        """

    # -- trajectory generation -------------------------------------------- #
    @abstractmethod
    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        steps: int,
        record_hook: Callable,         # record_hook(step:int, x:Tensor, logits:Tensor)
    ) -> torch.Tensor:
        """
        Run the denoising loop, calling record_hook(step, x_t, logits) at each step
        so the caller can stash x_t per step (the TrajRecorder pattern). Returns the
        final token ids. Implementations must produce x_t in the SAME [B, S] layout
        the rest of the pipeline assumes (prompt occupies [:, :L]).
        """