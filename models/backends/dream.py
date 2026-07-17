# models/backends/dream.py
"""
Dream backend. Wraps the EXISTING Dream-specific code paths verbatim so behavior
is identical to the pre-refactor pipeline. Nothing here is new logic; it is the
hardcoded Dream internals from dlig_attribution.py / hook_manager.py / model_manager.py
relocated behind the ModelBackend interface.

Dream specifics:
  - decoder stack:  model.model.layers      (DreamBaseModel)
  - positional:     RoPE via base.rotary_emb, position_embeddings shared per layer
  - final norm:     model.model.norm
  - lm_head:        model.lm_head
  - shift:          Qwen-inherited shifted readout (logits at i predict token i+1)
  - sampler:        HF diffusion_generate / _sample via GradientEnabledModel
"""

from typing import Callable, Optional
import torch

from .base import ModelBackend


class DreamBackend(ModelBackend):

    @property
    def family(self) -> str:
        return "dream"

    @property
    def predicts_shifted(self) -> bool:
        # Dream predicts masked tokens in a shifted manner (Qwen2.5 init):
        # logits at position i correspond to token i+1.
        return True

    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def get_layer_module(self, layer_spec: str) -> torch.nn.Module:
        if isinstance(layer_spec, str) and layer_spec.isdigit():
            return self.model.model.layers[int(layer_spec)]
        if layer_spec == "embed_tokens":
            return self.model.model.embed_tokens
        raise ValueError(f"Unsupported layer spec: {layer_spec}")

    def suffix_forward(self, hidden_states: torch.Tensor, start_layer_idx: int) -> torch.Tensor:
        """
        Verbatim port of the original _suffix_forward. Mirrors DreamBaseModel.forward
        for the no-cache, no-mask, is_causal=False regime DLIG runs in.
        """
        base = self.model.model  # DreamBaseModel
        B, S, H = hidden_states.shape
        device = hidden_states.device

        # position_ids = arange(S); RoPE position_embeddings shared across layers
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        position_embeddings = base.rotary_emb(hidden_states, position_ids)

        for decoder_layer in base.layers[start_layer_idx:]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        hidden_states = base.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)  # [B, S, |V|]
        return logits

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        try:
            out = self.model(input_ids=input_ids, attention_mask=None, use_cache=False)
        except TypeError:
            out = self.model(input_ids=input_ids, attention_mask=None)
        return out.logits

    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        steps: int,
        record_hook: Callable,
    ) -> torch.Tensor:
        """
        Uses Dream's HF diffusion sampler via the existing GradientEnabledModel
        wrapper. record_hook is adapted to Dream's (step, x, logits) hook signature.
        """
        # Imported here to avoid a hard dependency when running DiffuGPT-only.
        from models.model_manager import GradientEnabledModel

        grad_model = GradientEnabledModel(self.model)

        def _logits_hook(step, x, logits):
            record_hook(step, x, logits)
            return logits

        return grad_model.diffusion_generate_with_grad(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=steps,
            generation_logits_hook_func=_logits_hook,
        )