# models/backends/diffugpt.py
"""
DiffuGPT (GPT-2 backbone) backend.

Architecture facts (from HKUNLP/DiffuLLaMA model.py):
  - The diffusion wrapper splits embeddings out of the transformer:
        embed_tokens = model.transformer.wte      (and wte is deleted from the stack)
        denoise_model = model.transformer          (consumes inputs_embeds)
        lm_head      = model.lm_head
    so the decoder forward takes inputs_embeds, NOT input_ids.
  - Causal mask removed: every block's attn.bias is filled True -> full attention.
  - Positions: GPT-2 learned absolute (wpe), added at the embedding stage. There is
    NO RoPE and NO per-layer position_embeddings.
  - Final norm: transformer.ln_f. lm_head on top.
  - Sampler: random-reveal schedule, p_to_x0 = 1/(t+1), top-p filtered sampling.

This backend supports two ways the model may be exposed:
  (a) The HKUNLP DiscreteDiffusionModel wrapper, with .embed_tokens / .denoise_model
      (a GPT2Model) / .lm_head / .get_embeds().
  (b) A plain HF GPT2LMHeadModel (.transformer.{wte,wpe,h,ln_f}, .lm_head), which we
      drive directly. We detect which one we have and adapt.

SHIFT: DiffuGPT-medium is trained with shift=True (README: gpt2_full_ddm-sft.yaml).
So predicts_shifted defaults True here, matching Dream's readout convention. This is
config-dependent; verify_shift() in the harness must confirm it by decoding, since
the completeness check is shift-agnostic.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from .base import ModelBackend


def top_p_logits(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Verbatim port of HKUNLP top_p_logits (nucleus filtering)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    # shift right: keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


class DiffuGPTBackend(ModelBackend):

    def __init__(self, model, tokenizer, shift: bool = True,
                 logits_temp: float = 1.0, topp_temp: float = 0.9):
        super().__init__(model, tokenizer)
        self._shift = bool(shift)
        self.logits_temp = float(logits_temp)
        self.topp_temp = float(topp_temp)
        self._wrapped, self.wte, self.wpe, self.blocks, self.ln_f, self.lm_head = \
            self._bind_modules(model)

    # -- module binding: handle wrapper vs plain HF GPT-2 ------------------ #
    @staticmethod
    def _bind_modules(model):
        """
        Returns (is_wrapped, wte, wpe, blocks, ln_f, lm_head).

        HKUNLP wrapper: model.embed_tokens, model.denoise_model (GPT2Model w/ wte
        deleted), model.lm_head. The GPT2Model still owns wpe / h / ln_f.
        Plain HF: model.transformer.{wte,wpe,h,ln_f}, model.lm_head.
        """
        # (a) HKUNLP DiscreteDiffusionModel
        if hasattr(model, "denoise_model") and hasattr(model, "embed_tokens"):
            gpt2 = model.denoise_model            # GPT2Model
            wte = model.embed_tokens              # separated embedding
            wpe = gpt2.wpe
            blocks = gpt2.h
            ln_f = gpt2.ln_f
            lm_head = model.lm_head
            return True, wte, wpe, blocks, ln_f, lm_head

        # (b) plain HF GPT2LMHeadModel
        if hasattr(model, "transformer"):
            t = model.transformer
            return False, t.wte, t.wpe, t.h, t.ln_f, model.lm_head

        raise ValueError(
            "DiffuGPTBackend: model is neither the HKUNLP wrapper "
            "(.denoise_model/.embed_tokens) nor a GPT2LMHeadModel (.transformer)."
        )

    @property
    def family(self) -> str:
        return "diffugpt"

    @property
    def predicts_shifted(self) -> bool:
        return self._shift

    def num_layers(self) -> int:
        return len(self.blocks)

    def get_layer_module(self, layer_spec: str) -> torch.nn.Module:
        if isinstance(layer_spec, str) and layer_spec.isdigit():
            return self.blocks[int(layer_spec)]
        if layer_spec == "embed_tokens":
            return self.wte
        raise ValueError(f"Unsupported layer spec: {layer_spec}")

    # -- embedding (token + absolute position) ----------------------------- #
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        GPT-2 input embedding: wte(token) + wpe(position). Positions are baked in
        HERE (not per-layer), which is the key contrast with Dream's RoPE.
        """
        B, S = input_ids.shape
        device = input_ids.device
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        return self.wte(input_ids) + self.wpe(pos)

    # -- GPT-2 block call: full attention, no cache ------------------------ #
    @staticmethod
    def _full_attn_mask(B: int, S: int, device, dtype) -> torch.Tensor:
        """
        4D all-visible additive mask [B, 1, S, S] of zeros. Combined with the
        attention_patch (GPT2Model.forward uses a 4D mask verbatim) and bias.fill_(True)
        on each block (disables the buffer causal mask), this yields FULL bidirectional
        attention. Validated: row attends uniformly across all positions, not lower-tri.
        """
        return torch.zeros(B, 1, S, S, device=device, dtype=dtype)

    def _call_block(self, block, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Call one GPT2Block (4.44 signature) with a 4D all-visible mask so attention
        is bidirectional. Requires: model loaded with attn_implementation='eager',
        replace_attention_mask() applied, and block.attn.bias filled True (all done
        at load in ModelManager._load_diffugpt). Hidden states are element 0.
        """
        B, S, _ = hidden_states.shape
        mask = self._full_attn_mask(B, S, hidden_states.device, hidden_states.dtype)
        out = block(
            hidden_states,
            layer_past=None,
            attention_mask=mask,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def suffix_forward(self, hidden_states: torch.Tensor, start_layer_idx: int) -> torch.Tensor:
        """
        Replay blocks[start_layer_idx:] -> ln_f -> lm_head.

        NOTE: positions are already inside `hidden_states` (added at embed). No
        position argument is threaded through the blocks, unlike Dream's RoPE path.
        """
        for block in self.blocks[start_layer_idx:]:
            hidden_states = self._call_block(block, hidden_states)
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)   # [B, S, |V|]
        return logits

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward from token ids: embed (wte+wpe) -> all blocks -> ln_f -> lm_head.
        Equivalent to suffix_forward over the embeddings, kept explicit for the
        activation-capture path (hooks fire on real block modules during this call).
        """
        hidden_states = self._embed(input_ids)
        for block in self.blocks:
            hidden_states = self._call_block(block, hidden_states)
        hidden_states = self.ln_f(hidden_states)
        return self.lm_head(hidden_states)

    # -- trajectory sampler (ported from HKUNLP generate_samples) ---------- #
    @torch.no_grad()
    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        steps: int,
        record_hook: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Random-reveal masked-diffusion sampling, ported from HKUNLP/DiffuLLaMA
        model.py generate_samples. Differences from the original, all to fit the
        DLIG pipeline rather than change the algorithm:
          - We build the canvas as [prompt | max_new_tokens masks] and treat the
            prompt as src (never re-masked), instead of taking a pre-masked x with
            a src_mask. This matches how the rest of the pipeline lays out x_t
            (prompt at [:, :L]).
          - We call record_hook(step, x_t, logits) each step so the caller stores
            the trajectory (TrajRecorder), keying steps 0..steps-1. Pass
            record_hook=None (or omit it) to skip this entirely: the
            .detach().to("cpu").clone() below is a real per-step cost (a CUDA
            sync + host transfer) that every downstream DLIG attribution
            experiment needs (it stores the trajectory for later scoring), but
            a bare accuracy eval that only wants the final x0 should not pay
            it -- this is what made scripts/eval_task.py much slower than the
            official generate_samples, which never leaves the GPU mid-loop.

        Algorithm (unchanged):
          p_to_x0 = 1/(t+1) tokens revealed per step; top-p filtered categorical
          sample for x0; shift handling if self._shift; non-mask positions kept.
        """
        device = next(self.lm_head.parameters()).device
        mask_id = self.mask_token_id()

        prompt = input_ids.to(device)
        B, L = prompt.shape

        # Canvas: prompt followed by masked generation region.
        gen = torch.full((B, max_new_tokens), mask_id, dtype=prompt.dtype, device=device)
        x = torch.cat([prompt, gen], dim=1)        # [B, L + max_new_tokens]

        # src = prompt region; never maskable. maskable = generation region.
        src_mask = torch.zeros_like(x, dtype=torch.bool)
        src_mask[:, :L] = True
        init_maskable_mask = maskable_mask = ~src_mask

        # t = T (first step): all maskable positions are [MASK]
        xt = x.masked_fill(maskable_mask, mask_id)

        def _predict(xt_in):
            # Defensive: any token id fed back to wte must be < vocab_size, else a
            # CUDA index assert fires on the NEXT embed. Clamp guards against stray
            # ids (and makes a bad checkpoint fail loudly here, not deep in cuBLAS).
            vocab = self.wte.num_embeddings
            if (xt_in >= vocab).any() or (xt_in < 0).any():
                raise ValueError(
                    f"Token id out of range for wte (vocab={vocab}); "
                    f"max={int(xt_in.max())}, min={int(xt_in.min())}. "
                    f"Likely a mis-loaded checkpoint or wrong mask_token_id."
                )
            logits = self.forward_logits(xt_in)
            filt = top_p_logits(logits / self.logits_temp, p=self.topp_temp)
            scores = torch.log_softmax(filt, dim=-1)
            x0 = dists.Categorical(logits=scores).sample()
            if self._shift:
                # left-most token replaced anyway; shift predictions right by one
                x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)
            # keep already-revealed (non-maskable) positions as in xt
            x0 = xt_in.masked_scatter(maskable_mask, x0[maskable_mask])
            return logits, x0

        # --- step index 0 (t = T) ---
        logits, x0 = _predict(xt)
        if record_hook is not None:
            record_hook(0, xt.detach().to("cpu").clone(), logits)

        # --- steps t = T-1 .. 1 ---
        # The HKUNLP loop runs diffusion_steps-1 reveal iterations. We index the
        # recorded trajectory 1..steps-1 so downstream target_steps line up with
        # generation step counts.
        rec_idx = 1
        for t in range(steps - 1, 0, -1):
            p_to_x0 = 1.0 / (t + 1)
            masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
            xt = xt.masked_scatter(masked_to_x0, x0[masked_to_x0])
            maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)

            logits, x0 = _predict(xt)
            if record_hook is not None:
                record_hook(rec_idx, xt.detach().to("cpu").clone(), logits)
            rec_idx += 1

        return x0