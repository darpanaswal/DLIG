# models/model_manager.py
"""
Model loading and management utilities.
"""

import os
import torch
from utils.config import DREAM_PATH, GPT_PATH
from transformers import AutoModel, AutoTokenizer

# NOTE: TF32 (torch.backends.cuda.matmul.allow_tf32) was tried here as a
# "free" Ampere+ speedup but reverted -- verified on a real checkpoint
# (scripts/verify_batching.py) that it corrupts batched generation by up to
# ~0.8 in logit space relative to single-example generation (24-layer GPT-2
# accumulates TF32's reduced mantissa precision into much more than the
# claimed "negligible" impact). With TF32 off, batched vs single-example
# matches to ~1e-3, ordinary fp32 GPU noise. Precision matters more than
# speed here -- do not re-enable without re-running that verification.

class ModelManager:
    def __init__(self, family="dream", device_map="auto", torch_dtype="float32", model_path=None):
        """
        family: 'dream' | 'diffugpt'. Controls how the model is loaded and sets the path.
                'dream'   -> Uses DREAM_PATH. AutoModel.from_pretrained (trust_remote_code).
                'diffugpt'-> Uses GPT_PATH. Plain GPT2LMHeadModel with full attention bias-patch.
        model_path: optional override, e.g. a fine-tuned checkpoint dir (such as a
                    diffugpt family model saved outside GPT_PATH). Loading logic is
                    still selected by `family`; only the on-disk path changes.
        """
        self.family = family.lower()
        default_path = str(DREAM_PATH) if self.family == "dream" else str(GPT_PATH)
        self.model_path = str(model_path) if model_path is not None else default_path
        self.device_map = device_map
        
        # Convert string dtype to torch dtype
        if isinstance(torch_dtype, str):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype

        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer for diffusion attribution."""
        print(f"Loading {self.family} model from {self.model_path}...")

        if self.family == "diffugpt":
            # The DiffuGPT checkpoint's tokenizer_config.json declares
            # tokenizer_class "MaskTokenWrapper", a class that was never
            # shipped with the checkpoint (and doesn't exist in transformers
            # or this repo). The underlying files (vocab.json, merges.txt,
            # tokenizer.json) are plain GPT-2 BPE, so load with the concrete
            # GPT2 tokenizer class instead of AutoTokenizer.
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                self.model_path, local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, local_files_only=True
            )

        if self.family == "diffugpt":
            self.model = self._load_diffugpt()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                local_files_only=True
            ).eval()

        print("Model and tokenizer loaded successfully!")
        return self.model, self.tokenizer

    def _load_diffugpt(self):
        """
        Load a DiffuGPT checkpoint into a GPT2LMHeadModel configured for FULL
        bidirectional attention.

        The checkpoint is HKUNLP's DiscreteDiffusionModel wrapper, so its state-dict
        keys are prefixed and the embedding is split out:
            denoise_model.{h.*, wpe, ln_f}   <- GPT2Model body (wte was deleted)
            embed_tokens.weight              <- the token embedding (separated)
            lm_head.weight                   <- output head
        A plain GPT2LMHeadModel.from_pretrained CANNOT match these keys (it expects
        transformer.h.*, transformer.wte, etc.) and silently random-inits everything.
        So we instantiate the architecture from config and remap keys ourselves.

        Mapping onto GPT2LMHeadModel:
            denoise_model.h.*    -> transformer.h.*
            denoise_model.wpe.*  -> transformer.wpe.*
            denoise_model.ln_f.* -> transformer.ln_f.*
            embed_tokens.weight  -> transformer.wte.weight   (and lm_head.weight)
            lm_head.weight       -> lm_head.weight

        Bidirectional recipe (validated, transformers 4.44.x): eager attention +
        replace_attention_mask() + attn.bias.fill_(True) + 4D zero mask per block.
        """
        import glob
        import utils.attention_patch as attention_patch  # provided in project; used as-is
        attention_patch.replace_attention_mask()

        from transformers import GPT2LMHeadModel, GPT2Config
        from safetensors.torch import load_file

        cfg = GPT2Config.from_pretrained(self.model_path, local_files_only=True)
        # Build the model skeleton (eager so the bias buffer gates causality).
        cfg._attn_implementation = "eager"
        model = GPT2LMHeadModel(cfg)

        # Load the wrapper state dict. Prefer the canonical HF filename
        # (pytorch_model.bin / model.safetensors) over a bare glob: a
        # Trainer-saved checkpoint dir also contains training_args.bin,
        # optimizer.pt, scheduler.pt, rng_state.pth, etc., and glob order is
        # not guaranteed to put the actual weights file first.
        NON_WEIGHT_BIN_NAMES = {
            "training_args.bin", "optimizer.pt", "scheduler.pt",
            "rng_state.pth", "trainer_state.json",
        }
        st_path = os.path.join(self.model_path, "model.safetensors")
        bn_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.isfile(st_path):
            raw = load_file(st_path)
        elif os.path.isfile(bn_path):
            raw = torch.load(bn_path, map_location="cpu")
        else:
            st = glob.glob(os.path.join(self.model_path, "*.safetensors"))
            bn = [f for f in glob.glob(os.path.join(self.model_path, "*.bin"))
                  if os.path.basename(f) not in NON_WEIGHT_BIN_NAMES]
            if st:
                raw = load_file(st[0])
            elif bn:
                raw = torch.load(bn[0], map_location="cpu")
            else:
                raise FileNotFoundError(f"No .safetensors/.bin in {self.model_path}")
        if not isinstance(raw, dict):
            raise TypeError(
                f"Loaded weights file did not unpickle to a state dict "
                f"(got {type(raw).__name__}); the checkpoint dir may contain "
                f"a non-weight .bin file that got picked up instead."
            )

        # Vocab fix: DiffuGPT resized embeddings to add a mask token (HKUNLP:
        # resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)), but the
        # config's vocab_size was left at GPT-2's original 50257. The checkpoint's
        # wte/lm_head therefore have MORE rows than the freshly-built model. Resize
        # the model to the checkpoint's embedding size so the load matches exactly.
        ckpt_vocab = None
        for kk in ("embed_tokens.weight", "denoise_model.wte.weight",
                   "transformer.wte.weight", "lm_head.weight"):
            if kk in raw:
                ckpt_vocab = raw[kk].shape[0]
                break
        if ckpt_vocab is not None and ckpt_vocab != model.get_input_embeddings().weight.shape[0]:
            print(f"[INFO] Resizing embeddings {model.get_input_embeddings().weight.shape[0]} "
                  f"-> {ckpt_vocab} to match DiffuGPT checkpoint (mask-token resize).")
            model.resize_token_embeddings(ckpt_vocab)

        # Remap keys: wrapper -> GPT2LMHeadModel.
        remapped = {}
        for k, v in raw.items():
            if k.startswith("denoise_model."):
                remapped["transformer." + k[len("denoise_model."):]] = v
            elif k == "embed_tokens.weight":
                remapped["transformer.wte.weight"] = v
                # GPT-2 ties wte and lm_head; set head too unless an explicit head exists.
                remapped.setdefault("lm_head.weight", v)
            elif k == "lm_head.weight":
                remapped["lm_head.weight"] = v
            else:
                # e.g. embed_tokens.* variants, or already-correct keys
                remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # wte may be reported missing if only present via tie; verify the load really
        # populated the body. If transformer.h.0 weights are missing, the remap failed.
        critical = [m for m in missing if m.startswith("transformer.h.")
                    or m in ("transformer.wte.weight", "transformer.wpe.weight",
                             "transformer.ln_f.weight", "lm_head.weight")]
        if critical:
            raise RuntimeError(
                f"DiffuGPT load failed to populate critical weights: {critical[:8]}... "
                f"(total {len(critical)}). Key remap is wrong; inspect checkpoint keys."
            )
        if unexpected:
            print(f"[WARN] {len(unexpected)} unexpected keys ignored "
                  f"(e.g. {unexpected[:3]}).")
        print(f"[INFO] DiffuGPT weights loaded: {len(remapped)-len(unexpected)} tensors "
              f"mapped, {len(missing)} missing (non-critical).")

        model = model.eval()
        for block in model.transformer.h:
            attn = block.attn
            if hasattr(attn, "bias") and isinstance(attn.bias, torch.Tensor):
                attn.bias.fill_(True)            # disable buffer causal mask

        model = model.to(self.torch_dtype)
        if self.device_map and self.device_map != "auto":
            model = model.to(self.device_map)
        elif torch.cuda.is_available():
            model = model.to("cuda")

        return model

    def get_model_device(self):
        """Get the device of the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        return next(self.model.parameters()).device


class GradientEnabledModel:
    """Wrapper for enabling gradient computation during the diffusion generation loop."""

    def __init__(self, model):
        self.model = model

    def diffusion_generate_with_grad(self, input_ids, **kwargs):
        """
        Generates output while allowing DLIG to compute gradients at each timestep.

        Determinism & stability constraint:
        - Keep model in eval mode (DO NOT switch to train).
        - Enable gradients via torch.enable_grad().
        """
        # Always enforce eval mode before and after
        self.model.eval()

        with torch.enable_grad():
            try:
                generation_logits_hook_func = kwargs.pop(
                    "generation_logits_hook_func",
                    lambda step, x, logits: logits
                )

                generation_tokens_hook_func = kwargs.pop(
                    "generation_tokens_hook_func",
                    lambda step, x, logits: x
                )

                generation_config = self.model._prepare_generation_config(
                    kwargs.get('generation_config'),
                    **{k: v for k, v in kwargs.items() if k != 'generation_config'}
                )

                attention_mask = kwargs.get("attention_mask")
                device = input_ids.device
                self.model._prepare_special_tokens(generation_config, device=device)

                input_ids_length = input_ids.shape[-1]
                has_default_max_length = (
                    kwargs.get("max_length") is None and
                    generation_config.max_length is not None
                )

                generation_config = self.model._prepare_generated_length(
                    generation_config=generation_config,
                    has_default_max_length=has_default_max_length,
                    input_ids_length=input_ids_length,
                )

                result = self.model._sample(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    generation_tokens_hook_func=generation_tokens_hook_func,
                    generation_logits_hook_func=generation_logits_hook_func
                )
                return result
            finally:
                self.model.eval()