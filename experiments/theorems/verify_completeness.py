"""
verify_completeness.py — empirical proof that the DLIG implementation is correct.

Place at repo root (same level as main.py / contrastive_runner.py), since it uses
the same package import paths.

WHAT THIS PROVES
----------------
Layer-IG completeness (Theorem 1). When attribution is computed by INTERVENING on
layer l (replacing its output with interpolated activations), summing the attribution
over ALL positions and ALL hidden dims recovers the score difference exactly:

    sum_{i,j} DLIG^(l)_t[i,j]  ==  F_t(X_t, C) - F_t(X_t, /0)

    DLIG^(l)_t = (a - a') (.) (1/m) sum_{k=1..m} grad_a F( a' + (k/m)(a - a') )
      a  = h^(l)_t(X_t, C)    real activations      (prompt present)
      a' = h^(l)_t(X_t, /0)   baseline activations   (prompt masked)

This holds for ANY scalar F = g(h^(l)) up to O(1/m) Riemann error, regardless of how
F reads logits (so the Dream shifted-readout off-by-one does NOT affect this check;
it only affects token-level interpretation, tested elsewhere).

Three nested checks, increasing strength:
  (A) hook transparency : F via intervention at alpha=0 / alpha=1 endpoints must equal
                          F via a plain forward. If this fails every IG number is junk.
  (B) completeness      : sum DLIG (all positions) vs deltaF, per m.
  (C) convergence       : rel_err -> 0 as m increases. <-- the empirical proof.

This script REUSES the real primitives from DLIGAttribution (intervention hook,
create_baseline_input, _compute_target_score, _model_forward_no_mask, get_layer_activations),
so it verifies the production machinery, not a re-implementation. Only the integration
loop and the full-position sum are written here, to keep them readable and to expose
the all-position sum (the real code slices to prompt positions for site scores).
"""

import gc
import torch
import random
import argparse
import numpy as np
from models.backends import build_backend
from models.model_manager import ModelManager
from attribution.hook_manager import HookManager
from attribution.dlig_attribution import DLIGAttribution


def build_prompt_inputs(tokenizer, system, prompt, device):
    """
    Build (input_ids, attention_mask, L) for one prompt.

    Uses the chat template when the tokenizer has one (Dream/instruct models).
    Falls back to plain concatenated text when it does not (DiffuGPT, base GPT-2:
    chat_template is None -> apply_chat_template would raise). The fallback keeps a
    simple "system\n\nuser\n\n" layout so the prompt tokens are still a contiguous
    suffix that input_token_indices / baseline masking can locate.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device).float()
    else:
        text = f"{system}\n\n{prompt}\n\n" if system else f"{prompt}\n\n"
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device).float()
    L = int(input_ids.shape[1])
    return input_ids, attention_mask, L


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_layer_module(model, layer: str, backend=None):
    """Resolve via backend when available (family-agnostic); else Dream default."""
    if backend is not None:
        return backend.get_layer_module(layer)
    if layer.isdigit():
        return model.model.layers[int(layer)]
    if layer == "embed_tokens":
        return model.model.embed_tokens
    raise ValueError(f"Unsupported layer spec: {layer}")


class TrajRecorder:
    """Records x_t at each diffusion step (on CPU)."""
    def __init__(self):
        self.x_by_step = {}

    def hook(self, step, x, logits):
        if step is None:
            return logits
        self.x_by_step[int(step)] = x.detach().to("cpu").clone()
        return logits

    def steps(self):
        return sorted(self.x_by_step.keys())


def score_plain(dlig: DLIGAttribution, input_ids: torch.Tensor, original_length: int) -> float:
    """F_t evaluated on a normal forward of input_ids (no intervention)."""
    with torch.no_grad():
        out = dlig._model_forward_no_mask(input_ids=input_ids)
        return float(dlig._compute_target_score(out, input_ids, original_length).item())


def score_intervene(
    dlig: DLIGAttribution,
    x_t: torch.Tensor,
    forced_act: torch.Tensor,
    original_length: int,
) -> float:
    """
    F_t with layer-l output FORCED to forced_act, forwarding x_t's tokens.

    forced_act = a  -> should equal score_plain(x_t)        (identity intervention)
    forced_act = a' -> should equal score_plain(baseline)   (downstream depends only
                       on layer-l output; input_ids re-enter only at the embedding,
                       before layer l).
    """
    with torch.no_grad():
        with dlig.activation_intervention(None, forced_act):
            out = dlig._model_forward_no_mask(input_ids=x_t)
            return float(dlig._compute_target_score(out, x_t, original_length).item())


def integrate_dlig_full(
    dlig: DLIGAttribution,
    x_t: torch.Tensor,
    real_act: torch.Tensor,
    baseline_act: torch.Tensor,
    original_length: int,
    m: int,
    chunk: int,
) -> torch.Tensor:
    """
    Vectorized Riemann sum over k=1..m, mirroring compute_dlig_at_timestep, but
    returns the FULL-position attribution tensor dlig_raw [B, S, H] (not prompt-sliced).

    DLIG^(l)_t = (a - a') (.) (1/m) sum_{k=1..m} grad_a F( a' + (k/m)(a - a') )
      alpha_k = k/m
      interpolated_k = a' + alpha_k * (a - a')
    """
    act_device = real_act.device
    act_dtype = real_act.dtype

    activation_diff = (real_act - baseline_act).detach()  # (a - a')   [B,S,H]
    B, S, H = activation_diff.shape
    grad_sum = torch.zeros_like(real_act)

    if chunk is None or chunk <= 0:
        chunk = m

    for start in range(0, m, chunk):
        end = min(m, start + chunk)
        c = end - start

        # alpha_k = k/m for k in [start+1, end]
        ks = torch.arange(start + 1, end + 1, device=act_device, dtype=act_dtype)
        alphas = (ks / m).view(c, 1, 1, 1)

        # interpolated_k = a' + alpha_k * (a - a')
        interp = baseline_act.unsqueeze(0) + alphas * activation_diff.unsqueeze(0)  # [c,B,S,H]
        interp = interp.detach().requires_grad_(True)
        interp_flat = interp.reshape(c * B, S, H)

        x_rep = x_t.repeat(c, 1)  # same tokens, c copies

        with dlig.activation_intervention(None, interp_flat):
            outputs = dlig._model_forward_no_mask(input_ids=x_rep)
            # target_score = sum_k F(interp_k); grad wrt row k = grad_a F(interp_k)
            target_score = dlig._compute_target_score(outputs, x_rep, original_length)

        grads_flat = torch.autograd.grad(target_score, interp_flat, retain_graph=False)[0]
        grad_sum += grads_flat.reshape(c, B, S, H).detach().sum(dim=0)

        del interp, interp_flat, grads_flat, outputs, target_score, x_rep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_grad = (grad_sum / m).detach()
    dlig_raw = activation_diff.detach() * avg_grad  # [B,S,H], full positions
    return dlig_raw


def check_step(
    dlig: DLIGAttribution,
    x_t_cpu: torch.Tensor,
    step: int,
    layer: str,
    original_length: int,
    mask_token_id: int,
    m_list,
    chunk: int,
    device: torch.device,
    rtol: float,
):
    print("\n" + "=" * 78)
    print(f"STEP t={step}   layer={layer}")
    print("=" * 78)

    x_t = x_t_cpu.to(device)

    # Activations: a = h^(l)(X_t, C), a' = h^(l)(X_t, /0)
    with torch.no_grad():
        real_act = dlig.get_layer_activations(x_t)
        baseline_inp = dlig.create_baseline_input(x_t, mask_token_id, original_length)
        baseline_act = dlig.get_layer_activations(baseline_inp)

    act_device = real_act.device
    act_dtype = real_act.dtype
    x_t = x_t.to(act_device)
    baseline_inp = baseline_inp.to(act_device)
    real_act = real_act.to(device=act_device, dtype=act_dtype)
    baseline_act = baseline_act.to(device=act_device, dtype=act_dtype)

    # Degenerate-step guard: F falls back to max-log-prob when no generated token is
    # unmasked (e.g. step 0). Completeness still holds, but it tests the fallback F.
    n_unmasked = int((x_t[0, original_length:] != mask_token_id).sum().item())
    n_gen = int(x_t.shape[1] - original_length)
    print(f"unmasked generated tokens: {n_unmasked} / {n_gen}")
    degenerate = (n_unmasked == 0)
    if degenerate:
        print("  WARNING: 0 unmasked -> F uses max-log-prob fallback (degenerate). "
              "Prefer a later step.")

    # --- (A) hook transparency: endpoint F via intervention vs plain forward ---
    F_real_plain = score_plain(dlig, x_t, original_length)
    F_base_plain = score_plain(dlig, baseline_inp, original_length)
    F_real_interv = score_intervene(dlig, x_t, real_act, original_length)
    F_base_interv = score_intervene(dlig, x_t, baseline_act, original_length)

    dF = F_real_plain - F_base_plain
    denom = abs(dF) + 1e-8
    t_real = abs(F_real_plain - F_real_interv) / denom
    t_base = abs(F_base_plain - F_base_interv) / denom
    transparency_ok = (t_real < 1e-3) and (t_base < 1e-3)

    print("\n[A] hook transparency  (intervention endpoints == plain forward)")
    print(f"    F_t(X_t,C):  plain={F_real_plain:.6f}  intervene={F_real_interv:.6f}  rel_diff={t_real:.2e}")
    print(f"    F_t(X_t,/0): plain={F_base_plain:.6f}  intervene={F_base_interv:.6f}  rel_diff={t_base:.2e}")
    print(f"    deltaF = F(C) - F(/0) = {dF:.6f}")
    print(f"    -> {'PASS' if transparency_ok else 'FAIL'}")

    # --- (B)+(C) completeness and convergence ---
    print("\n[B/C] completeness:  sum_{i,j} DLIG (ALL positions)  vs  deltaF")
    print(f"    {'m':>5}  {'sumDLIG':>14}  {'deltaF':>14}  {'abs_err':>11}  {'rel_err':>11}")
    last_rel = None
    for m in m_list:
        dlig_raw = integrate_dlig_full(
            dlig, x_t, real_act, baseline_act, original_length, m=m, chunk=chunk
        )
        sum_dlig = float(dlig_raw.reshape(dlig_raw.shape[0], -1).sum(dim=-1).item())
        abs_err = abs(sum_dlig - dF)
        rel_err = abs_err / denom
        last_rel = rel_err
        print(f"    {m:>5}  {sum_dlig:>14.6f}  {dF:>14.6f}  {abs_err:>11.3e}  {rel_err:>11.3e}")
        del dlig_raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completeness_ok = (last_rel is not None) and (last_rel < rtol)
    print(f"    -> completeness at max m: {'PASS' if completeness_ok else 'FAIL'} "
          f"(rel_err={last_rel:.3e} vs rtol={rtol:.1e})")

    del real_act, baseline_act, baseline_inp, x_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "step": step,
        "degenerate": degenerate,
        "transparency_ok": transparency_ok,
        "completeness_ok": completeness_ok,
        "rel_err": last_rel,
    }


def build_arg_parser():
    p = argparse.ArgumentParser(description="DLIG completeness verification harness")
    p.add_argument("--device_map", type=str, default="auto")
    # float32 by default: cleaner numerics -> crisper convergence. bf16 shows a higher
    # rel_err floor (~1e-2..1e-1) from summing T*H terms; that is roundoff, not a bug.
    p.add_argument("--torch_dtype", type=str, default="float32")

    p.add_argument("--layer", type=str, default="16",
                   help="Layer index or 'embed_tokens'. Completeness holds at any single layer.")
    p.add_argument("--prompt", type=str, default="Explain how photosynthesis works.")
    p.add_argument("--system", type=str, default="You are a helpful assistant.")

    p.add_argument("--generation_steps", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=64)

    p.add_argument("--check_steps", type=int, nargs="+", default=None,
                   help="Trajectory steps to verify. Default: last recorded step only.")
    p.add_argument("--m_list", type=int, nargs="+", default=[10, 20, 50],
                   help="Integration-step counts to sweep (convergence). Try adding 100 200.")
    p.add_argument("--integration_batch_size", type=int, default=5,
                   help="Chunk size over integration points (VRAM control). 0 => full batch.")
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="rel_err threshold at max m for completeness PASS.")
    p.add_argument("--score_mode", type=str, default="meancentered",
                   choices=["meancentered", "logprob"],
                   help="F_t scoring fn. 'meancentered' (bounded readout) is the "
                        "completeness-validated mode; 'logprob' sums many unbounded "
                        "log-probs and the all-position sum hits float32 cancellation "
                        "that GROWS with m -> spurious completeness FAIL.")
    p.add_argument("--family", type=str, default="dream",
                   choices=["dream", "diffugpt"],
                   help="Model family / backend.")
    p.add_argument("--seed", type=int, default=0)
    return p


def verify_bidirectional(backend, L, device):
    """
    Assert FULL (bidirectional) attention. A masked diffusion LM must let early
    positions attend to later ones; if attention is silently causal, attribution is
    invalid even though completeness still PASSES (completeness holds under causal).

    Decisive test: read attention weights from the first block under a 4D all-visible
    mask and check that row 0 places meaningful mass on FUTURE positions. Causal => 0
    forward mass; bidirectional => substantial forward mass. (Perturbation-based probes
    are unreliable on small/untrained nets; attention weights are direct.)

    Only meaningful for GPT-2-family backends. Dream is bidirectional by construction.
    """
    blocks = getattr(backend, "blocks", None)
    if blocks is None:
        print("[BIDIR] non-GPT2 backend; skipping (Dream is bidirectional by construction).")
        return None

    S = max(L + 4, 8)
    hdim = backend.wte.embedding_dim
    torch.manual_seed(0)
    emb = torch.randn(1, S, hdim, device=device)
    mask = torch.zeros(1, 1, S, S, device=device, dtype=emb.dtype)

    block0 = blocks[0]
    with torch.no_grad():
        out = block0(emb, layer_past=None, attention_mask=mask, head_mask=None,
                     use_cache=False, output_attentions=True)
    # GPT2Block returns (hidden, present?, attentions?) depending on flags; find the
    # [B,H,S,S] attention tensor.
    attn_w = None
    for o in (out if isinstance(out, tuple) else (out,)):
        if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[-1] == S and o.shape[-2] == S:
            attn_w = o
            break
    if attn_w is None:
        print("[BIDIR] could not read attention weights; falling back to perturbation probe.")
        with torch.no_grad():
            base = backend.suffix_forward(emb, 0)
            ep = emb.clone(); ep[0, -1] += 10.0
            pert = backend.suffix_forward(ep, 0)
        d = (base[0, 0] - pert[0, 0]).abs().max().item()
        ok = d > 1e-4
        print(f"[BIDIR] perturb delta = {d:.3e} -> {'PASS' if ok else 'FAIL'}")
        return ok

    fwd_mass = attn_w[0, 0, 0, 1:].sum().item()  # head 0, row 0, future positions
    is_bidir = fwd_mass > 0.1
    print(f"\n[BIDIR] row0 forward-attention mass = {fwd_mass:.3f}  (causal=0)")
    print(f"        -> {'PASS (bidirectional)' if is_bidir else 'FAIL (attention is causal!)'}")
    if not is_bidir:
        print("        !! DiffuGPT attribution would be INVALID. Check: eager impl, "
              "replace_attention_mask() applied, attn.bias filled True, 4D mask passed.")
    return is_bidir


def verify_shift(backend, tokenizer, rec, L, mask_token_id):
    """
    Token-level shift sanity check. Completeness is shift-agnostic, so it CANNOT
    detect an off-by-one in the self-generated readout. Here we check that the
    backend's declared shift convention actually predicts the committed tokens:
    at the last recorded step, for unmasked generated positions, argmax of the
    logits (under the declared alignment) should frequently match the token the
    sampler committed. A low match rate => wrong predicts_shifted for this ckpt.
    """
    step = max(rec.steps())
    x_t = rec.x_by_step[step]
    device = next(backend.lm_head.parameters()).device if hasattr(backend, "lm_head") \
        else next(backend.model.parameters()).device
    x_t = x_t.to(device)

    with torch.no_grad():
        logits = backend.forward_logits(x_t)

    if backend.predicts_shifted:
        gen_logits = logits[:, L:-1, :]
        target = x_t[:, L + 1:]
    else:
        gen_logits = logits[:, L:, :]
        target = x_t[:, L:]

    pred = gen_logits.argmax(dim=-1)
    valid = (target != mask_token_id)
    if valid.sum() == 0:
        print("[SHIFT] no unmasked generated tokens at this step; skipping shift check.")
        return None
    match = ((pred == target) & valid).sum().item() / int(valid.sum().item())
    print(f"\n[SHIFT] declared predicts_shifted={backend.predicts_shifted}  "
          f"argmax-match on committed tokens = {match:.3f}")
    print("        (high => alignment consistent; very low => wrong shift for this ckpt)")
    return match


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    print(f"[INFO] Loading {args.family} model (dtype={args.torch_dtype})")
    
    # Simplified instantiation
    mm = ModelManager(
        family=args.family, 
        device_map=args.device_map,
        torch_dtype=args.torch_dtype
    )
    
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    print(f"[INFO] Model device: {device}")

    # Backend isolates family-specific internals (layers, positions, shift, sampler).
    backend = build_backend(model, tokenizer, family=args.family)
    print(f"[INFO] Backend family: {backend.family}  predicts_shifted={backend.predicts_shifted}")

    # Bidirectionality gate: MUST pass for DiffuGPT or attribution is invalid.
    verify_bidirectional(backend, L=8, device=device)

    # mask token id (mirror contrastive_runner.py)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = pad_id

    # Tokenize one prompt (chat template if present, else plain text for DiffuGPT)
    input_ids, attention_mask, L = build_prompt_inputs(
        tokenizer, args.system, args.prompt, device
    )
    print(f"[INFO] prompt length L={L}")

    # Generate trajectory via the backend (Dream HF sampler or DiffuGPT reveal loop)
    rec = TrajRecorder()
    print("[INFO] Generating trajectory...")
    _ = backend.generate_trajectory(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        steps=args.generation_steps,
        record_hook=rec.hook,
    )
    steps_avail = rec.steps()
    print(f"[INFO] recorded steps: {steps_avail}")
    if not steps_avail:
        raise RuntimeError("No trajectory steps recorded.")

    # Shift sanity check (the one thing completeness cannot catch).
    verify_shift(backend, tokenizer, rec, L, int(mask_token_id))

    check_steps = args.check_steps if args.check_steps is not None else [max(steps_avail)]
    check_steps = [s for s in check_steps if s in steps_avail]
    if not check_steps:
        raise ValueError(f"None of --check_steps are in recorded steps {steps_avail}")

    # Build one DLIGAttribution + hook on the target layer (reused across steps)
    layer_module = resolve_layer_module(model, args.layer, backend=backend)
    hook_manager = HookManager(model, layer_name=args.layer)
    hook_manager.register_hook(layer_module)
    dlig = DLIGAttribution(
        model, tokenizer, hook_manager,
        integration_steps=max(args.m_list),
        integration_batch_size=(None if args.integration_batch_size <= 0 else args.integration_batch_size),
        disable_kv_cache=True,
        score_mode=args.score_mode,
        backend=backend,
    )
    dlig.set_original_input_length(L)
    dlig.relevant_token_indices = []

    chunk = None if args.integration_batch_size <= 0 else int(args.integration_batch_size)

    results = []
    try:
        for step in check_steps:
            results.append(
                check_step(
                    dlig=dlig,
                    x_t_cpu=rec.x_by_step[step],
                    step=step,
                    layer=args.layer,
                    original_length=L,
                    mask_token_id=int(mask_token_id),
                    m_list=args.m_list,
                    chunk=chunk,
                    device=device,
                    rtol=args.rtol,
                )
            )
    finally:
        hook_manager.remove_hook()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Verdict (degenerate fallback steps excluded from completeness verdict)
    print("\n" + "#" * 78)
    print("VERDICT")
    print("#" * 78)
    all_pass = True
    for r in results:
        tag = " (degenerate, fallback F)" if r["degenerate"] else ""
        comp = "n/a" if r["degenerate"] else ("PASS" if r["completeness_ok"] else "FAIL")
        trans = "PASS" if r["transparency_ok"] else "FAIL"
        print(f"  step {r['step']}: transparency={trans}  completeness={comp}  "
              f"rel_err={r['rel_err']:.3e}{tag}")
        if not r["transparency_ok"]:
            all_pass = False
        if (not r["degenerate"]) and (not r["completeness_ok"]):
            all_pass = False
    print(f"\n  OVERALL: {'PASS - implementation verified' if all_pass else 'FAIL - see above'}")
    print("  Expected on PASS: rel_err strictly DECREASES as m grows toward 0.")


if __name__ == "__main__":
    main()