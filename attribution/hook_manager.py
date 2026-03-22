"""
Hook management for capturing model activations.

Includes:
- HookManager: Original single-layer hook manager (unchanged API).
- MultiLayerHookManager: Captures activations at multiple layers in a single forward pass.
"""
import torch
from typing import Dict, List, Optional
from utils.layer_utils import resolve_layer_path


class HookManager:
    """Original single-layer hook manager. API unchanged for backward compatibility."""

    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        self.hook_handle = None
        self._resolved_layer = None

    def _capture_fn(self, module, inp, out):
        if isinstance(out, tuple):
            return out[0].detach().clone()
        return out.detach().clone()

    def register_hook(self, target_layer=None):
        """Registers a hook and saves the layer reference."""
        self.remove_hook()
        if target_layer is not None:
            self._resolved_layer = target_layer
        else:
            self._resolved_layer = self._get_layer()

        captured = []

        def forward_hook(module, inp, out):
            captured.append(self._capture_fn(module, inp, out))
            return out

        self.hook_handle = self._resolved_layer.register_forward_hook(forward_hook)
        return captured

    def _get_layer(self, layer_path=None):
        """Returns the saved module or resolves a new one."""
        if self._resolved_layer is not None and layer_path is None:
            return self._resolved_layer
        path = layer_path if layer_path is not None else self.layer_name
        if path is None:
            raise ValueError("No layer path or module provided to HookManager.")
        return resolve_layer_path(self.model, path)

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class MultiLayerHookManager:
    """
    Captures activations at multiple layers simultaneously in a single forward pass.

    Usage:
        mlhm = MultiLayerHookManager(model, layer_specs=["0", "7", "14", "21", "26"])
        all_acts = mlhm.capture_activations(input_ids)
        # all_acts["7"] -> Tensor [B, S, H]

    For DLIG integration, provides per-layer HookManager-compatible wrappers
    that can be passed to DLIGAttribution without changing its intervention logic.
    """

    def __init__(self, model, layer_specs: List[str], resolve_fn=None):
        """
        Args:
            model: The model (e.g., Dream).
            layer_specs: List of layer identifiers (e.g., ["0", "7", "14", "21", "26"]).
            resolve_fn: Optional callable(model, layer_spec) -> nn.Module.
                        If None, uses the default resolve logic.
        """
        self.model = model
        self.layer_specs = list(layer_specs)

        if resolve_fn is None:
            resolve_fn = self._default_resolve
        self._resolve_fn = resolve_fn

        # Resolve all layer modules once
        self.layer_modules: Dict[str, torch.nn.Module] = {}
        for spec in self.layer_specs:
            self.layer_modules[spec] = self._resolve_fn(model, spec)

        # Per-layer SingleLayerView wrappers for DLIGAttribution compatibility
        self._layer_views: Dict[str, "SingleLayerView"] = {}
        for spec in self.layer_specs:
            self._layer_views[spec] = SingleLayerView(
                model=model,
                layer_spec=spec,
                module=self.layer_modules[spec],
            )

    @staticmethod
    def _default_resolve(model, layer_spec: str) -> torch.nn.Module:
        if layer_spec.isdigit():
            return model.model.layers[int(layer_spec)]
        if layer_spec == "embed_tokens":
            return model.model.embed_tokens
        raise ValueError(f"Unsupported layer spec: {layer_spec}")

    def capture_activations(
        self,
        input_ids: torch.Tensor,
        disable_kv_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run ONE forward pass with hooks on ALL layers. Returns dict of activations.

        Args:
            input_ids: [B, S] token IDs.
            disable_kv_cache: Whether to pass use_cache=False.

        Returns:
            Dict mapping layer_spec -> activation tensor [B, S, H] (detached, same device).
        """
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        for spec in self.layer_specs:
            module = self.layer_modules[spec]

            def make_hook(layer_key):
                def hook_fn(mod, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    captured[layer_key] = hidden.detach().clone()
                    return out
                return hook_fn

            h = module.register_forward_hook(make_hook(spec))
            handles.append(h)

        try:
            with torch.no_grad():
                kwargs = {"input_ids": input_ids, "attention_mask": None}
                if disable_kv_cache:
                    try:
                        self.model(use_cache=False, **kwargs)
                    except TypeError:
                        self.model(**kwargs)
                else:
                    self.model(**kwargs)
        finally:
            for h in handles:
                h.remove()

        if len(captured) != len(self.layer_specs):
            missing = set(self.layer_specs) - set(captured.keys())
            raise RuntimeError(f"Failed to capture activations at layers: {missing}")

        return captured

    def get_layer_view(self, layer_spec: str) -> "SingleLayerView":
        """
        Returns a HookManager-compatible wrapper for a single layer.

        This allows DLIGAttribution to use its existing activation_intervention()
        and _get_layer() logic unchanged.
        """
        if layer_spec not in self._layer_views:
            raise ValueError(f"Layer {layer_spec} not in managed layers: {self.layer_specs}")
        return self._layer_views[layer_spec]


class SingleLayerView:
    """
    HookManager-compatible wrapper around a single layer within MultiLayerHookManager.

    DLIGAttribution calls:
        - hook_manager._get_layer(layer_name)  -> returns the nn.Module
        - hook_manager.layer_name              -> used as key for activation_intervention

    This class satisfies both interfaces.
    """

    def __init__(self, model, layer_spec: str, module: torch.nn.Module):
        self.model = model
        self.layer_name = layer_spec
        self._resolved_layer = module
        self.hook_handle = None  # Compatibility

    def _get_layer(self, layer_path=None):
        """Returns the resolved module. Ignores layer_path (already resolved)."""
        return self._resolved_layer

    def register_hook(self, target_layer=None):
        """No-op for compatibility. Actual hooks are managed by MultiLayerHookManager."""
        pass

    def remove_hook(self):
        """No-op for compatibility."""
        pass