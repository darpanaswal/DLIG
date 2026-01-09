"""
Hook management for capturing model activations.
"""
import torch
from utils.layer_utils import resolve_layer_path

class HookManager:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        self.hook_handle = None
        self._resolved_layer = None # Store the actual module here
        
    def _capture_fn(self, module, inp, out):
        if isinstance(out, tuple):
            return out[0].detach().clone()
        return out.detach().clone()
        
    def register_hook(self, target_layer=None):
        """Registers a hook and saves the layer reference."""
        self.remove_hook()
            
        # If target_layer is passed (nn.Module), save it. 
        # Otherwise, resolve from layer_name string.
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
        # If we already have a resolved module, return it
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