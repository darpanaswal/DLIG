"""
Hook management for capturing model activations.
"""
import re
import torch
from config import LAYER_NAME
from utils.layer_utils import resolve_layer_path


class HookManager:
    def __init__(self, model, layer_name=LAYER_NAME):
        self.model = model
        self.layer_name = layer_name
        self.real_activations = None
        self.baseline_activations = None
        self.hook_handle = None
        self.forward_hook = None
        
    def capture_activations(self, module, inp, out):
        print(f"Capturing activations from {self.layer_name}")
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out
        return out_tensor.detach().clone()
        
    def register_hooks(self):
        """Register forward hooks for both real and baseline passes."""
        hooked_layer = self._get_layer()
        
        # Remove any existing hooks
        if self.hook_handle is not None:
            self.hook_handle.remove()
            
        # Register new hook
        def forward_hook(module, inp, out):
            self.real_activations = self.capture_activations(module, inp, out)
            return out
            
        self.hook_handle = hooked_layer.register_forward_hook(forward_hook)
        print(f"Hook registered on layer: {self.layer_name}")
        return self.hook_handle

    def register_baseline_hook(self):
        """Register a separate hook for baseline pass."""
        hooked_layer = self._get_layer()
        
        def baseline_hook(module, inp, out):
            self.baseline_activations = self.capture_activations(module, inp, out)
            return out
            
        return hooked_layer.register_forward_hook(baseline_hook)

    def _get_layer(self, layer_name=None):
        if layer_name is None:
            layer_name = self.layer_name
        return resolve_layer_path(self.model, layer_name)
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            print("Hook removed successfully")
        else:
            print("No hook to remove")
    
    def get_activations(self):
        """Get both real and baseline activations."""
        return self.real_activations, self.baseline_activations