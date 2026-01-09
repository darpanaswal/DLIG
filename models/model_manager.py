"""
Model loading and management utilities.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from utils.config import MODEL_PATH

class ModelManager:
    def __init__(self, model_path=str(MODEL_PATH), device_map="auto", torch_dtype="float32"):
        """
        Hyperparameters are now initialized here from main.py arguments.
        """
        self.model_path = model_path
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
        print(f"Loading model from {self.model_path}...")
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("Model and tokenizer loaded successfully!")
        return self.model, self.tokenizer
    
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
        Generates output while allowing DLIG to compute gradients 
        at each timestep t[cite: 17, 20].
        """
        with torch.enable_grad():
            self.model.train() 
            try:
                # 1. Extract hooks. DLIG specifically uses the logits hook for the 
                # Riemann sum calculation[cite: 14, 27].
                generation_logits_hook_func = kwargs.pop(
                    "generation_logits_hook_func", 
                    lambda step, x, logits: logits
                )
                
                # Restore the tokens hook as a no-op to satisfy the positional argument
                generation_tokens_hook_func = kwargs.pop(
                    "generation_tokens_hook_func", 
                    lambda step, x, logits: x
                )
                
                # 2. Standard generation setup
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
                
                # 3. Call _sample with all required positional arguments
                # DLIG logic is triggered inside generation_logits_hook_func.
                result = self.model._sample(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    generation_tokens_hook_func=generation_tokens_hook_func,  # Restored
                    generation_logits_hook_func=generation_logits_hook_func   # Target for DLIG
                )
                return result
            finally:
                self.model.eval()