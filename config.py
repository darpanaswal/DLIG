"""
Configuration settings for DLIG attribution analysis.
"""

# Model configuration
MODEL_PATH = "./models/Dream"
LAYER_NAME = 'model.layers[26]'

# DLIG computation parameters
INTEGRATION_STEPS = 20  # Number of integration steps (m in the formula)

# Generation parameters
MAX_NEW_TOKENS = 256
GENERATION_STEPS = 15

# Output configuration
OUTPUT_DIR = "./outputs/15 steps"
CSV_FILENAME = f"{LAYER_NAME.replace('model.', '').replace('[', '').replace(']', '')}_{GENERATION_STEPS}.csv"

# Device configuration (can be overridden)
DEVICE_MAP = "auto"
TORCH_DTYPE = "float32"