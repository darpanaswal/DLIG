DLIG Attribution Analysis
This project implements DLIG (Diffusion Language Integrated Gradients) attribution analysis for language models. The code has been modularized into separate components for better maintainability and reusability.
Project Structure
├── config.py                      # Configuration settings
├── main.py                        # Main execution script
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── models/                        # Model management
│   ├── __init__.py
│   └── model_manager.py           # Model loading and management
├── attribution/                   # Attribution methods
│   ├── __init__.py
│   ├── dlig_attribution.py        # DLIG attribution implementation
│   └── hook_manager.py            # Hook management for activation capture
└── utils/                         # Utility functions
    ├── __init__.py
    └── data_processor.py          # Input processing and results export
Module Descriptions
config.py
Contains all configuration settings including:

Model path and layer configuration
DLIG computation parameters
Generation parameters
Output settings

models/model_manager.py
Handles model and tokenizer loading with:

ModelManager: Loads and manages the model and tokenizer
GradientEnabledModel: Wrapper for enabling gradient computation during generation

attribution/dlig_attribution.py
Core DLIG attribution implementation:

DLIGAttribution: Main class for computing DLIG scores
Baseline input creation
Integrated gradients computation
Hook function for generation-time attribution

attribution/hook_manager.py
Manages model hooks for activation capture:

HookManager: Registers and manages forward hooks
Activation capture and cleanup

utils/data_processor.py
Handles input/output processing:

DataProcessor: Input preparation and results analysis
CSV export functionality
Results visualization and analysis

main.py
Main execution script that orchestrates all components.
Usage

Install dependencies:

bashpip install -r requirements.txt

Update the configuration in config.py:

Set MODEL_PATH to your model directory
Adjust other parameters as needed


Run the analysis:

bashpython main.py
Customization
Changing the Input Message
Edit the messages list in main.py:
pythonmessages = [
    {"role": "user", "content": "Your custom prompt here"}
]
Adjusting DLIG Parameters
Modify parameters in config.py:

INTEGRATION_STEPS: Number of integration steps
MAX_NEW_TOKENS: Maximum tokens to generate
GENERATION_STEPS: Number of generation steps

Using Different Layers
Change LAYER_NAME in config.py to target different model layers:
pythonLAYER_NAME = 'model.layers.10'  # Example for layer 10
Output
The script generates:

Console output with step-by-step attribution analysis
CSV file with detailed attribution scores in the attribution_csv directory

Error Handling
The code includes comprehensive error handling:

Model loading errors
DLIG computation failures
Hook management cleanup
Memory management with garbage collection

Requirements

Python 3.8+
PyTorch 2.0+
Transformers 4.30+
CUDA-compatible GPU (recommended)

Notes

The code assumes a specific model architecture with model.embed_tokens and diffusion generation capabilities
Memory usage can be significant during DLIG computation due to gradient retention
The hook is automatically cleaned up after execution