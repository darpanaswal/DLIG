import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
dotenv_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

hf_token = os.getenv("HUGGINGFACE_API_KEY")

DREAM_PATH = BASE_DIR / "models/Dream"
GPT_PATH = BASE_DIR / "models/Diffugpt"
OUTPUT_DIR = BASE_DIR / "outputs"
CONTRAST_DATASET = BASE_DIR / "data/contrastive_dataset.json"