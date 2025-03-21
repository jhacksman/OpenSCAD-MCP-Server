import os
from typing import Dict, Any

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Output subdirectories
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SCAD_DIR = os.path.join(BASE_DIR, "scad")

# AI model paths
SAM_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "sam_vit_h.pth")
THREESTUDIO_PATH = os.path.join(BASE_DIR, "threestudio")

# Venice.ai API configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")  # Set via environment variable
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = "fluently-xl"  # Default model that works with the API

# Create necessary directories
for directory in [OUTPUT_DIR, IMAGES_DIR, MASKS_DIR, MODELS_DIR, SCAD_DIR]:
    os.makedirs(directory, exist_ok=True)
