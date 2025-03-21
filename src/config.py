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
SAM2_CHECKPOINT_PATH = os.getenv("SAM2_CHECKPOINT_PATH", os.path.join(BASE_DIR, "models", "sam2_vit_b.pth"))
SAM2_MODEL_TYPE = os.getenv("SAM2_MODEL_TYPE", "vit_b")  # Using vit_b for better CPU performance
SAM2_USE_GPU = os.getenv("SAM2_USE_GPU", "False").lower() == "true"  # Default to CPU for macOS compatibility
THREESTUDIO_PATH = os.path.join(BASE_DIR, "threestudio")

# Venice.ai API configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")  # Set via environment variable
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = "fluently-xl"  # Default model for fastest image generation (2.30s)

# Create necessary directories
for directory in [OUTPUT_DIR, IMAGES_DIR, MASKS_DIR, MODELS_DIR, SCAD_DIR]:
    os.makedirs(directory, exist_ok=True)
