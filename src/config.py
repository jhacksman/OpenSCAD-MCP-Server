import os
from typing import Dict, Any

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Output subdirectories
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MULTI_VIEW_DIR = os.path.join(OUTPUT_DIR, "multi_view")
APPROVED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "approved_images")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SCAD_DIR = os.path.join(BASE_DIR, "scad")

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.0-flash-exp-image-generation"  # Default model for image generation

# CUDA Multi-View Stereo configuration
CUDA_MVS_PATH = os.getenv("CUDA_MVS_PATH", os.path.join(BASE_DIR, "cuda-mvs"))
CUDA_MVS_USE_GPU = os.getenv("CUDA_MVS_USE_GPU", "False").lower() == "true"  # Default to CPU for macOS compatibility

# Venice.ai API configuration (optional)
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")  # Set via environment variable
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = "fluently-xl"  # Default model for fastest image generation (2.30s)

# Deprecated configurations (moved to old folder)
# These are kept for reference but not used in the new workflow
DEPRECATED = {
    "SAM2_CHECKPOINT_PATH": os.getenv("SAM2_CHECKPOINT_PATH", os.path.join(BASE_DIR, "models", "sam2_vit_b.pth")),
    "SAM2_MODEL_TYPE": os.getenv("SAM2_MODEL_TYPE", "vit_b"),
    "SAM2_USE_GPU": os.getenv("SAM2_USE_GPU", "False").lower() == "true",
    "THREESTUDIO_PATH": os.path.join(BASE_DIR, "threestudio")
}

# Create necessary directories
for directory in [OUTPUT_DIR, IMAGES_DIR, MULTI_VIEW_DIR, APPROVED_IMAGES_DIR, MODELS_DIR, SCAD_DIR]:
    os.makedirs(directory, exist_ok=True)
