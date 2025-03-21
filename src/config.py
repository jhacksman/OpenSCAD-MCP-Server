import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# CUDA Multi-View Stereo configuration (local)
CUDA_MVS_PATH = os.getenv("CUDA_MVS_PATH", os.path.join(BASE_DIR, "cuda-mvs"))
CUDA_MVS_USE_GPU = os.getenv("CUDA_MVS_USE_GPU", "False").lower() == "true"  # Default to CPU for macOS compatibility

# Remote CUDA Multi-View Stereo configuration
REMOTE_CUDA_MVS = {
    # General settings
    "ENABLED": os.getenv("REMOTE_CUDA_MVS_ENABLED", "True").lower() == "true",
    "USE_LAN_DISCOVERY": os.getenv("REMOTE_CUDA_MVS_USE_LAN_DISCOVERY", "True").lower() == "true",
    
    # Server connection
    "SERVER_URL": os.getenv("REMOTE_CUDA_MVS_SERVER_URL", ""),  # Empty means use LAN discovery
    "API_KEY": os.getenv("REMOTE_CUDA_MVS_API_KEY", ""),
    "DISCOVERY_PORT": int(os.getenv("REMOTE_CUDA_MVS_DISCOVERY_PORT", "8765")),
    
    # Connection parameters
    "CONNECTION_TIMEOUT": int(os.getenv("REMOTE_CUDA_MVS_CONNECTION_TIMEOUT", "10")),
    "UPLOAD_CHUNK_SIZE": int(os.getenv("REMOTE_CUDA_MVS_UPLOAD_CHUNK_SIZE", "1048576")),  # 1MB
    "DOWNLOAD_CHUNK_SIZE": int(os.getenv("REMOTE_CUDA_MVS_DOWNLOAD_CHUNK_SIZE", "1048576")),  # 1MB
    
    # Retry and error handling
    "MAX_RETRIES": int(os.getenv("REMOTE_CUDA_MVS_MAX_RETRIES", "3")),
    "BASE_RETRY_DELAY": float(os.getenv("REMOTE_CUDA_MVS_BASE_RETRY_DELAY", "1.0")),
    "MAX_RETRY_DELAY": float(os.getenv("REMOTE_CUDA_MVS_MAX_RETRY_DELAY", "60.0")),
    "JITTER_FACTOR": float(os.getenv("REMOTE_CUDA_MVS_JITTER_FACTOR", "0.1")),
    
    # Health check
    "HEALTH_CHECK_INTERVAL": int(os.getenv("REMOTE_CUDA_MVS_HEALTH_CHECK_INTERVAL", "60")),
    "CIRCUIT_BREAKER_THRESHOLD": int(os.getenv("REMOTE_CUDA_MVS_CIRCUIT_BREAKER_THRESHOLD", "5")),
    "CIRCUIT_BREAKER_RECOVERY_TIMEOUT": float(os.getenv("REMOTE_CUDA_MVS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30.0")),
    
    # Processing parameters
    "DEFAULT_RECONSTRUCTION_QUALITY": os.getenv("REMOTE_CUDA_MVS_DEFAULT_QUALITY", "normal"),  # low, normal, high
    "DEFAULT_OUTPUT_FORMAT": os.getenv("REMOTE_CUDA_MVS_DEFAULT_FORMAT", "obj"),
    "WAIT_FOR_COMPLETION": os.getenv("REMOTE_CUDA_MVS_WAIT_FOR_COMPLETION", "True").lower() == "true",
    "POLL_INTERVAL": int(os.getenv("REMOTE_CUDA_MVS_POLL_INTERVAL", "5")),
    
    # Output directories
    "OUTPUT_DIR": MODELS_DIR,
    "IMAGES_DIR": IMAGES_DIR,
    "MULTI_VIEW_DIR": MULTI_VIEW_DIR,
    "APPROVED_IMAGES_DIR": APPROVED_IMAGES_DIR,
}

# Venice.ai API configuration (optional)
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")  # Set via environment variable
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = "fluently-xl"  # Default model for fastest image generation (2.30s)

# Image approval configuration
IMAGE_APPROVAL = {
    "ENABLED": os.getenv("IMAGE_APPROVAL_ENABLED", "True").lower() == "true",
    "AUTO_APPROVE": os.getenv("IMAGE_APPROVAL_AUTO_APPROVE", "False").lower() == "true",
    "APPROVAL_TIMEOUT": int(os.getenv("IMAGE_APPROVAL_TIMEOUT", "300")),  # 5 minutes
    "MIN_APPROVED_IMAGES": int(os.getenv("IMAGE_APPROVAL_MIN_IMAGES", "3")),
    "APPROVED_IMAGES_DIR": APPROVED_IMAGES_DIR,
}

# Multi-view to model pipeline configuration
MULTI_VIEW_PIPELINE = {
    "DEFAULT_NUM_VIEWS": int(os.getenv("MULTI_VIEW_DEFAULT_NUM_VIEWS", "4")),
    "MIN_NUM_VIEWS": int(os.getenv("MULTI_VIEW_MIN_NUM_VIEWS", "3")),
    "MAX_NUM_VIEWS": int(os.getenv("MULTI_VIEW_MAX_NUM_VIEWS", "8")),
    "VIEW_ANGLES": [0, 90, 180, 270],  # Default view angles (degrees)
    "OUTPUT_DIR": MULTI_VIEW_DIR,
}

# Natural language processing configuration for MCP
NLP = {
    "ENABLE_INTERACTIVE_PARAMS": os.getenv("NLP_ENABLE_INTERACTIVE_PARAMS", "True").lower() == "true",
    "PARAM_EXTRACTION_PROMPT_TEMPLATE": """
    Extract the following parameters from the user's request for 3D model generation:
    
    1. Object description
    2. Number of views requested (default: 4)
    3. Reconstruction quality (low, normal, high)
    4. Output format (obj, ply, stl, scad)
    5. Any specific view angles mentioned
    
    If a parameter is not specified, return the default value or leave blank.
    Format the response as a JSON object.
    
    User request: {user_request}
    """,
}

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
