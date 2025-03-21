"""
Test script for the GeminiImageGenerator class.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GeminiImageGenerator class
from src.ai.gemini_api import GeminiImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_image_generation():
    """Test generating a single image with the Gemini API."""
    # Get API key from environment
    api_key = os.environ.get("google", "")
    
    if not api_key:
        logger.error("API key not found in environment variable 'google'")
        sys.exit(1)
    
    # Initialize the generator
    generator = GeminiImageGenerator(api_key)
    
    # Test prompt for 3D object generation
    prompt = "Create a 3D rendered image of a low-poly rabbit with black background. The rabbit should be white with red eyes, in a minimalist style suitable for 3D printing."
    
    # Generate the image
    result = generator.generate_image(prompt, output_path="output/gemini_test_client/rabbit.png")
    
    if result["success"]:
        logger.info(f"Image generated successfully: {result['local_path']}")
        return True
    else:
        logger.error(f"Failed to generate image: {result.get('error', 'Unknown error')}")
        if 'text_response' in result and result['text_response']:
            logger.info(f"Text response: {result['text_response']}")
        return False

def test_multi_view_generation():
    """Test generating multiple views of the same 3D object."""
    # Get API key from environment
    api_key = os.environ.get("google", "")
    
    if not api_key:
        logger.error("API key not found in environment variable 'google'")
        sys.exit(1)
    
    # Initialize the generator
    generator = GeminiImageGenerator(api_key)
    
    # Base prompt for the 3D object
    base_prompt = "A low-poly white rabbit with red eyes on a black background, in a minimalist 3D style suitable for 3D printing."
    
    # Views to generate
    views = ["front", "side (45 degrees from the right)", "top-down"]
    
    # Generate the multi-view images
    result = generator.generate_multiple_views(
        prompt=base_prompt, 
        num_views=len(views),
        output_dir="output/gemini_multi_view_test"
    )
    
    if result["success"]:
        logger.info(f"All views generated successfully in: {result['output_dir']}")
        for view, view_result in result["views"].items():
            logger.info(f"- {view}: {view_result['local_path']}")
        return True
    else:
        logger.error("Failed to generate all views")
        for view, view_result in result["views"].items():
            status = "SUCCESS" if view_result["success"] else "FAILED"
            logger.info(f"- {view}: {status}")
            if not view_result["success"] and "error" in view_result:
                logger.error(f"  Error: {view_result['error']}")
        return False

def test_list_models():
    """Test listing available Gemini models."""
    # Get API key from environment
    api_key = os.environ.get("google", "")
    
    if not api_key:
        logger.error("API key not found in environment variable 'google'")
        sys.exit(1)
    
    # Initialize the generator
    generator = GeminiImageGenerator(api_key)
    
    # List available models
    models = generator.list_available_models()
    
    logger.info("Available models that might support image generation:")
    for model in models:
        logger.info(f"- {model['name']}")
        logger.info(f"  Supported methods: {model['supported_methods']}")
    
    return True

if __name__ == "__main__":
    logger.info("Testing Gemini Image Generator")
    
    # Test listing models
    logger.info("\n=== Testing Model Listing ===")
    test_list_models()
    
    # Test single image generation
    logger.info("\n=== Testing Single Image Generation ===")
    single_result = test_single_image_generation()
    
    # Test multi-view generation
    logger.info("\n=== Testing Multi-View Generation ===")
    multi_result = test_multi_view_generation()
    
    # Report overall results
    if single_result and multi_result:
        logger.info("\n✅ All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed")
        if not single_result:
            logger.error("- Single image generation test failed")
        if not multi_result:
            logger.error("- Multi-view generation test failed")
