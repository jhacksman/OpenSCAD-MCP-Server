"""
Test script for generating images with Google Gemini API.
"""

import os
import sys
import logging
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import google.generativeai as genai

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_image(image_data, output_path):
    """Save image data to a file."""
    with open(output_path, "wb") as f:
        f.write(image_data)
    logger.info(f"Image saved to: {output_path}")
    return output_path

def generate_image_with_gemini(prompt, api_key, output_dir="output/gemini_test"):
    """Generate an image using Google Gemini API."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # List available models to find the image generation model
        models = genai.list_models()
        logger.info("Available models:")
        for model in models:
            logger.info(f"- {model.name}")
            if "image" in model.name.lower() or "flash" in model.name.lower():
                logger.info(f"  Supported generation methods: {model.supported_generation_methods}")
        
        # Define the model name
        model_name = "models/gemini-2.0-flash-exp-image-generation"
        logger.info(f"Using model: {model_name}")
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Generate the image
        logger.info(f"Generating content with prompt: '{prompt}'")
        
        # Create the content with specific configuration for image generation
        contents = [
            {"text": prompt},
            {"text": "Please generate an image based on this description."}
        ]
        
        # Set generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 4096,
        }
        
        # Send the request with the prompt
        response = model.generate_content(
            contents,
            generation_config=generation_config
        )
        
        # Log the response structure
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response attributes: {dir(response)}")
        
        # Process the response
        if hasattr(response, 'text'):
            logger.info(f"Text response: {response.text}")
        
        # Check for image in the response
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    logger.info(f"Part text: {part.text}")
                
                if hasattr(part, 'inline_data'):
                    logger.info(f"Found image data")
                    output_path = os.path.join(output_dir, "gemini_generated.png")
                    save_image(part.inline_data.data, output_path)
                    return {
                        "success": True,
                        "prompt": prompt,
                        "local_path": output_path
                    }
        
        # Try alternative response structure
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            logger.info(f"Found image data in candidate")
                            output_path = os.path.join(output_dir, "gemini_generated.png")
                            save_image(part.inline_data.data, output_path)
                            return {
                                "success": True,
                                "prompt": prompt,
                                "local_path": output_path
                            }
        
        # If we get here, no image was found in the response
        logger.info("No image found in the response")
        logger.info(f"Full response: {response}")
        return {
            "success": False,
            "error": "No image found in the response",
            "response_text": response.text if hasattr(response, 'text') else 'No text'
        }
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("google", "")
    
    if not api_key:
        logger.error("API key not found in environment variable 'google'")
        sys.exit(1)
    
    # Test prompt for 3D object generation
    prompt = "Create a 3D rendered image of a low-poly rabbit with black background. The rabbit should be white with red eyes, in a minimalist style suitable for 3D printing."
    
    # Generate the image
    result = generate_image_with_gemini(prompt, api_key)
    
    if result["success"]:
        logger.info(f"Image generated successfully: {result['local_path']}")
    else:
        logger.error(f"Failed to generate image: {result.get('error', 'Unknown error')}")
        if 'response_text' in result:
            logger.info(f"Response text: {result['response_text']}")
