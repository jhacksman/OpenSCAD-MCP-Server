"""
Test script for generating images with Google Gemini API.
"""

import os
import sys
import logging
import base64
from pathlib import Path
import google.generativeai as genai

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_image_from_base64(base64_data, output_path):
    """Save base64 image data to a file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_data))
    logger.info(f"Image saved to: {output_path}")
    return output_path

def generate_image_with_gemini(prompt, api_key, output_dir="output/gemini_test"):
    """Generate an image using Google Gemini API."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # Initialize the model
        model_name = "gemini-2.0-flash-exp-image-generation"
        logger.info(f"Using model: {model_name}")
        
        # Set generation config
        generation_config = {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048,
        }
        
        # Create the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
        
        # Generate the image
        logger.info(f"Generating content with prompt: '{prompt}'")
        
        # Use the special image generation format with specific generation config
        response = model.generate_content(
            [
                {"text": prompt},
                {"text": "Please generate an image based on this description."}
            ],
            generation_config={
                "temperature": 0.9,
                "top_p": 1.0,
                "top_k": 32,
                "max_output_tokens": 4096,
                "candidate_count": 1
            },
            stream=False
        )
        
        # Log the response structure
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response text: {response.text if hasattr(response, 'text') else 'No text'}")
        
        # Check if the response contains parts that might have images
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        # Check if this part has image data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            mime_type = part.inline_data.mime_type
                            if mime_type.startswith('image/'):
                                logger.info(f"Found image with MIME type: {mime_type}")
                                image_data = part.inline_data.data
                                
                                # Save the image
                                output_path = os.path.join(output_dir, "gemini_generated.png")
                                save_image_from_base64(image_data, output_path)
                                
                                return {
                                    "success": True,
                                    "prompt": prompt,
                                    "local_path": output_path
                                }
        
        # If we get here, no image was found in the response
        logger.info("No image found in the response")
        return {
            "success": False,
            "error": "No image found in the response",
            "response_text": response.text if hasattr(response, 'text') else 'No text'
        }
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
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
    
    # Test prompt
    prompt = "A low-poly rabbit with black background, 3D model style. The rabbit should be white with red eyes, in a minimalist style suitable for 3D printing."
    
    # Generate the image
    result = generate_image_with_gemini(prompt, api_key)
    
    if result["success"]:
        logger.info(f"Image generated successfully: {result['local_path']}")
    else:
        logger.error(f"Failed to generate image: {result.get('error', 'Unknown error')}")
        if 'response_text' in result:
            logger.info(f"Response text: {result['response_text']}")
