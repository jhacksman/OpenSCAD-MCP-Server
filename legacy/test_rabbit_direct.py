import os
import sys
import json
import requests
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venice.ai API configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "B9Y68yQgatQw8wmpmnIMYcGip1phCt-43CS0OktZU6")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API configuration
url = "https://api.venice.ai/api/v1/image/generate"
headers = {
    "Authorization": f"Bearer {VENICE_API_KEY}",
    "Content-Type": "application/json"
}

# Payload for image generation
payload = {
    "height": 1024,
    "width": 1024,
    "steps": 20,
    "return_binary": True,  # Request binary data directly
    "hide_watermark": False,
    "format": "png",
    "embed_exif_metadata": False,
    "model": "flux-dev",
    "prompt": "A low-poly rabbit with black background. 3d file"
}

def generate_image():
    """Generate image using Venice.ai API with the rabbit prompt."""
    try:
        logger.info(f"Sending request to {url} with prompt: '{payload['prompt']}'")
        response = requests.post(url, json=payload, headers=headers)
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            # Save the raw binary response
            filename = "rabbit_low_poly_3d.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Image saved to {output_path}")
            return output_path
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
        
        return None
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting Venice.ai image generation test with rabbit prompt")
    image_path = generate_image()
    
    if image_path:
        logger.info(f"Successfully generated and saved image to {image_path}")
        print(f"\nImage saved to: {image_path}")
    else:
        logger.error("Failed to generate image")
