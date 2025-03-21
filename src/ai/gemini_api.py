"""
Google Gemini API integration for image generation.
"""

import os
import logging
import base64
from typing import Dict, Any, List, Optional
from io import BytesIO
from PIL import Image
import requests

logger = logging.getLogger(__name__)

class GeminiImageGenerator:
    """
    Wrapper for Google Gemini API for generating images.
    """
    
    def __init__(self, api_key: str, output_dir: str = "output/images"):
        """
        Initialize the Gemini image generator.
        
        Args:
            api_key: Google Gemini API key
            output_dir: Directory to store generated images
        """
        self.api_key = api_key
        self.output_dir = output_dir
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_image(self, prompt: str, model: str = "gemini-2.0-flash-exp-image-generation", 
                       output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate an image using Google Gemini API.
        
        Args:
            prompt: Text description for image generation
            model: Gemini model to use
            output_path: Path to save the generated image
            **kwargs: Additional parameters for Gemini API
            
        Returns:
            Dictionary containing image data and metadata
        """
        logger.info(f"Generating image with prompt: {prompt}")
        
        try:
            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["Text", "Image"]
                }
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/models/{model}:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                },
                json=payload
            )
            
            # Check for errors
            response.raise_for_status()
            result = response.json()
            
            # Extract image data
            image_data = None
            for part in result["candidates"][0]["content"]["parts"]:
                if "inlineData" in part:
                    image_data = base64.b64decode(part["inlineData"]["data"])
                    break
            
            if not image_data:
                raise ValueError("No image was generated in the response")
            
            # Save image if output_path is provided
            if not output_path:
                # Generate output path if not provided
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, f"{prompt[:20].replace(' ', '_')}.png")
            
            # Save image
            image = Image.open(BytesIO(image_data))
            image.save(output_path)
            
            logger.info(f"Image saved to {output_path}")
            
            return {
                "prompt": prompt,
                "model": model,
                "local_path": output_path,
                "image_data": image_data
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    def generate_multiple_views(self, prompt: str, num_views: int = 4, 
                              base_image_path: Optional[str] = None,
                              output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple views of the same 3D object.
        
        Args:
            prompt: Text description of the object
            num_views: Number of views to generate
            base_image_path: Optional path to a base image
            output_dir: Directory to save the generated images
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        if not output_dir:
            output_dir = os.path.join(self.output_dir, prompt[:20].replace(' ', '_'))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # View directions to include in prompts
        view_directions = [
            "front view", "side view from the right", 
            "side view from the left", "back view",
            "top view", "bottom view", "45-degree angle view"
        ]
        
        results = []
        
        # Generate images for each view direction
        for i in range(min(num_views, len(view_directions))):
            view_prompt = f"{prompt} - {view_directions[i]}, same object, consistent style and details"
            
            # Generate the image
            output_path = os.path.join(output_dir, f"view_{i+1}.png")
            result = self.generate_image(view_prompt, output_path=output_path)
            
            # Add view direction to result
            result["view_direction"] = view_directions[i]
            result["view_index"] = i + 1
            
            results.append(result)
        
        return results
