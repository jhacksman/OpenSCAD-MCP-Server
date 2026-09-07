"""
Venice.ai API client for image generation using the Flux model.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Venice.ai model mapping and descriptions
VENICE_MODELS = {
    # Model name: (aliases, description)
    "fluently-xl": (
        ["fast", "quick", "fastest", "speed", "rapid", "efficient"],
        "Fastest model (2.30s) with good quality"
    ),
    "flux-dev": (
        ["high quality", "detailed", "hq", "best quality", "premium"],
        "High-quality model with detailed results"
    ),
    "flux-dev-uncensored": (
        ["uncensored", "unfiltered", "unrestricted"],
        "Uncensored version of the flux-dev model"
    ),
    "stable-diffusion-3.5": (
        ["stable diffusion", "sd3", "sd3.5", "standard"],
        "Stable Diffusion 3.5 model"
    ),
    "pony-realism": (
        ["realistic", "realism", "pony", "photorealistic"],
        "Specialized model for realistic outputs"
    ),
    "lustify-sdxl": (
        ["stylized", "artistic", "creative", "lustify"],
        "Artistic stylization model"
    ),
}

class VeniceImageGenerator:
    """Client for Venice.ai's image generation API."""
    
    def __init__(self, api_key: str, output_dir: str = "output/images"):
        """
        Initialize the Venice.ai API client.
        
        Args:
            api_key: API key for Venice.ai
            output_dir: Directory to store generated images
        """
        self.api_key = api_key
        if not self.api_key:
            logger.warning("No Venice.ai API key provided")
        
        # API endpoint from documentation
        self.base_url = "https://api.venice.ai/api/v1"
        self.api_endpoint = f"{self.base_url}/image/generate"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def map_model_preference(self, preference: str) -> str:
        """
        Map a natural language preference to a Venice.ai model name.
        
        Args:
            preference: Natural language description of desired model
            
        Returns:
            Name of the matching Venice.ai model
        """
        if not preference or preference.lower() in ["default", "fluently-xl", "fluently xl"]:
            return "fluently-xl"
            
        preference = preference.lower()
        
        # Check for exact matches first
        for model_name in VENICE_MODELS:
            if model_name.lower() == preference:
                return model_name
        
        # Check for keyword matches
        for model_name, (aliases, _) in VENICE_MODELS.items():
            for alias in aliases:
                if alias in preference:
                    return model_name
        
        # Default to fluently-xl if no match found
        return "fluently-xl"
    
    def generate_image(self, prompt: str, model: str = "fluently-xl", 
                      width: int = 1024, height: int = 1024,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an image using Venice.ai's API.
        
        Args:
            prompt: Text description for image generation
            model: Model to use - can be a specific model name or natural language description:
                - "fluently-xl" (default): Fastest model (2.30s) with good quality
                - "flux-dev": High-quality model with detailed results
                - "flux-dev-uncensored": Uncensored version of the flux-dev model
                - "stable-diffusion-3.5": Stable Diffusion 3.5 model
                - "pony-realism": Specialized model for realistic outputs
                - "lustify-sdxl": Artistic stylization model
                - Or use natural language like "high quality", "fastest", "realistic", etc.
            width: Image width
            height: Image height
            output_path: Optional path to save the generated image
            
        Returns:
            Dictionary containing image data and metadata
        """
        if not self.api_key:
            raise ValueError("Venice.ai API key is required")
        
        # Map the model preference to a specific model name
        mapped_model = self.map_model_preference(model)
        
        # Prepare request payload
        payload = {
            "model": mapped_model,
            "prompt": prompt,
            "height": height,
            "width": width,
            "steps": 20,
            "return_binary": False,
            "hide_watermark": True,  # Remove watermark as requested
            "format": "png",
            "embed_exif_metadata": False
        }
        
        # Set up headers with API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make API request
            logger.info(f"Sending request to {self.api_endpoint}")
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers
            )
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"Error generating image: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Process response
            result = response.json()
            
            # Add the mapped model to the result
            result["model"] = mapped_model
            
            # Generate output path if not provided
            if not output_path:
                # Create a filename based on the prompt
                filename = f"{prompt[:20].replace(' ', '_')}_{mapped_model}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            # Save image if images array is in the result
            if "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0]
                self._download_image(image_url, output_path)
                result["local_path"] = output_path
                result["image_url"] = image_url
            
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating image with Venice.ai: {str(e)}")
            raise
    
    def _download_image(self, image_url: str, output_path: str) -> None:
        """
        Download image from URL and save to local path.
        
        Args:
            image_url: URL of the image to download
            output_path: Path to save the downloaded image
        """
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Image saved to {output_path}")
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise
