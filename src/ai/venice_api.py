"""
Venice.ai API client for image generation using the Flux model.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VeniceImageGenerator:
    """Client for Venice.ai's image generation API using the Flux model."""
    
    def __init__(self, api_key: str = None, output_dir: str = "output/images"):
        """
        Initialize the Venice.ai API client.
        
        Args:
            api_key: API key for Venice.ai (defaults to environment variable)
            output_dir: Directory to store generated images
        """
        self.api_key = api_key or os.getenv("VENICE_API_KEY")
        if not self.api_key:
            logger.warning("No Venice.ai API key provided")
        
        self.api_endpoint = "https://api.venice.ai/api/v1/image/generate"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_image(self, prompt: str, model: str = "flux", 
                      width: int = 1024, height: int = 1024,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an image using Venice.ai's Flux model.
        
        Args:
            prompt: Text description for image generation
            model: Model to use ("flux" or "flux-dev-uncensored")
            width: Image width
            height: Image height
            output_path: Optional path to save the generated image
            
        Returns:
            Dictionary containing image data and metadata
        """
        if not self.api_key:
            raise ValueError("Venice.ai API key is required")
        
        # Prepare request payload
        payload = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height
        }
        
        # Set up headers with API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make API request
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            # Process response
            result = response.json()
            
            # Generate output path if not provided
            if not output_path:
                # Create a filename based on the prompt
                filename = f"{prompt[:20].replace(' ', '_')}_{model}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            # Save image if image_url is in the result
            if "image_url" in result:
                self._download_image(result["image_url"], output_path)
                result["local_path"] = output_path
            
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
