"""
Google Gemini API integration for image generation.
"""

import os
import logging
import base64
import uuid
from typing import Dict, Any, List, Optional
from io import BytesIO
from PIL import Image
import google.generativeai as genai

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
        self.model_name = "models/gemini-2.0-flash-exp-image-generation"
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_image(self, prompt: str, output_path: Optional[str] = None, 
                       temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """
        Generate an image using Google Gemini API.
        
        Args:
            prompt: Text description for image generation
            output_path: Path to save the generated image
            temperature: Controls randomness in generation (0.0 to 1.0)
            **kwargs: Additional parameters for Gemini API
            
        Returns:
            Dictionary containing image data and metadata
        """
        logger.info(f"Generating image with prompt: {prompt}")
        
        try:
            # Create the model
            model = genai.GenerativeModel(self.model_name)
            
            # Create the content with specific configuration for image generation
            contents = [
                {"text": prompt},
                {"text": "Please generate an image based on this description."}
            ]
            
            # Set generation config
            generation_config = {
                "temperature": temperature,
                "top_p": 1.0,
                "top_k": 32,
                "max_output_tokens": 4096,
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                generation_config[key] = value
            
            # Generate the image
            response = model.generate_content(
                contents,
                generation_config=generation_config
            )
            
            # Extract text response if available
            text_response = None
            if hasattr(response, 'text'):
                text_response = response.text
            
            # Process the response to find image data
            image_data = None
            
            # Check for image in the response
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        break
            
            # Try alternative response structure
            if not image_data and hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_data = part.inline_data.data
                                break
            
            if not image_data:
                raise ValueError("No image was generated in the response")
            
            # Save image if output_path is provided
            if not output_path:
                # Generate output path if not provided
                os.makedirs(self.output_dir, exist_ok=True)
                filename = f"gemini_{uuid.uuid4()}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            # Save image
            image = Image.open(BytesIO(image_data))
            image.save(output_path)
            
            logger.info(f"Image saved to {output_path}")
            
            return {
                "success": True,
                "prompt": prompt,
                "local_path": output_path,
                "image_data": image_data,
                "text_response": text_response
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
    def generate_multiple_views(self, prompt: str, num_views: int = 4, 
                              base_image_path: Optional[str] = None,
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate multiple views of the same 3D object.
        
        Args:
            prompt: Text description of the object
            num_views: Number of views to generate
            base_image_path: Optional path to a base image
            output_dir: Directory to save the generated images
            
        Returns:
            Dictionary containing results for each view
        """
        if not output_dir:
            output_dir = os.path.join(self.output_dir, f"multi_view_{uuid.uuid4()}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # View directions to include in prompts
        view_directions = [
            "front view", "side view from the right", 
            "side view from the left", "back view",
            "top view", "bottom view", "45-degree angle view"
        ]
        
        results = {}
        all_successful = True
        
        # Generate images for each view direction
        for i in range(min(num_views, len(view_directions))):
            view = view_directions[i]
            view_prompt = f"{prompt} - {view}, same object, consistent style and details"
            
            # Generate the image
            output_path = os.path.join(output_dir, f"view_{i+1}_{view.replace(' ', '_')}.png")
            result = self.generate_image(view_prompt, output_path=output_path)
            
            # Add view direction to result
            result["view_direction"] = view
            result["view_index"] = i + 1
            
            results[view] = result
            if not result["success"]:
                all_successful = False
        
        return {
            "success": all_successful,
            "views": results,
            "base_prompt": prompt,
            "output_dir": output_dir
        }
    
    @staticmethod
    def list_available_models():
        """List available Gemini models, focusing on those that might support image generation."""
        models = genai.list_models()
        image_models = []
        
        for model in models:
            if "image" in model.name.lower() or "flash" in model.name.lower():
                supported_methods = getattr(model, 'supported_generation_methods', [])
                image_models.append({
                    "name": model.name,
                    "supported_methods": supported_methods
                })
        
        return image_models
