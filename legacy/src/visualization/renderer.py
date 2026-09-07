import os
import subprocess
import logging
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class Renderer:
    """
    Handles rendering of OpenSCAD models to preview images.
    Implements multi-angle views and fallback rendering when headless mode fails.
    """
    
    def __init__(self, openscad_wrapper):
        """
        Initialize the renderer.
        
        Args:
            openscad_wrapper: Instance of OpenSCADWrapper for generating previews
        """
        self.openscad_wrapper = openscad_wrapper
        
        # Standard camera angles for multi-view rendering
        self.camera_angles = {
            'front': "0,0,0,0,0,0,50",
            'top': "0,0,0,90,0,0,50",
            'right': "0,0,0,0,90,0,50",
            'perspective': "20,20,20,55,0,25,100"
        }
    
    def generate_preview(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a preview image for a SCAD file.
        
        Args:
            scad_file: Path to the SCAD file
            parameters: Optional parameters to override in the SCAD file
            
        Returns:
            Path to the generated preview image
        """
        try:
            # Try to generate a preview using OpenSCAD
            preview_file = self.openscad_wrapper.generate_preview(
                scad_file, 
                parameters, 
                camera_position=self.camera_angles['perspective'],
                image_size="800,600"
            )
            
            # Check if the file exists and has content
            if os.path.exists(preview_file) and os.path.getsize(preview_file) > 0:
                return preview_file
            else:
                # If the file doesn't exist or is empty, create a placeholder
                return self._create_placeholder_image(preview_file)
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            # Create a placeholder image
            model_id = os.path.basename(scad_file).split('.')[0]
            preview_file = os.path.join(self.openscad_wrapper.preview_dir, f"{model_id}.png")
            return self._create_placeholder_image(preview_file)
    
    def generate_multi_angle_previews(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate preview images from multiple angles.
        
        Args:
            scad_file: Path to the SCAD file
            parameters: Optional parameters to override in the SCAD file
            
        Returns:
            Dictionary mapping angle names to preview image paths
        """
        previews = {}
        model_id = os.path.basename(scad_file).split('.')[0]
        
        for angle_name, camera_position in self.camera_angles.items():
            preview_file = os.path.join(
                self.openscad_wrapper.preview_dir, 
                f"{model_id}_{angle_name}.png"
            )
            
            try:
                # Try to generate a preview using OpenSCAD
                preview_file = self.openscad_wrapper.generate_preview(
                    scad_file, 
                    parameters, 
                    camera_position=camera_position,
                    image_size="800,600"
                )
                
                # Check if the file exists and has content
                if os.path.exists(preview_file) and os.path.getsize(preview_file) > 0:
                    previews[angle_name] = preview_file
                else:
                    # If the file doesn't exist or is empty, create a placeholder
                    previews[angle_name] = self._create_placeholder_image(preview_file, angle_name)
            except Exception as e:
                logger.error(f"Error generating {angle_name} preview: {str(e)}")
                # Create a placeholder image
                previews[angle_name] = self._create_placeholder_image(preview_file, angle_name)
        
        return previews
    
    def _create_placeholder_image(self, output_path: str, angle_name: str = "perspective") -> str:
        """
        Create a placeholder image when OpenSCAD rendering fails.
        
        Args:
            output_path: Path to save the placeholder image
            angle_name: Name of the camera angle for the placeholder
            
        Returns:
            Path to the created placeholder image
        """
        try:
            # Create a blank image
            img = Image.new('RGB', (800, 600), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Add text
            draw.text((400, 280), f"Preview not available", fill=(0, 0, 0))
            draw.text((400, 320), f"View: {angle_name}", fill=(0, 0, 0))
            
            # Save the image
            img.save(output_path)
            logger.info(f"Created placeholder image: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating placeholder image: {str(e)}")
            # If all else fails, return the path anyway
            return output_path
    
    def create_composite_preview(self, previews: Dict[str, str], output_path: str) -> str:
        """
        Create a composite image from multiple angle previews.
        
        Args:
            previews: Dictionary mapping angle names to preview image paths
            output_path: Path to save the composite image
            
        Returns:
            Path to the created composite image
        """
        try:
            # Create a blank image
            img = Image.new('RGB', (1600, 1200), color=(240, 240, 240))
            
            # Load and paste each preview
            positions = {
                'perspective': (0, 0),
                'front': (800, 0),
                'top': (0, 600),
                'right': (800, 600)
            }
            
            for angle_name, preview_path in previews.items():
                if angle_name in positions and os.path.exists(preview_path):
                    try:
                        angle_img = Image.open(preview_path)
                        # Resize if needed
                        angle_img = angle_img.resize((800, 600))
                        # Paste into composite
                        img.paste(angle_img, positions[angle_name])
                    except Exception as e:
                        logger.error(f"Error processing {angle_name} preview: {str(e)}")
            
            # Save the composite image
            img.save(output_path)
            logger.info(f"Created composite preview: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating composite preview: {str(e)}")
            # If all else fails, return the path anyway
            return output_path
