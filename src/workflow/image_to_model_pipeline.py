"""
Workflow orchestration for the image-to-model pipeline.
"""

import os
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageToModelPipeline:
    """
    Orchestrates the workflow from text prompt to 3D model:
    1. Generate image with Venice.ai
    2. Segment object with SAM2
    3. Create 3D model with threestudio
    4. Convert to OpenSCAD for parametric editing
    """
    
    def __init__(self, 
                venice_generator,
                sam_segmenter,
                threestudio_generator,
                openscad_wrapper,
                output_dir: str = "output/pipeline"):
        """
        Initialize the pipeline.
        
        Args:
            venice_generator: Instance of VeniceImageGenerator
            sam_segmenter: Instance of SAMSegmenter
            threestudio_generator: Instance of ThreeStudioGenerator
            openscad_wrapper: Instance of OpenSCADWrapper
            output_dir: Directory to store output files
        """
        self.venice_generator = venice_generator
        self.sam_segmenter = sam_segmenter
        self.threestudio_generator = threestudio_generator
        self.openscad_wrapper = openscad_wrapper
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "scad"), exist_ok=True)
    
    def generate_model_from_text(self, prompt: str, 
                               venice_params: Optional[Dict[str, Any]] = None,
                               sam_params: Optional[Dict[str, Any]] = None,
                               threestudio_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a 3D model from a text prompt.
        
        Args:
            prompt: Text description for image generation
            venice_params: Optional parameters for Venice.ai
            sam_params: Optional parameters for SAM2
            threestudio_params: Optional parameters for threestudio
            
        Returns:
            Dictionary containing paths to generated files and metadata
        """
        try:
            # Generate a unique ID for this pipeline run
            pipeline_id = str(uuid.uuid4())
            logger.info(f"Starting pipeline {pipeline_id} for prompt: {prompt}")
            
            # Step 1: Generate image with Venice.ai
            image_path = os.path.join(self.output_dir, "images", f"{pipeline_id}.png")
            venice_result = self._generate_image(prompt, image_path, venice_params)
            
            # Step 2: Segment object with SAM2
            masks_dir = os.path.join(self.output_dir, "masks", pipeline_id)
            sam_result = self._segment_image(image_path, masks_dir, sam_params)
            
            # Get the best mask (highest score)
            best_mask_idx = sam_result["scores"].index(max(sam_result["scores"]))
            best_mask_path = sam_result["masked_images"][best_mask_idx]
            
            # Step 3: Create 3D model with threestudio
            threestudio_result = self._generate_3d_model(best_mask_path, threestudio_params)
            
            # Step 4: Convert to OpenSCAD for parametric editing
            scad_result = self._convert_to_openscad(threestudio_result["exported_files"][0], pipeline_id)
            
            # Compile results
            result = {
                "pipeline_id": pipeline_id,
                "prompt": prompt,
                "image": venice_result,
                "segmentation": sam_result,
                "model_3d": threestudio_result,
                "openscad": scad_result
            }
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise
    
    def _generate_image(self, prompt: str, output_path: str, 
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate image with Venice.ai.
        
        Args:
            prompt: Text description for image generation
            output_path: Path to save the generated image
            params: Optional parameters for Venice.ai
            
        Returns:
            Dictionary containing image data and metadata
        """
        logger.info(f"Generating image for prompt: {prompt}")
        
        # Default parameters
        default_params = {
            "model": "flux",
            "width": 1024,
            "height": 1024
        }
        
        # Merge with provided parameters
        if params:
            default_params.update(params)
        
        # Generate image
        result = self.venice_generator.generate_image(
            prompt=prompt,
            output_path=output_path,
            **default_params
        )
        
        logger.info(f"Image generated: {output_path}")
        return result
    
    def _segment_image(self, image_path: str, output_dir: str,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Segment object with SAM2.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save segmentation results
            params: Optional parameters for SAM2
            
        Returns:
            Dictionary containing segmentation masks and metadata
        """
        logger.info(f"Segmenting image: {image_path}")
        
        # Segment image
        result = self.sam_segmenter.segment_image(
            image_path=image_path,
            output_dir=output_dir,
            **(params or {})
        )
        
        logger.info(f"Image segmented, {result['mask_count']} masks generated")
        return result
    
    def _generate_3d_model(self, image_path: str,
                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate 3D model with threestudio.
        
        Args:
            image_path: Path to input image
            params: Optional parameters for threestudio
            
        Returns:
            Dictionary containing paths to generated model files
        """
        logger.info(f"Generating 3D model from image: {image_path}")
        
        # Default parameters
        default_params = {
            "method": "zero123",
            "num_iterations": 5000,
            "export_format": "obj"
        }
        
        # Merge with provided parameters
        if params:
            default_params.update(params)
        
        # Generate 3D model
        result = self.threestudio_generator.generate_model_from_image(
            image_path=image_path,
            **default_params
        )
        
        logger.info(f"3D model generated: {result['exported_files']}")
        return result
    
    def _convert_to_openscad(self, model_path: str, model_id: str) -> Dict[str, Any]:
        """
        Convert 3D model to OpenSCAD format.
        
        Args:
            model_path: Path to input model
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary containing paths to generated files
        """
        logger.info(f"Converting model to OpenSCAD: {model_path}")
        
        # Generate OpenSCAD code for importing the model
        scad_code = f"""// Generated OpenSCAD code for model {model_id}
// Imported from {os.path.basename(model_path)}

// Parameters
scale_factor = 1.0;
position_x = 0;
position_y = 0;
position_z = 0;
rotation_x = 0;
rotation_y = 0;
rotation_z = 0;

// Import and transform the model
translate([position_x, position_y, position_z])
rotate([rotation_x, rotation_y, rotation_z])
scale(scale_factor)
import("{model_path}");
"""
        
        # Save SCAD code to file
        scad_file = self.openscad_wrapper.generate_scad(scad_code, model_id)
        
        # Generate previews
        previews = self.openscad_wrapper.generate_multi_angle_previews(scad_file)
        
        return {
            "scad_file": scad_file,
            "previews": previews,
            "model_path": model_path
        }
