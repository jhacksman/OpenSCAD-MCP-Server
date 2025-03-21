"""
Workflow orchestration for the multi-view to model pipeline.
"""

import os
import logging
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiViewToModelPipeline:
    """
    Orchestrates the workflow from image or text prompt to 3D model:
    1. Generate image with Venice.ai or Google Gemini (optional)
    2. Generate multiple views with Google Gemini
    3. Process user approval of views
    4. Create 3D model with CUDA Multi-View Stereo
    5. Convert to OpenSCAD for parametric editing (optional)
    """
    
    def __init__(self,
               gemini_generator=None,
               venice_generator=None,
               cuda_mvs=None,
               openscad_wrapper=None,
               approval_tool=None,
               output_dir: str = "output/pipeline"):
        """
        Initialize the pipeline.
        
        Args:
            gemini_generator: Instance of GeminiImageGenerator
            venice_generator: Instance of VeniceImageGenerator (optional)
            cuda_mvs: Instance of CUDAMultiViewStereo
            openscad_wrapper: Instance of OpenSCADWrapper (optional)
            approval_tool: Instance of ImageApprovalTool
            output_dir: Directory to store output files
        """
        self.gemini_generator = gemini_generator
        self.venice_generator = venice_generator
        self.cuda_mvs = cuda_mvs
        self.openscad_wrapper = openscad_wrapper
        self.approval_tool = approval_tool
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "multi_view"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "approved"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "scad"), exist_ok=True)
    
    def generate_model_from_text(self, prompt: str,
                               use_venice: bool = False,
                               num_views: int = 4,
                               gemini_params: Optional[Dict[str, Any]] = None,
                               venice_params: Optional[Dict[str, Any]] = None,
                               cuda_mvs_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a 3D model from a text prompt.
        
        Args:
            prompt: Text description for image generation
            use_venice: Whether to use Venice.ai for initial image
            num_views: Number of views to generate
            gemini_params: Optional parameters for Google Gemini
            venice_params: Optional parameters for Venice.ai
            cuda_mvs_params: Optional parameters for CUDA MVS
            
        Returns:
            Dictionary containing paths to generated files and metadata
        """
        try:
            # Generate a unique ID for this pipeline run
            pipeline_id = str(uuid.uuid4())
            logger.info(f"Starting pipeline {pipeline_id} for prompt: {prompt}")
            
            # Step 1: Generate initial image
            if use_venice and self.venice_generator:
                # Use Venice.ai for initial image
                logger.info("Using Venice.ai for initial image generation")
                image_path = os.path.join(self.output_dir, "images", f"{pipeline_id}_venice.png")
                initial_result = self.venice_generator.generate_image(
                    prompt=prompt,
                    output_path=image_path,
                    **(venice_params or {})
                )
            else:
                # Use Google Gemini for initial image
                logger.info("Using Google Gemini for initial image generation")
                image_path = os.path.join(self.output_dir, "images", f"{pipeline_id}_gemini.png")
                initial_result = self.gemini_generator.generate_image(
                    prompt=prompt,
                    output_path=image_path,
                    **(gemini_params or {})
                )
            
            # Step 2: Generate multiple views
            logger.info(f"Generating {num_views} views with Google Gemini")
            multi_view_dir = os.path.join(self.output_dir, "multi_view", pipeline_id)
            
            multi_views = self.gemini_generator.generate_multiple_views(
                prompt=prompt,
                num_views=num_views,
                base_image_path=image_path,
                output_dir=multi_view_dir
            )
            
            # Step 3: Present images for approval
            # In a real implementation, this would be handled by the MCP client
            # through the MCP tools interface
            logger.info("Preparing images for approval")
            approval_requests = []
            for view in multi_views:
                approval_request = self.approval_tool.present_image_for_approval(
                    image_path=view["local_path"],
                    metadata={
                        "prompt": view.get("prompt"),
                        "view_direction": view.get("view_direction"),
                        "view_index": view.get("view_index")
                    }
                )
                approval_requests.append(approval_request)
            
            # For the purpose of this implementation, we'll assume all views are approved
            # In a real implementation, this would be handled by the MCP client
            approved_images = []
            for req in approval_requests:
                approval_result = self.approval_tool.process_approval(
                    approval_id=req["approval_id"],
                    approved=True,
                    image_path=req["image_path"]
                )
                if approval_result["approved"]:
                    approved_images.append(approval_result["approved_path"])
            
            # Step 4: Generate 3D model with CUDA MVS
            logger.info("Generating 3D model with CUDA MVS")
            model_result = self.cuda_mvs.generate_model_from_images(
                image_paths=approved_images,
                output_name=pipeline_id,
                **(cuda_mvs_params or {})
            )
            
            # Step 5: Convert to OpenSCAD (if wrapper is available)
            scad_result = None
            if self.openscad_wrapper and model_result.get("point_cloud_file"):
                logger.info("Converting to OpenSCAD")
                scad_result = self._convert_to_openscad(model_result["point_cloud_file"], pipeline_id)
            
            # Compile results
            result = {
                "pipeline_id": pipeline_id,
                "prompt": prompt,
                "initial_image": initial_result,
                "multi_views": multi_views,
                "approved_images": approved_images,
                "model_3d": model_result,
            }
            
            if scad_result:
                result["openscad"] = scad_result
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise
    
    def generate_model_from_image(self, image_path: str,
                                prompt: Optional[str] = None,
                                num_views: int = 4,
                                gemini_params: Optional[Dict[str, Any]] = None,
                                cuda_mvs_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a 3D model from an existing image.
        
        Args:
            image_path: Path to input image
            prompt: Optional text description to guide multi-view generation
            num_views: Number of views to generate
            gemini_params: Optional parameters for Google Gemini
            cuda_mvs_params: Optional parameters for CUDA MVS
            
        Returns:
            Dictionary containing paths to generated files and metadata
        """
        try:
            # Generate a unique ID for this pipeline run
            pipeline_id = str(uuid.uuid4())
            logger.info(f"Starting pipeline {pipeline_id} from image: {image_path}")
            
            # Use provided prompt or generate one from the image
            if not prompt:
                # In a real implementation, you might use an image captioning model
                # to generate a description of the image
                prompt = f"3D object in the image {os.path.basename(image_path)}"
            
            # Step 1: Generate multiple views
            logger.info(f"Generating {num_views} views with Google Gemini")
            multi_view_dir = os.path.join(self.output_dir, "multi_view", pipeline_id)
            
            multi_views = self.gemini_generator.generate_multiple_views(
                prompt=prompt,
                num_views=num_views,
                base_image_path=image_path,
                output_dir=multi_view_dir
            )
            
            # Step 2: Present images for approval
            logger.info("Preparing images for approval")
            approval_requests = []
            for view in multi_views:
                approval_request = self.approval_tool.present_image_for_approval(
                    image_path=view["local_path"],
                    metadata={
                        "prompt": view.get("prompt"),
                        "view_direction": view.get("view_direction"),
                        "view_index": view.get("view_index")
                    }
                )
                approval_requests.append(approval_request)
            
            # For the purpose of this implementation, we'll assume all views are approved
            approved_images = []
            for req in approval_requests:
                approval_result = self.approval_tool.process_approval(
                    approval_id=req["approval_id"],
                    approved=True,
                    image_path=req["image_path"]
                )
                if approval_result["approved"]:
                    approved_images.append(approval_result["approved_path"])
            
            # Step 3: Generate 3D model with CUDA MVS
            logger.info("Generating 3D model with CUDA MVS")
            model_result = self.cuda_mvs.generate_model_from_images(
                image_paths=approved_images,
                output_name=pipeline_id,
                **(cuda_mvs_params or {})
            )
            
            # Step 4: Convert to OpenSCAD (if wrapper is available)
            scad_result = None
            if self.openscad_wrapper and model_result.get("point_cloud_file"):
                logger.info("Converting to OpenSCAD")
                scad_result = self._convert_to_openscad(model_result["point_cloud_file"], pipeline_id)
            
            # Compile results
            result = {
                "pipeline_id": pipeline_id,
                "prompt": prompt,
                "input_image": image_path,
                "multi_views": multi_views,
                "approved_images": approved_images,
                "model_3d": model_result,
            }
            
            if scad_result:
                result["openscad"] = scad_result
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise
    
    def _convert_to_openscad(self, model_path: str, model_id: str) -> Dict[str, Any]:
        """
        Convert 3D model to OpenSCAD format.
        
        Args:
            model_path: Path to input model (PLY file)
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary containing paths to generated files
        """
        logger.info(f"Converting model to OpenSCAD: {model_path}")
        
        # Convert PLY to OBJ if needed
        if model_path.endswith('.ply'):
            obj_path = self.cuda_mvs.convert_ply_to_obj(model_path)
            model_path = obj_path
        
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
    
    def process_approval_results(self, approval_results: List[Dict[str, Any]]) -> List[str]:
        """
        Process approval results from the MCP client.
        
        Args:
            approval_results: List of approval results from the client
            
        Returns:
            List of paths to approved images
        """
        approved_images = []
        
        for result in approval_results:
            if result.get("approved", False) and "approved_path" in result:
                approved_images.append(result["approved_path"])
        
        return approved_images
