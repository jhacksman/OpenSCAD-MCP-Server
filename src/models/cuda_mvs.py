"""
CUDA Multi-View Stereo wrapper for 3D reconstruction from multiple images.
"""

import os
import subprocess
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CUDAMultiViewStereo:
    """
    Wrapper for CUDA Multi-View Stereo for 3D reconstruction from multiple images.
    """
    
    def __init__(self, cuda_mvs_path: str, output_dir: str = "output/models"):
        """
        Initialize the CUDA MVS wrapper.
        
        Args:
            cuda_mvs_path: Path to CUDA MVS installation
            output_dir: Directory to store output files
        """
        self.cuda_mvs_path = cuda_mvs_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate installation
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """
        Validate CUDA MVS installation.
        
        Raises:
            FileNotFoundError: If CUDA MVS installation is not found
        """
        if not os.path.exists(self.cuda_mvs_path):
            raise FileNotFoundError(f"CUDA MVS not found at {self.cuda_mvs_path}")
        
        # Check for required executables
        required_files = ["app_patch_match_mvs"]
        for file in required_files:
            exec_path = os.path.join(self.cuda_mvs_path, "build", file)
            if not os.path.exists(exec_path):
                raise FileNotFoundError(f"Required executable {file} not found at {exec_path}")
    
    def generate_model_from_images(self, image_paths: List[str], 
                                camera_params: Optional[Dict[str, Any]] = None,
                                output_name: str = "model") -> Dict[str, Any]:
        """
        Generate a 3D model from multiple images using CUDA MVS.
        
        Args:
            image_paths: List of paths to input images
            camera_params: Optional camera parameters
            output_name: Name for the output files
            
        Returns:
            Dictionary containing paths to generated model files
        """
        try:
            # Create a unique directory for this reconstruction
            model_dir = os.path.join(self.output_dir, output_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a camera parameters file if provided
            params_file = None
            if camera_params:
                params_file = os.path.join(model_dir, "camera_params.json")
                with open(params_file, 'w') as f:
                    json.dump(camera_params, f, indent=2)
            
            # Generate camera parameters if not provided
            if not params_file:
                params_file = self._generate_camera_params(image_paths, model_dir)
            
            # Generate point cloud
            point_cloud_file = os.path.join(model_dir, f"{output_name}.ply")
            
            # Run CUDA MVS
            cmd = [
                os.path.join(self.cuda_mvs_path, "build", "app_patch_match_mvs"),
                "--image_dir", os.path.dirname(image_paths[0]),
                "--camera_params", params_file,
                "--output_file", point_cloud_file
            ]
            
            logger.info(f"Running CUDA MVS with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error running CUDA MVS: {stderr}")
                raise RuntimeError(f"CUDA MVS failed with exit code {process.returncode}")
            
            # Check if output file was created
            if not os.path.exists(point_cloud_file):
                raise FileNotFoundError(f"Output point cloud file not found at {point_cloud_file}")
            
            return {
                "model_id": output_name,
                "output_dir": model_dir,
                "point_cloud_file": point_cloud_file,
                "camera_params_file": params_file,
                "input_images": image_paths
            }
        
        except Exception as e:
            logger.error(f"Error generating 3D model with CUDA MVS: {str(e)}")
            raise
    
    def _generate_camera_params(self, image_paths: List[str], model_dir: str) -> str:
        """
        Generate camera parameters from images.
        
        Args:
            image_paths: List of paths to input images
            model_dir: Directory to save parameter file
            
        Returns:
            Path to camera parameters file
        """
        # This is a simplified version for demonstration
        # In a real implementation, this would use SfM or camera estimation
        
        params = []
        for i, img_path in enumerate(image_paths):
            # Extract image dimensions
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Generate simple camera parameters
            # In reality, these would be estimated from the images
            # or provided by the user
            params.append({
                "image_id": i,
                "image_name": os.path.basename(img_path),
                "width": width,
                "height": height,
                "camera": {
                    "model": "PINHOLE",
                    "focal_length": min(width, height),
                    "principal_point": [width / 2, height / 2],
                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "translation": [0, 0, 0]
                }
            })
        
        # Write parameters to file
        params_file = os.path.join(model_dir, "camera_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        return params_file
    
    def convert_ply_to_obj(self, ply_file: str, output_dir: Optional[str] = None) -> str:
        """
        Convert PLY point cloud to OBJ mesh.
        
        Args:
            ply_file: Path to input PLY file
            output_dir: Directory to save output OBJ file
            
        Returns:
            Path to output OBJ file
        """
        # In a real implementation, this would use a mesh reconstruction library
        # such as Open3D or PyMeshLab to convert the point cloud to a mesh
        
        if not output_dir:
            output_dir = os.path.dirname(ply_file)
        
        # Generate output file path
        obj_file = os.path.join(output_dir, f"{Path(ply_file).stem}.obj")
        
        logger.info(f"Converting PLY to OBJ: {ply_file} -> {obj_file}")
        
        # This is a placeholder for the actual conversion
        # In a real implementation, you would use a library like Open3D:
        # import open3d as o3d
        # pcd = o3d.io.read_point_cloud(ply_file)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)[0]
        # o3d.io.write_triangle_mesh(obj_file, mesh)
        
        # For now, we'll just create a dummy OBJ file
        with open(obj_file, 'w') as f:
            f.write(f"# Converted from {os.path.basename(ply_file)}\n")
            f.write("# This is a placeholder OBJ file\n")
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 0 1 0\n")
            f.write("f 1 2 3\n")
        
        return obj_file
