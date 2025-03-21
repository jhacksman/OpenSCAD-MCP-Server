"""
threestudio integration for 3D model generation from images.
"""

import os
import subprocess
import logging
import json
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ThreeStudioGenerator:
    """
    Wrapper for threestudio for 3D model generation from images.
    """
    
    def __init__(self, threestudio_path: str, output_dir: str = "output/models"):
        """
        Initialize the threestudio generator.
        
        Args:
            threestudio_path: Path to threestudio installation
            output_dir: Directory to store output files
        """
        self.threestudio_path = threestudio_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate threestudio installation
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """
        Validate threestudio installation.
        
        Raises:
            FileNotFoundError: If threestudio installation is not found
        """
        if not os.path.exists(self.threestudio_path):
            raise FileNotFoundError(f"threestudio not found at {self.threestudio_path}")
        
        # Check for required files
        required_files = ["launch.py", "README.md"]
        for file in required_files:
            if not os.path.exists(os.path.join(self.threestudio_path, file)):
                raise FileNotFoundError(f"Required file {file} not found in threestudio directory")
    
    def generate_model_from_image(self, image_path: str, method: str = "zero123",
                                 num_iterations: int = 5000, export_format: str = "obj",
                                 config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a 3D model from an image using threestudio.
        
        Args:
            image_path: Path to input image
            method: Method to use ("zero123", "sjc", "magic3d", etc.)
            num_iterations: Number of training iterations
            export_format: Format to export ("obj", "glb", "ply")
            config_overrides: Optional configuration overrides
            
        Returns:
            Dictionary containing paths to generated model files
        """
        try:
            # Create a unique ID for this generation
            model_id = Path(image_path).stem
            
            # Create a temporary config file
            config_file = self._create_config_file(image_path, method, num_iterations, config_overrides)
            
            # Run threestudio
            output_dir = os.path.join(self.output_dir, model_id)
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                "python", "launch.py",
                "--config", config_file,
                "--train",
                "--gpu", "0",
                "--output_dir", output_dir
            ]
            
            logger.info(f"Running threestudio with command: {' '.join(cmd)}")
            
            # Execute in threestudio directory
            process = subprocess.Popen(
                cmd,
                cwd=self.threestudio_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error running threestudio: {stderr}")
                raise RuntimeError(f"threestudio failed with exit code {process.returncode}")
            
            # Export model
            exported_files = self._export_model(output_dir, export_format)
            
            return {
                "model_id": model_id,
                "output_dir": output_dir,
                "exported_files": exported_files,
                "preview_images": self._get_preview_images(output_dir)
            }
        except Exception as e:
            logger.error(f"Error generating 3D model with threestudio: {str(e)}")
            raise
    
    def _create_config_file(self, image_path: str, method: str, num_iterations: int,
                          config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a configuration file for threestudio.
        
        Args:
            image_path: Path to input image
            method: Method to use
            num_iterations: Number of training iterations
            config_overrides: Optional configuration overrides
            
        Returns:
            Path to the created configuration file
        """
        # Base configuration
        config = {
            "method": method,
            "image_path": os.path.abspath(image_path),
            "num_iterations": num_iterations,
            "save_interval": 1000,
            "export_interval": 1000
        }
        
        # Apply overrides
        if config_overrides:
            config.update(config_overrides)
        
        # Write to temporary file
        fd, config_file = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def _export_model(self, output_dir: str, export_format: str) -> List[str]:
        """
        Export the model in the specified format.
        
        Args:
            output_dir: Directory containing the model
            export_format: Format to export
            
        Returns:
            List of paths to exported files
        """
        # Find the latest checkpoint
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
        
        # Get the latest checkpoint
        checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")])
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        
        latest_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])
        
        # Export command
        cmd = [
            "python", "launch.py",
            "--config", os.path.join(output_dir, "config.yaml"),
            "--export",
            "--gpu", "0",
            "--checkpoint", latest_checkpoint,
            "--export_format", export_format
        ]
        
        logger.info(f"Exporting model with command: {' '.join(cmd)}")
        
        # Execute in threestudio directory
        process = subprocess.Popen(
            cmd,
            cwd=self.threestudio_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error exporting model: {stderr}")
            raise RuntimeError(f"Model export failed with exit code {process.returncode}")
        
        # Find exported files
        exports_dir = os.path.join(output_dir, "exports")
        if not os.path.exists(exports_dir):
            raise FileNotFoundError(f"Exports directory not found: {exports_dir}")
        
        exported_files = [os.path.join(exports_dir, f) for f in os.listdir(exports_dir)]
        
        return exported_files
    
    def _get_preview_images(self, output_dir: str) -> List[str]:
        """
        Get paths to preview images.
        
        Args:
            output_dir: Directory containing the model
            
        Returns:
            List of paths to preview images
        """
        # Find preview images
        previews_dir = os.path.join(output_dir, "images")
        if not os.path.exists(previews_dir):
            return []
        
        preview_images = [os.path.join(previews_dir, f) for f in os.listdir(previews_dir) 
                         if f.endswith(".png") or f.endswith(".jpg")]
        
        return sorted(preview_images)
