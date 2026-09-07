import os
import logging
import uuid
import shutil
from typing import Dict, Any, Optional, Tuple

from src.utils.stl_validator import STLValidator

logger = logging.getLogger(__name__)

class STLExporter:
    """
    Handles STL file export and validation for 3D printing.
    """
    
    def __init__(self, openscad_wrapper, output_dir: str):
        """
        Initialize the STL exporter.
        
        Args:
            openscad_wrapper: Instance of OpenSCADWrapper for generating STL files
            output_dir: Directory to store output files
        """
        self.openscad_wrapper = openscad_wrapper
        self.output_dir = output_dir
        self.stl_dir = os.path.join(output_dir, "stl")
        
        # Create directory if it doesn't exist
        os.makedirs(self.stl_dir, exist_ok=True)
    
    def export_stl(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, Optional[str]]:
        """
        Export a SCAD file to STL format.
        
        Args:
            scad_file: Path to the SCAD file
            parameters: Optional parameters to override in the SCAD file
            
        Returns:
            Tuple of (stl_file_path, is_valid, error_message)
        """
        try:
            # Generate STL file
            stl_file = self.openscad_wrapper.generate_stl(scad_file, parameters)
            
            # Validate STL file
            is_valid, error = STLValidator.validate_stl(stl_file)
            
            if not is_valid:
                logger.warning(f"STL validation failed: {error}")
                
                # Attempt to repair if validation fails
                repair_success, repair_error = STLValidator.repair_stl(stl_file)
                if repair_success:
                    # Validate again after repair
                    is_valid, error = STLValidator.validate_stl(stl_file)
                else:
                    logger.error(f"STL repair failed: {repair_error}")
            
            return stl_file, is_valid, error
        except Exception as e:
            logger.error(f"Error exporting STL: {str(e)}")
            return "", False, str(e)
    
    def export_stl_with_metadata(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None, 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export a SCAD file to STL format and include metadata.
        
        Args:
            scad_file: Path to the SCAD file
            parameters: Optional parameters to override in the SCAD file
            metadata: Optional metadata to include with the STL file
            
        Returns:
            Dictionary with STL file information
        """
        # Export STL file
        stl_file, is_valid, error = self.export_stl(scad_file, parameters)
        
        # Create metadata file if metadata is provided
        metadata_file = None
        if metadata and stl_file:
            metadata_file = self._create_metadata_file(stl_file, metadata)
        
        # Extract model ID from filename
        model_id = os.path.basename(scad_file).split('.')[0] if scad_file else str(uuid.uuid4())
        
        return {
            "model_id": model_id,
            "stl_file": stl_file,
            "is_valid": is_valid,
            "error": error,
            "metadata_file": metadata_file,
            "metadata": metadata
        }
    
    def _create_metadata_file(self, stl_file: str, metadata: Dict[str, Any]) -> str:
        """Create a metadata file for an STL file."""
        metadata_file = f"{os.path.splitext(stl_file)[0]}.json"
        
        try:
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created metadata file: {metadata_file}")
            return metadata_file
        except Exception as e:
            logger.error(f"Error creating metadata file: {str(e)}")
            return ""
    
    def copy_stl_to_location(self, stl_file: str, destination: str) -> str:
        """
        Copy an STL file to a specified location.
        
        Args:
            stl_file: Path to the STL file
            destination: Destination path or directory
            
        Returns:
            Path to the copied STL file
        """
        try:
            if not os.path.exists(stl_file):
                raise FileNotFoundError(f"STL file not found: {stl_file}")
            
            # If destination is a directory, create a filename
            if os.path.isdir(destination):
                filename = os.path.basename(stl_file)
                destination = os.path.join(destination, filename)
            
            # Copy the file
            shutil.copy2(stl_file, destination)
            logger.info(f"Copied STL file to: {destination}")
            
            return destination
        except Exception as e:
            logger.error(f"Error copying STL file: {str(e)}")
            return ""
