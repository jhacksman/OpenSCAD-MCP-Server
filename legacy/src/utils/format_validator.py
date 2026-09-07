import os
import logging
import zipfile
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class FormatValidator:
    """Validates 3D model formats for compatibility with printers."""
    
    @staticmethod
    def validate_3mf(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a 3MF file for compatibility with Prusa and Bambu printers.
        
        Args:
            file_path: Path to the 3MF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        try:
            # 3MF files are ZIP archives with XML content
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check for required files
                required_files = ['3D/3dmodel.model', '[Content_Types].xml']
                for req_file in required_files:
                    try:
                        zip_ref.getinfo(req_file)
                    except KeyError:
                        return False, f"Missing required file in 3MF: {req_file}"
                
                # Validate 3D model file
                with zip_ref.open('3D/3dmodel.model') as model_file:
                    tree = ET.parse(model_file)
                    root = tree.getroot()
                    
                    # Check for required elements
                    if root.tag != '{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}model':
                        return False, "Invalid 3MF: Missing model element"
                    
                    # Verify resources section exists
                    resources = root.find('.//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}resources')
                    if resources is None:
                        return False, "Invalid 3MF: Missing resources element"
                    
            return True, None
        except Exception as e:
            logger.error(f"Error validating 3MF file: {str(e)}")
            return False, f"Error validating 3MF file: {str(e)}"
    
    @staticmethod
    def validate_amf(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an AMF file for compatibility with printers.
        
        Args:
            file_path: Path to the AMF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        try:
            # Parse the AMF file (XML format)
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Check for required elements
            if root.tag != 'amf':
                return False, "Invalid AMF: Missing amf root element"
            
            # Check for at least one object
            objects = root.findall('./object')
            if not objects:
                return False, "Invalid AMF: No objects found"
            
            # Check that each object has a mesh
            for obj in objects:
                mesh = obj.find('./mesh')
                if mesh is None:
                    return False, f"Invalid AMF: Object {obj.get('id', 'unknown')} is missing a mesh"
                
                # Check for vertices and volumes
                vertices = mesh.find('./vertices')
                volumes = mesh.findall('./volume')
                
                if vertices is None:
                    return False, f"Invalid AMF: Mesh in object {obj.get('id', 'unknown')} is missing vertices"
                
                if not volumes:
                    return False, f"Invalid AMF: Mesh in object {obj.get('id', 'unknown')} has no volumes"
            
            return True, None
        except Exception as e:
            logger.error(f"Error validating AMF file: {str(e)}")
            return False, f"Error validating AMF file: {str(e)}"
    
    @staticmethod
    def extract_metadata(file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a 3MF or AMF file.
        
        Args:
            file_path: Path to the 3D model file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.3mf':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    metadata_path = "Metadata/model_metadata.xml"
                    try:
                        with zip_ref.open(metadata_path) as f:
                            tree = ET.parse(f)
                            root = tree.getroot()
                            
                            for meta in root.findall('./meta'):
                                name = meta.get('name')
                                if name:
                                    metadata[name] = meta.text
                    except KeyError:
                        # Metadata file doesn't exist
                        pass
            
            elif ext == '.amf':
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                for meta in root.findall('./metadata'):
                    name = meta.get('type')
                    if name:
                        metadata[name] = meta.text
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    @staticmethod
    def check_printer_compatibility(file_path: str, printer_type: str = "prusa") -> Tuple[bool, Optional[str]]:
        """
        Check if a 3D model file is compatible with a specific printer type.
        
        Args:
            file_path: Path to the 3D model file
            printer_type: Type of printer ("prusa" or "bambu")
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Validate based on file format
        if ext == '.3mf':
            is_valid, error = FormatValidator.validate_3mf(file_path)
            if not is_valid:
                return False, error
            
            # Additional printer-specific checks
            if printer_type.lower() == "prusa":
                # Prusa-specific checks for 3MF
                # For now, just basic validation is sufficient
                return True, None
            
            elif printer_type.lower() == "bambu":
                # Bambu-specific checks for 3MF
                # For now, just basic validation is sufficient
                return True, None
            
            else:
                return False, f"Unknown printer type: {printer_type}"
            
        elif ext == '.amf':
            is_valid, error = FormatValidator.validate_amf(file_path)
            if not is_valid:
                return False, error
            
            # Additional printer-specific checks
            if printer_type.lower() == "prusa":
                # Prusa-specific checks for AMF
                # For now, just basic validation is sufficient
                return True, None
            
            elif printer_type.lower() == "bambu":
                # Bambu-specific checks for AMF
                # For now, just basic validation is sufficient
                return True, None
            
            else:
                return False, f"Unknown printer type: {printer_type}"
            
        else:
            return False, f"Unsupported file format for printer compatibility check: {ext}"
