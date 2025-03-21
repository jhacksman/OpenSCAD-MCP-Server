import os
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

class CADExporter:
    """
    Exports OpenSCAD models to various CAD formats that preserve parametric properties.
    """
    
    def __init__(self, openscad_path: str = "openscad"):
        """
        Initialize the CAD exporter.
        
        Args:
            openscad_path: Path to the OpenSCAD executable
        """
        self.openscad_path = openscad_path
        
        # Supported export formats
        self.supported_formats = {
            "csg": "OpenSCAD CSG format (preserves all parametric properties)",
            "amf": "Additive Manufacturing File Format (preserves some metadata)",
            "3mf": "3D Manufacturing Format (modern replacement for STL with metadata)",
            "scad": "OpenSCAD source code (fully parametric)",
            "dxf": "Drawing Exchange Format (for 2D designs)",
            "svg": "Scalable Vector Graphics (for 2D designs)"
        }
    
    def export_model(self, scad_file: str, output_format: str = "csg", 
                    parameters: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Export an OpenSCAD model to the specified format.
        
        Args:
            scad_file: Path to the SCAD file
            output_format: Format to export to (csg, amf, 3mf, etc.)
            parameters: Optional parameters to override in the SCAD file
            metadata: Optional metadata to include in the export
            
        Returns:
            Tuple of (success, output_file_path, error_message)
        """
        if not os.path.exists(scad_file):
            return False, "", f"SCAD file not found: {scad_file}"
        
        # Create output file path
        output_dir = os.path.dirname(scad_file)
        model_id = os.path.basename(scad_file).split('.')[0]
        
        # Special case for SCAD format - just copy the file with parameters embedded
        if output_format.lower() == "scad" and parameters:
            return self._export_parametric_scad(scad_file, parameters, metadata)
        
        # For native OpenSCAD formats
        output_file = os.path.join(output_dir, f"{model_id}.{output_format.lower()}")
        
        # Build command
        cmd = [self.openscad_path, "-o", output_file]
        
        # Add parameters if provided
        if parameters:
            for key, value in parameters.items():
                cmd.extend(["-D", f"{key}={value}"])
        
        # Add input file
        cmd.append(scad_file)
        
        try:
            # Run OpenSCAD
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check if file was created
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Exported model to {output_format}: {output_file}")
                
                # Add metadata if supported and provided
                if metadata and output_format.lower() in ["amf", "3mf"]:
                    self._add_metadata_to_file(output_file, metadata, output_format)
                
                return True, output_file, None
            else:
                error_msg = f"Failed to export model to {output_format}"
                logger.error(error_msg)
                logger.error(f"OpenSCAD output: {result.stdout}")
                logger.error(f"OpenSCAD error: {result.stderr}")
                return False, "", error_msg
        except subprocess.CalledProcessError as e:
            error_msg = f"Error exporting model to {output_format}: {e.stderr}"
            logger.error(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Error exporting model to {output_format}: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def _export_parametric_scad(self, scad_file: str, parameters: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Create a new SCAD file with parameters embedded as variables.
        
        Args:
            scad_file: Path to the original SCAD file
            parameters: Parameters to embed in the SCAD file
            metadata: Optional metadata to include as comments
            
        Returns:
            Tuple of (success, output_file_path, error_message)
        """
        try:
            # Read the original SCAD file
            with open(scad_file, 'r') as f:
                content = f.read()
            
            # Create output file path
            output_dir = os.path.dirname(scad_file)
            model_id = os.path.basename(scad_file).split('.')[0]
            output_file = os.path.join(output_dir, f"{model_id}_parametric.scad")
            
            # Create parameter declarations
            param_declarations = []
            for key, value in parameters.items():
                if isinstance(value, str):
                    param_declarations.append(f'{key} = "{value}";')
                else:
                    param_declarations.append(f'{key} = {value};')
            
            # Create metadata comments
            metadata_comments = []
            if metadata:
                metadata_comments.append("// Metadata:")
                for key, value in metadata.items():
                    metadata_comments.append(f"// {key}: {value}")
            
            # Combine everything
            new_content = "// Parametric model generated by OpenSCAD MCP Server\n"
            new_content += "\n".join(metadata_comments) + "\n\n" if metadata_comments else "\n"
            new_content += "// Parameters:\n"
            new_content += "\n".join(param_declarations) + "\n\n"
            new_content += content
            
            # Write to the new file
            with open(output_file, 'w') as f:
                f.write(new_content)
            
            logger.info(f"Exported parametric SCAD file: {output_file}")
            return True, output_file, None
        except Exception as e:
            error_msg = f"Error creating parametric SCAD file: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def _add_metadata_to_file(self, file_path: str, metadata: Dict[str, Any], format_type: str) -> None:
        """
        Add metadata to supported file formats.
        
        Args:
            file_path: Path to the file
            metadata: Metadata to add
            format_type: File format
        """
        if format_type.lower() == "amf":
            self._add_metadata_to_amf(file_path, metadata)
        elif format_type.lower() == "3mf":
            self._add_metadata_to_3mf(file_path, metadata)
    
    def _add_metadata_to_amf(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Add metadata to AMF file."""
        try:
            import xml.etree.ElementTree as ET
            
            # Parse the AMF file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find or create metadata element
            metadata_elem = root.find("metadata")
            if metadata_elem is None:
                metadata_elem = ET.SubElement(root, "metadata")
            
            # Add metadata
            for key, value in metadata.items():
                meta = ET.SubElement(metadata_elem, "meta", name=key)
                meta.text = str(value)
            
            # Write back to file
            tree.write(file_path)
            logger.info(f"Added metadata to AMF file: {file_path}")
        except Exception as e:
            logger.error(f"Error adding metadata to AMF file: {str(e)}")
    
    def _add_metadata_to_3mf(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Add metadata to 3MF file."""
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            # 3MF files are ZIP archives
            with zipfile.ZipFile(file_path, 'a') as z:
                # Check if metadata file exists
                metadata_path = "Metadata/model_metadata.xml"
                try:
                    z.getinfo(metadata_path)
                    # Extract existing metadata
                    with z.open(metadata_path) as f:
                        tree = ET.parse(f)
                        root = tree.getroot()
                except KeyError:
                    # Create new metadata file
                    root = ET.Element("metadata")
                    tree = ET.ElementTree(root)
                
                # Add metadata
                for key, value in metadata.items():
                    meta = ET.SubElement(root, "meta", name=key)
                    meta.text = str(value)
                
                # Write metadata to a temporary file
                temp_path = file_path + ".metadata.tmp"
                tree.write(temp_path)
                
                # Add to ZIP
                z.write(temp_path, metadata_path)
                
                # Remove temporary file
                os.remove(temp_path)
            
            logger.info(f"Added metadata to 3MF file: {file_path}")
        except Exception as e:
            logger.error(f"Error adding metadata to 3MF file: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return list(self.supported_formats.keys())
    
    def get_format_description(self, format_name: str) -> str:
        """Get description of a format."""
        return self.supported_formats.get(format_name.lower(), "Unknown format")
