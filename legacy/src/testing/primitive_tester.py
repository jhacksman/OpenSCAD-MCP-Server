import os
import logging
from typing import Dict, Any, List, Optional, Tuple

from src.models.code_generator import OpenSCADCodeGenerator
from src.utils.cad_exporter import CADExporter

logger = logging.getLogger(__name__)

class PrimitiveTester:
    """Tests OpenSCAD primitives with different export formats."""
    
    def __init__(self, code_generator: OpenSCADCodeGenerator, cad_exporter: CADExporter, 
                output_dir: str = "test_output"):
        """
        Initialize the primitive tester.
        
        Args:
            code_generator: CodeGenerator instance for generating OpenSCAD code
            cad_exporter: CADExporter instance for exporting models
            output_dir: Directory to store test output
        """
        self.code_generator = code_generator
        self.cad_exporter = cad_exporter
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Primitive types to test
        self.primitives = [
            "cube", "sphere", "cylinder", "cone", "torus", 
            "rounded_box", "hexagonal_prism", "text"
        ]
        
        # Export formats to test (no STL per requirements)
        self.formats = ["3mf", "amf", "csg", "scad"]
        
    def test_all_primitives(self) -> Dict[str, Dict[str, Any]]:
        """
        Test all primitives with all formats.
        
        Returns:
            Dictionary of test results for each primitive
        """
        results = {}
        
        for primitive in self.primitives:
            results[primitive] = self.test_primitive(primitive)
            
        return results
        
    def test_primitive(self, primitive_type: str) -> Dict[str, Any]:
        """
        Test a single primitive with all formats.
        
        Args:
            primitive_type: Type of primitive to test
            
        Returns:
            Dictionary of test results for the primitive
        """
        results = {
            "primitive": primitive_type,
            "formats": {}
        }
        
        # Generate default parameters for the primitive
        params = self._get_default_parameters(primitive_type)
        
        # Generate the SCAD code
        scad_file = self.code_generator.generate_code(primitive_type, params)
        
        # Test export to each format
        for format_type in self.formats:
            success, output_file, error = self.cad_exporter.export_model(
                scad_file, 
                format_type,
                params,
                metadata={"primitive_type": primitive_type}
            )
            
            results["formats"][format_type] = {
                "success": success,
                "output_file": output_file,
                "error": error
            }
            
        return results
        
    def _get_default_parameters(self, primitive_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a primitive type.
        
        Args:
            primitive_type: Type of primitive
            
        Returns:
            Dictionary of default parameters
        """
        params = {}
        
        if primitive_type == "cube":
            params = {"width": 20, "depth": 20, "height": 20, "center": True}
        elif primitive_type == "sphere":
            params = {"radius": 10, "segments": 32}
        elif primitive_type == "cylinder":
            params = {"radius": 10, "height": 20, "center": True, "segments": 32}
        elif primitive_type == "cone":
            params = {"bottom_radius": 10, "top_radius": 0, "height": 20, "center": True}
        elif primitive_type == "torus":
            params = {"outer_radius": 20, "inner_radius": 5, "segments": 32}
        elif primitive_type == "rounded_box":
            params = {"width": 30, "depth": 20, "height": 15, "radius": 3}
        elif primitive_type == "hexagonal_prism":
            params = {"radius": 10, "height": 20}
        elif primitive_type == "text":
            params = {"text": "OpenSCAD", "size": 10, "height": 3}
        
        return params
    
    def test_with_parameter_variations(self, primitive_type: str) -> Dict[str, Any]:
        """
        Test a primitive with variations of parameters.
        
        Args:
            primitive_type: Type of primitive to test
            
        Returns:
            Dictionary of test results for different parameter variations
        """
        results = {
            "primitive": primitive_type,
            "variations": {}
        }
        
        # Define parameter variations for the primitive
        variations = self._get_parameter_variations(primitive_type)
        
        # Test each variation
        for variation_name, params in variations.items():
            # Generate the SCAD code
            scad_file = self.code_generator.generate_code(primitive_type, params)
            
            # Test export to each format
            format_results = {}
            for format_type in self.formats:
                success, output_file, error = self.cad_exporter.export_model(
                    scad_file, 
                    format_type,
                    params,
                    metadata={"primitive_type": primitive_type, "variation": variation_name}
                )
                
                format_results[format_type] = {
                    "success": success,
                    "output_file": output_file,
                    "error": error
                }
            
            results["variations"][variation_name] = {
                "parameters": params,
                "formats": format_results
            }
            
        return results
    
    def _get_parameter_variations(self, primitive_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter variations for a primitive type.
        
        Args:
            primitive_type: Type of primitive
            
        Returns:
            Dictionary of parameter variations
        """
        variations = {}
        
        if primitive_type == "cube":
            variations = {
                "small": {"width": 5, "depth": 5, "height": 5, "center": True},
                "large": {"width": 50, "depth": 50, "height": 50, "center": True},
                "flat": {"width": 50, "depth": 50, "height": 2, "center": True},
                "tall": {"width": 10, "depth": 10, "height": 100, "center": True}
            }
        elif primitive_type == "sphere":
            variations = {
                "small": {"radius": 2, "segments": 16},
                "large": {"radius": 30, "segments": 64},
                "low_res": {"radius": 10, "segments": 8},
                "high_res": {"radius": 10, "segments": 128}
            }
        elif primitive_type == "cylinder":
            variations = {
                "small": {"radius": 2, "height": 5, "center": True, "segments": 16},
                "large": {"radius": 30, "height": 50, "center": True, "segments": 64},
                "thin": {"radius": 1, "height": 50, "center": True, "segments": 32},
                "disc": {"radius": 30, "height": 2, "center": True, "segments": 32}
            }
        # Add variations for other primitives as needed
        
        return variations
