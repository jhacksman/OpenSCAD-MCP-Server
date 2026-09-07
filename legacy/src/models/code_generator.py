import os
import logging
import uuid
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    Generates OpenSCAD code from natural language descriptions and parameters.
    Implements translation of requirements to OpenSCAD primitives and modules.
    """
    
    def __init__(self, scad_templates_dir: str, output_dir: str, ai_service=None):
        """
        Initialize the code generator.
        
        Args:
            scad_templates_dir: Directory containing SCAD template files
            output_dir: Directory to store generated SCAD files
            ai_service: Optional AI service for enhanced code generation
        """
        self.scad_templates_dir = scad_templates_dir
        self.output_dir = output_dir
        self.ai_service = ai_service
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Map of shape types to their corresponding module names
        self.shape_module_map = {
            'cube': 'parametric_cube',
            'sphere': 'parametric_sphere',
            'cylinder': 'parametric_cylinder',
            'box': 'hollow_box',
            'rounded_box': 'rounded_box',
            'container': 'rounded_container',
            'tube': 'tube',
            'cone': 'cone',
            'wedge': 'wedge',
            'rounded_cylinder': 'rounded_cylinder',
            'torus': 'torus',
            'hexagonal_prism': 'hexagonal_prism',
            'text': 'text_3d',
            'prism': 'triangular_prism',
            'custom': 'custom_shape'
        }
        
        # Parameter mapping from natural language to OpenSCAD parameters
        self.parameter_map = {
            'width': 'width',
            'depth': 'depth',
            'height': 'height',
            'radius': 'radius',
            'thickness': 'thickness',
            'segments': 'segments',
            'center': 'center',
            'inner_radius': 'inner_radius',
            'outer_radius': 'outer_radius',
            'corner_radius': 'corner_radius',
            'text': 'text',
            'size': 'size',
            'font': 'font',
            'base_radius': 'base_radius',
            'major_radius': 'major_radius',
            'minor_radius': 'minor_radius',
            'angle': 'angle',
            'scale': 'scale',
            'resolution': 'resolution'
        }
    
    def generate_code(self, model_type: str, parameters: Dict[str, Any], description: Optional[str] = None) -> str:
        """
        Generate OpenSCAD code for a given model type and parameters.
        
        Args:
            model_type: Type of model to generate
            parameters: Dictionary of parameters for the model
            description: Optional natural language description for AI-driven generation
            
        Returns:
            Path to the generated SCAD file
        """
        # Generate a unique ID for the model
        model_id = str(uuid.uuid4())
        scad_file = os.path.join(self.output_dir, f"{model_id}.scad")
        
        # Check if we should use AI-driven generation for complex models
        if model_type == 'custom' and description and self.ai_service:
            scad_code = self._generate_ai_driven_code(description, parameters)
        else:
            # Get the module name for the model type
            module_name = self.shape_module_map.get(model_type)
            if not module_name:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Map parameters to OpenSCAD parameter names
            scad_params = self._map_parameters(parameters)
            
            # Generate the OpenSCAD code
            scad_code = self._generate_scad_code(module_name, scad_params)
        
        # Write the code to a file
        with open(scad_file, 'w') as f:
            f.write(scad_code)
        
        logger.info(f"Generated OpenSCAD code: {scad_file}")
        return scad_file
    
    def update_code(self, scad_file: str, parameters: Dict[str, Any]) -> str:
        """
        Update an existing SCAD file with new parameters.
        
        Args:
            scad_file: Path to the SCAD file to update
            parameters: New parameters to apply
            
        Returns:
            Path to the updated SCAD file
        """
        if not os.path.exists(scad_file):
            raise FileNotFoundError(f"SCAD file not found: {scad_file}")
        
        # Read the existing SCAD file
        with open(scad_file, 'r') as f:
            scad_code = f.read()
        
        # Determine the module name from the code
        module_name = None
        for shape_type, module in self.shape_module_map.items():
            if module in scad_code:
                module_name = module
                break
        
        if not module_name:
            raise ValueError("Could not determine module name from existing SCAD file")
        
        # Map parameters to OpenSCAD parameter names
        scad_params = self._map_parameters(parameters)
        
        # Generate the updated OpenSCAD code
        updated_code = self._generate_scad_code(module_name, scad_params)
        
        # Write the updated code to the file
        with open(scad_file, 'w') as f:
            f.write(updated_code)
        
        logger.info(f"Updated OpenSCAD code: {scad_file}")
        return scad_file
    
    def combine_models(self, operations: List[Dict[str, Any]]) -> str:
        """
        Combine multiple models using CSG operations.
        
        Args:
            operations: List of operations, each containing:
                - model_type: Type of model
                - parameters: Parameters for the model
                - operation: CSG operation (union, difference, intersection)
                - transform: Optional transformation to apply
            
        Returns:
            Path to the generated SCAD file
        """
        # Generate a unique ID for the combined model
        model_id = str(uuid.uuid4())
        scad_file = os.path.join(self.output_dir, f"{model_id}.scad")
        
        # Include the basic shapes library
        scad_code = f"""// Combined model
include <{os.path.join(self.scad_templates_dir, "basic_shapes.scad")}>;

"""
        
        # Process each operation
        current_op = None
        for i, op in enumerate(operations):
            model_type = op.get('model_type')
            parameters = op.get('parameters', {})
            operation = op.get('operation')
            transform = op.get('transform')
            
            # Get the module name for the model type
            if model_type is None:
                raise ValueError("Model type cannot be None")
            module_name = self.shape_module_map.get(str(model_type))
            if not module_name:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Map parameters to OpenSCAD parameter names
            scad_params = self._map_parameters(parameters)
            
            # Format parameters for the module call
            params_str = ", ".join([f"{k}={v}" for k, v in scad_params.items()])
            
            # Start or continue the CSG operation chain
            if i == 0:
                # First operation doesn't need an operator
                if operation:
                    current_op = operation
                    scad_code += f"{operation}() {{\n"
                
                # Add the module call with optional transformation
                if transform:
                    scad_code += f"    {transform} {module_name}({params_str});\n"
                else:
                    scad_code += f"    {module_name}({params_str});\n"
            else:
                # Check if we need to close the previous operation and start a new one
                if operation and operation != current_op:
                    if current_op:
                        scad_code += "}\n\n"
                    current_op = operation
                    scad_code += f"{operation}() {{\n"
                
                # Add the module call with optional transformation
                if transform:
                    scad_code += f"    {transform} {module_name}({params_str});\n"
                else:
                    scad_code += f"    {module_name}({params_str});\n"
        
        # Close the final operation if needed
        if current_op:
            scad_code += "}\n"
        
        # Write the code to a file
        with open(scad_file, 'w') as f:
            f.write(scad_code)
        
        logger.info(f"Generated combined OpenSCAD code: {scad_file}")
        return scad_file
    
    def _map_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map natural language parameters to OpenSCAD parameters."""
        scad_params = {}
        
        for param, value in parameters.items():
            # Map the parameter name if it exists in the mapping
            scad_param = self.parameter_map.get(param, param)
            
            # Format the value appropriately for OpenSCAD
            if isinstance(value, bool):
                scad_params[scad_param] = str(value).lower()
            elif isinstance(value, str):
                if value.lower() == 'true' or value.lower() == 'false':
                    scad_params[scad_param] = value.lower()
                else:
                    # For text parameters, add quotes
                    scad_params[scad_param] = f'"{value}"'
            else:
                scad_params[scad_param] = value
        
        return scad_params
    
    def _generate_scad_code(self, module_name: str, parameters: Dict[str, Any]) -> str:
        """Generate OpenSCAD code for a module with parameters."""
        # Include the basic shapes library
        scad_code = f"""// Generated OpenSCAD code
include <{os.path.join(self.scad_templates_dir, "basic_shapes.scad")}>;

// Parameters
"""
        
        # Add parameter declarations
        for param, value in parameters.items():
            scad_code += f"{param} = {value};\n"
        
        # Add the module call
        scad_code += f"\n// Model\n{module_name}("
        
        # Add parameters to the module call
        param_list = [f"{param}={param}" for param in parameters.keys()]
        scad_code += ", ".join(param_list)
        
        scad_code += ");\n"
        
        return scad_code
        
    def _generate_ai_driven_code(self, description: str, parameters: Dict[str, Any]) -> str:
        """
        Generate OpenSCAD code using AI-driven techniques based on natural language description.
        
        Args:
            description: Natural language description of the model
            parameters: Dictionary of parameters for the model
            
        Returns:
            Generated OpenSCAD code
        """
        if not self.ai_service:
            logger.warning("AI service not available, falling back to basic shape generation")
            # Fall back to a basic cube if AI service is not available
            return self._generate_scad_code('parametric_cube', {'width': 10, 'height': 10, 'depth': 10})
        
        try:
            # Use the AI service to generate OpenSCAD code
            logger.info(f"Generating OpenSCAD code from description: {description}")
            
            # Prepare context for the AI service
            context = {
                "description": description,
                "parameters": parameters,
                "templates_dir": self.scad_templates_dir
            }
            
            # Call the AI service to generate code
            scad_code = self.ai_service.generate_openscad_code(context)
            
            # Ensure the code includes the basic shapes library
            if "include <" not in scad_code:
                scad_code = f"""// AI-generated OpenSCAD code
include <{os.path.join(self.scad_templates_dir, "basic_shapes.scad")}>;

{scad_code}
"""
            
            return scad_code
        except Exception as e:
            logger.error(f"Error generating AI-driven code: {e}")
            # Fall back to a basic shape if there's an error
            return self._generate_scad_code('parametric_cube', {'width': 10, 'height': 10, 'depth': 10})
