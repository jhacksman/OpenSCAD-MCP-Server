import os
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AIService:
    """
    Service for AI-driven OpenSCAD code generation.
    Translates natural language descriptions into OpenSCAD code.
    """
    
    def __init__(self, templates_dir: str, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI service.
        
        Args:
            templates_dir: Directory containing OpenSCAD templates
            model_config: Optional configuration for the AI model
        """
        self.templates_dir = templates_dir
        self.model_config = model_config or {}
        
        # Load templates
        self.templates = self._load_templates()
        
        logger.info(f"Initialized AI service with {len(self.templates)} templates")
    
    def generate_openscad_code(self, context: Dict[str, Any]) -> str:
        """
        Generate OpenSCAD code from natural language description.
        
        Args:
            context: Dictionary containing:
                - description: Natural language description
                - parameters: Dictionary of parameters
                - templates_dir: Directory containing templates
                
        Returns:
            Generated OpenSCAD code
        """
        description = context.get("description", "")
        parameters = context.get("parameters", {})
        
        logger.info(f"Generating OpenSCAD code for: {description}")
        
        # Parse the description to identify key components
        components = self._parse_description(description)
        
        # Generate code based on identified components
        code = self._generate_code_from_components(components, parameters)
        
        return code
    
    def _load_templates(self) -> Dict[str, str]:
        """Load OpenSCAD code templates from the templates directory."""
        templates = {}
        
        # Check if templates directory exists
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates
        
        # Load all .scad files in the templates directory
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".scad"):
                template_name = os.path.splitext(filename)[0]
                template_path = os.path.join(self.templates_dir, filename)
                
                try:
                    with open(template_path, 'r') as f:
                        templates[template_name] = f.read()
                except Exception as e:
                    logger.error(f"Error loading template {template_path}: {e}")
        
        return templates
    
    def _parse_description(self, description: str) -> Dict[str, Any]:
        """
        Parse a natural language description to identify key components.
        
        Args:
            description: Natural language description of the model
            
        Returns:
            Dictionary of identified components
        """
        components = {
            "primary_shape": None,
            "operations": [],
            "features": [],
            "modifiers": []
        }
        
        # Identify primary shape
        shape_patterns = {
            "cube": r'\b(cube|box|rectangular|block)\b',
            "sphere": r'\b(sphere|ball|round|circular)\b',
            "cylinder": r'\b(cylinder|tube|pipe|rod)\b',
            "cone": r'\b(cone|pyramid|tapered)\b',
            "torus": r'\b(torus|donut|ring)\b'
        }
        
        for shape, pattern in shape_patterns.items():
            if re.search(pattern, description, re.IGNORECASE):
                components["primary_shape"] = shape
                break
        
        # Identify operations
        operation_patterns = {
            "union": r'\b(combine|join|merge|add)\b',
            "difference": r'\b(subtract|remove|cut|hole|hollow)\b',
            "intersection": r'\b(intersect|common|shared)\b'
        }
        
        for operation, pattern in operation_patterns.items():
            if re.search(pattern, description, re.IGNORECASE):
                components["operations"].append(operation)
        
        # Identify features
        feature_patterns = {
            "rounded_corners": r'\b(rounded corners|fillets|chamfer)\b',
            "holes": r'\b(holes|perforations|openings)\b',
            "text": r'\b(text|label|inscription)\b',
            "pattern": r'\b(pattern|array|grid|repeat)\b'
        }
        
        for feature, pattern in feature_patterns.items():
            if re.search(pattern, description, re.IGNORECASE):
                components["features"].append(feature)
        
        # Identify modifiers
        modifier_patterns = {
            "scale": r'\b(scale|resize|proportion)\b',
            "rotate": r'\b(rotate|turn|spin|angle)\b',
            "translate": r'\b(move|shift|position|place)\b',
            "mirror": r'\b(mirror|reflect|flip)\b'
        }
        
        for modifier, pattern in modifier_patterns.items():
            if re.search(pattern, description, re.IGNORECASE):
                components["modifiers"].append(modifier)
        
        logger.info(f"Parsed components: {components}")
        return components
    
    def _generate_code_from_components(self, components: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """
        Generate OpenSCAD code based on identified components.
        
        Args:
            components: Dictionary of identified components
            parameters: Dictionary of parameters
            
        Returns:
            Generated OpenSCAD code
        """
        code = []
        
        # Add header
        code.append("// AI-generated OpenSCAD code")
        code.append("// Generated from natural language description")
        code.append("")
        
        # Add parameter declarations
        code.append("// Parameters")
        for param, value in parameters.items():
            if isinstance(value, str) and not (value.lower() == 'true' or value.lower() == 'false'):
                code.append(f'{param} = "{value}";')
            else:
                code.append(f"{param} = {value};")
        code.append("")
        
        # Generate code for primary shape
        primary_shape = components.get("primary_shape")
        if not primary_shape:
            primary_shape = "cube"  # Default to cube if no shape is identified
        
        # Start with operations if any
        operations = components.get("operations", [])
        if operations:
            for operation in operations:
                code.append(f"{operation}() {{")
            code.append("    // Primary shape")
        
        # Add modifiers if any
        modifiers = components.get("modifiers", [])
        indent = "    " if operations else ""
        
        if modifiers:
            for modifier in modifiers:
                if modifier == "scale":
                    scale_value = parameters.get("scale", 1)
                    code.append(f"{indent}scale([{scale_value}, {scale_value}, {scale_value}])")
                elif modifier == "rotate":
                    angle = parameters.get("angle", 0)
                    code.append(f"{indent}rotate([0, 0, {angle}])")
                elif modifier == "translate":
                    x = parameters.get("x", 0)
                    y = parameters.get("y", 0)
                    z = parameters.get("z", 0)
                    code.append(f"{indent}translate([{x}, {y}, {z}])")
                elif modifier == "mirror":
                    code.append(f"{indent}mirror([0, 0, 1])")
        
        # Add the primary shape
        if primary_shape == "cube":
            width = parameters.get("width", 10)
            depth = parameters.get("depth", 10)
            height = parameters.get("height", 10)
            center = parameters.get("center", "true")
            code.append(f"{indent}cube([{width}, {depth}, {height}], center={center});")
        elif primary_shape == "sphere":
            radius = parameters.get("radius", 10)
            segments = parameters.get("segments", 32)
            code.append(f"{indent}sphere(r={radius}, $fn={segments});")
        elif primary_shape == "cylinder":
            radius = parameters.get("radius", 10)
            height = parameters.get("height", 20)
            center = parameters.get("center", "true")
            segments = parameters.get("segments", 32)
            code.append(f"{indent}cylinder(h={height}, r={radius}, center={center}, $fn={segments});")
        elif primary_shape == "cone":
            base_radius = parameters.get("base_radius", 10)
            height = parameters.get("height", 20)
            center = parameters.get("center", "true")
            segments = parameters.get("segments", 32)
            code.append(f"{indent}cylinder(h={height}, r1={base_radius}, r2=0, center={center}, $fn={segments});")
        elif primary_shape == "torus":
            major_radius = parameters.get("major_radius", 20)
            minor_radius = parameters.get("minor_radius", 5)
            segments = parameters.get("segments", 32)
            code.append(f"{indent}rotate_extrude($fn={segments})")
            code.append(f"{indent}    translate([{major_radius}, 0, 0])")
            code.append(f"{indent}    circle(r={minor_radius}, $fn={segments});")
        
        # Add features if any
        features = components.get("features", [])
        if features and "holes" in features:
            code.append("")
            code.append(f"{indent}// Add holes")
            code.append(f"{indent}difference() {{")
            code.append(f"{indent}    children(0);")  # Reference the primary shape
            
            # Add a sample hole
            hole_radius = parameters.get("hole_radius", 2)
            code.append(f"{indent}    translate([0, 0, 0])")
            code.append(f"{indent}    cylinder(h=100, r={hole_radius}, center=true, $fn=32);")
            
            code.append(f"{indent}}}")
        
        # Close operations if any
        if operations:
            code.append("}")
        
        return "\n".join(code)
