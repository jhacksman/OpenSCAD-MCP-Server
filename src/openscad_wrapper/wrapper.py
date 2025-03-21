import os
import subprocess
import uuid
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class OpenSCADWrapper:
    """
    Wrapper for OpenSCAD command-line interface.
    Provides methods to generate SCAD code, STL files, and preview images.
    """
    
    def __init__(self, scad_dir: str, output_dir: str):
        """
        Initialize the OpenSCAD wrapper.
        
        Args:
            scad_dir: Directory to store SCAD files
            output_dir: Directory to store output files (STL, PNG)
        """
        self.scad_dir = scad_dir
        self.output_dir = output_dir
        self.stl_dir = os.path.join(output_dir, "stl")
        self.preview_dir = os.path.join(output_dir, "preview")
        
        # Create directories if they don't exist
        os.makedirs(self.scad_dir, exist_ok=True)
        os.makedirs(self.stl_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        
        # Basic shape templates
        self.shape_templates = {
            "cube": self._cube_template,
            "sphere": self._sphere_template,
            "cylinder": self._cylinder_template,
            "box": self._box_template,
            "rounded_box": self._rounded_box_template,
        }
    
    def generate_scad_code(self, model_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate OpenSCAD code for a given model type and parameters.
        
        Args:
            model_type: Type of model to generate (cube, sphere, cylinder, etc.)
            parameters: Dictionary of parameters for the model
            
        Returns:
            Path to the generated SCAD file
        """
        model_id = str(uuid.uuid4())
        scad_file = os.path.join(self.scad_dir, f"{model_id}.scad")
        
        # Get the template function for the model type
        template_func = self.shape_templates.get(model_type)
        if not template_func:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Generate SCAD code using the template
        scad_code = template_func(parameters)
        
        # Write SCAD code to file
        with open(scad_file, 'w') as f:
            f.write(scad_code)
        
        logger.info(f"Generated SCAD file: {scad_file}")
        return scad_file
    
    def update_scad_code(self, model_id: str, parameters: Dict[str, Any]) -> str:
        """
        Update an existing SCAD file with new parameters.
        
        Args:
            model_id: ID of the model to update
            parameters: New parameters for the model
            
        Returns:
            Path to the updated SCAD file
        """
        scad_file = os.path.join(self.scad_dir, f"{model_id}.scad")
        if not os.path.exists(scad_file):
            raise FileNotFoundError(f"SCAD file not found: {scad_file}")
        
        # Read the existing SCAD file to determine its type
        with open(scad_file, 'r') as f:
            scad_code = f.read()
        
        # Determine model type from the code (simplified approach)
        model_type = None
        for shape_type in self.shape_templates:
            if shape_type in scad_code.lower():
                model_type = shape_type
                break
        
        if not model_type:
            raise ValueError("Could not determine model type from existing SCAD file")
        
        # Generate new SCAD code
        new_scad_code = self.shape_templates[model_type](parameters)
        
        # Write updated SCAD code to file
        with open(scad_file, 'w') as f:
            f.write(new_scad_code)
        
        logger.info(f"Updated SCAD file: {scad_file}")
        return scad_file
    
    def generate_stl(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an STL file from a SCAD file.
        
        Args:
            scad_file: Path to the SCAD file
            parameters: Optional parameters to override in the SCAD file
            
        Returns:
            Path to the generated STL file
        """
        model_id = os.path.basename(scad_file).split('.')[0]
        stl_file = os.path.join(self.stl_dir, f"{model_id}.stl")
        
        # Build command
        cmd = ["openscad", "-o", stl_file]
        
        # Add parameters if provided
        if parameters:
            for key, value in parameters.items():
                cmd.extend(["-D", f"{key}={value}"])
        
        # Add input file
        cmd.append(scad_file)
        
        # Run OpenSCAD
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Generated STL file: {stl_file}")
            logger.debug(result.stdout)
            return stl_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating STL file: {e.stderr}")
            raise RuntimeError(f"Failed to generate STL file: {e.stderr}")
    
    def generate_preview(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None,
                        camera_position: str = "0,0,0,0,0,0,50", 
                        image_size: str = "800,600") -> str:
        """
        Generate a preview image from a SCAD file.
        
        Args:
            scad_file: Path to the SCAD file
            parameters: Optional parameters to override in the SCAD file
            camera_position: Camera position in format "tx,ty,tz,rx,ry,rz,dist"
            image_size: Image size in format "width,height"
            
        Returns:
            Path to the generated preview image
        """
        model_id = os.path.basename(scad_file).split('.')[0]
        preview_file = os.path.join(self.preview_dir, f"{model_id}.png")
        
        # Build command
        cmd = ["openscad", "--camera", camera_position, "--imgsize", image_size, "-o", preview_file]
        
        # Add parameters if provided
        if parameters:
            for key, value in parameters.items():
                cmd.extend(["-D", f"{key}={value}"])
        
        # Add input file
        cmd.append(scad_file)
        
        # Run OpenSCAD
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Generated preview image: {preview_file}")
            logger.debug(result.stdout)
            return preview_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating preview image: {e.stderr}")
            # Since we know there might be issues with headless rendering, we'll create a placeholder
            logger.warning("Using placeholder image due to rendering error")
            return self._create_placeholder_image(preview_file)
    
    def _create_placeholder_image(self, output_path: str) -> str:
        """Create a simple placeholder image when rendering fails."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a blank image
            img = Image.new('RGB', (800, 600), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Add text
            draw.text((400, 300), "Preview not available", fill=(0, 0, 0))
            
            # Save the image
            img.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error creating placeholder image: {str(e)}")
            # If all else fails, return the path anyway
            return output_path
    
    # Template functions for basic shapes
    
    def _cube_template(self, params: Dict[str, Any]) -> str:
        """Generate SCAD code for a cube."""
        size_x = params.get('width', 10)
        size_y = params.get('depth', 10)
        size_z = params.get('height', 10)
        center = params.get('center', 'false').lower() == 'true'
        
        return f"""// Cube
// Parameters:
//   width = {size_x}
//   depth = {size_y}
//   height = {size_z}
//   center = {str(center).lower()}

width = {size_x};
depth = {size_y};
height = {size_z};
center = {str(center).lower()};

cube([width, depth, height], center=center);
"""
    
    def _sphere_template(self, params: Dict[str, Any]) -> str:
        """Generate SCAD code for a sphere."""
        radius = params.get('radius', 10)
        segments = params.get('segments', 32)
        
        return f"""// Sphere
// Parameters:
//   radius = {radius}
//   segments = {segments}

radius = {radius};
$fn = {segments};

sphere(r=radius);
"""
    
    def _cylinder_template(self, params: Dict[str, Any]) -> str:
        """Generate SCAD code for a cylinder."""
        radius = params.get('radius', 10)
        height = params.get('height', 20)
        center = params.get('center', 'false').lower() == 'true'
        segments = params.get('segments', 32)
        
        return f"""// Cylinder
// Parameters:
//   radius = {radius}
//   height = {height}
//   center = {str(center).lower()}
//   segments = {segments}

radius = {radius};
height = {height};
center = {str(center).lower()};
$fn = {segments};

cylinder(h=height, r=radius, center=center);
"""
    
    def _box_template(self, params: Dict[str, Any]) -> str:
        """Generate SCAD code for a hollow box."""
        width = params.get('width', 30)
        depth = params.get('depth', 20)
        height = params.get('height', 15)
        thickness = params.get('thickness', 2)
        
        return f"""// Hollow Box
// Parameters:
//   width = {width}
//   depth = {depth}
//   height = {height}
//   thickness = {thickness}

width = {width};
depth = {depth};
height = {height};
thickness = {thickness};

module box(width, depth, height, thickness) {{
    difference() {{
        cube([width, depth, height]);
        translate([thickness, thickness, thickness])
        cube([width - 2*thickness, depth - 2*thickness, height - thickness]);
    }}
}}

box(width, depth, height, thickness);
"""
    
    def _rounded_box_template(self, params: Dict[str, Any]) -> str:
        """Generate SCAD code for a rounded box."""
        width = params.get('width', 30)
        depth = params.get('depth', 20)
        height = params.get('height', 15)
        radius = params.get('radius', 3)
        segments = params.get('segments', 32)
        
        return f"""// Rounded Box
// Parameters:
//   width = {width}
//   depth = {depth}
//   height = {height}
//   radius = {radius}
//   segments = {segments}

width = {width};
depth = {depth};
height = {height};
radius = {radius};
$fn = {segments};

module rounded_box(width, depth, height, radius) {{
    hull() {{
        translate([radius, radius, radius])
        sphere(r=radius);
        
        translate([width-radius, radius, radius])
        sphere(r=radius);
        
        translate([radius, depth-radius, radius])
        sphere(r=radius);
        
        translate([width-radius, depth-radius, radius])
        sphere(r=radius);
        
        translate([radius, radius, height-radius])
        sphere(r=radius);
        
        translate([width-radius, radius, height-radius])
        sphere(r=radius);
        
        translate([radius, depth-radius, height-radius])
        sphere(r=radius);
        
        translate([width-radius, depth-radius, height-radius])
        sphere(r=radius);
    }}
}}

rounded_box(width, depth, height, radius);
"""
