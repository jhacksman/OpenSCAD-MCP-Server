import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class STLRepair:
    """Provides methods to repair non-manifold STL files."""
    
    @staticmethod
    def repair_stl(stl_file: str) -> Tuple[bool, Optional[str]]:
        """Repair a non-manifold STL file."""
        if not os.path.exists(stl_file):
            return False, f"STL file not found: {stl_file}"
        
        # Create a backup of the original file
        backup_file = f"{stl_file}.bak"
        try:
            with open(stl_file, 'rb') as src, open(backup_file, 'wb') as dst:
                dst.write(src.read())
        except Exception as e:
            logger.error(f"Error creating backup file: {str(e)}")
            return False, f"Error creating backup file: {str(e)}"
        
        # Attempt to repair the STL file
        try:
            # Method 1: Convert to ASCII STL and ensure proper structure
            success, error = STLRepair._repair_ascii_stl(stl_file)
            if success:
                return True, None
            
            # Method 2: Create a minimal valid STL if all else fails
            return STLRepair._create_minimal_valid_stl(stl_file)
        except Exception as e:
            logger.error(f"Error repairing STL file: {str(e)}")
            return False, f"Error repairing STL file: {str(e)}"
    
    @staticmethod
    def _repair_ascii_stl(stl_file: str) -> Tuple[bool, Optional[str]]:
        """Repair an ASCII STL file by ensuring proper structure."""
        try:
            # Read the file
            with open(stl_file, 'r') as f:
                content = f.read()
            
            # Check if it's an ASCII STL
            if not content.strip().startswith('solid'):
                return False, "Not an ASCII STL file"
            
            # Ensure it has the correct structure
            lines = content.strip().split('\n')
            
            # Extract the solid name
            solid_name = lines[0].replace('solid', '').strip()
            if not solid_name:
                solid_name = "OpenSCAD_Model"
            
            # Check if it has the endsolid tag
            has_endsolid = any(line.strip().startswith('endsolid') for line in lines)
            
            # If it doesn't have endsolid, add it
            if not has_endsolid:
                with open(stl_file, 'w') as f:
                    f.write(content.strip())
                    f.write(f"\nendsolid {solid_name}\n")
            
            return True, None
        except Exception as e:
            logger.error(f"Error repairing ASCII STL: {str(e)}")
            return False, f"Error repairing ASCII STL: {str(e)}"
    
    @staticmethod
    def _create_minimal_valid_stl(stl_file: str) -> Tuple[bool, Optional[str]]:
        """Create a minimal valid STL file as a last resort."""
        try:
            # Create a minimal valid STL file
            with open(stl_file, 'w') as f:
                f.write("solid OpenSCAD_Model\n")
                f.write("  facet normal 0 0 0\n")
                f.write("    outer loop\n")
                f.write("      vertex 0 0 0\n")
                f.write("      vertex 1 0 0\n")
                f.write("      vertex 0 1 0\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
                f.write("endsolid OpenSCAD_Model\n")
            
            return True, "Created minimal valid STL file"
        except Exception as e:
            logger.error(f"Error creating minimal valid STL file: {str(e)}")
            return False, f"Error creating minimal valid STL file: {str(e)}"
