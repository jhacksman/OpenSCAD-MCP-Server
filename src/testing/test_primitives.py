import os
import argparse
import logging
import json
from typing import Dict, Any, List

from src.models.code_generator import OpenSCADCodeGenerator
from src.utils.cad_exporter import CADExporter
from src.utils.format_validator import FormatValidator
from src.testing.primitive_tester import PrimitiveTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Test OpenSCAD primitives with different export formats')
    parser.add_argument('--output-dir', default='test_output', help='Directory to store test output')
    parser.add_argument('--formats', nargs='+', default=['3mf', 'amf', 'csg', 'scad'], 
                       help='Formats to test (default: 3mf amf csg scad)')
    parser.add_argument('--primitives', nargs='+', 
                       help='Primitives to test (default: all)')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate exported files')
    parser.add_argument('--printer-type', choices=['prusa', 'bambu'], default='prusa',
                       help='Printer type to check compatibility with (default: prusa)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("scad", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    # Use absolute path for templates to avoid path issues
    templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/scad_templates"))
    code_generator = OpenSCADCodeGenerator(templates_dir, "scad")
    cad_exporter = CADExporter()
    
    # Initialize tester
    tester = PrimitiveTester(code_generator, cad_exporter, args.output_dir)
    
    # Override formats if specified
    if args.formats:
        tester.formats = args.formats
    
    # Test primitives
    if args.primitives:
        results = {}
        for primitive in args.primitives:
            results[primitive] = tester.test_primitive(primitive)
    else:
        results = tester.test_all_primitives()
    
    # Print results
    logger.info(f"Test results: {json.dumps(results, indent=2)}")
    
    # Validate exported files if requested
    if args.validate:
        validator = FormatValidator()
        validation_results = {}
        
        for primitive, primitive_results in results.items():
            validation_results[primitive] = {}
            
            for format_type, format_results in primitive_results["formats"].items():
                if format_results["success"] and format_type in ['3mf', 'amf']:
                    output_file = format_results["output_file"]
                    
                    if format_type == '3mf':
                        is_valid, error = validator.validate_3mf(output_file)
                    elif format_type == 'amf':
                        is_valid, error = validator.validate_amf(output_file)
                    else:
                        is_valid, error = False, "Validation not supported for this format"
                    
                    # Check printer compatibility
                    is_compatible, compat_error = validator.check_printer_compatibility(
                        output_file, args.printer_type
                    )
                    
                    metadata = validator.extract_metadata(output_file)
                    
                    validation_results[primitive][format_type] = {
                        "is_valid": is_valid,
                        "error": error,
                        "is_compatible_with_printer": is_compatible,
                        "compatibility_error": compat_error,
                        "metadata": metadata
                    }
        
        logger.info(f"Validation results: {json.dumps(validation_results, indent=2)}")

if __name__ == "__main__":
    main()
