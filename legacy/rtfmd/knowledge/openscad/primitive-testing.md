# Testing OpenSCAD Primitives

<metadata>
  author: devin-ai-integration
  timestamp: 2025-03-21T12:00:00Z
  version: 1.0.0
  tags: [openscad, primitives, testing, 3d-modeling]
</metadata>

## Overview

Testing OpenSCAD primitives is essential to ensure that the MCP server can reliably generate and export 3D models. This document outlines approaches for programmatically testing primitives and validating their exports.

## Primitive Types

The OpenSCAD MCP Server supports these primitive types:

- **Basic Shapes**: cube, sphere, cylinder
- **Complex Shapes**: cone, torus, hexagonal_prism
- **Containers**: hollow_box, rounded_box, tube
- **Text**: 3D text with customizable parameters

## Testing Approach

### Parameter Testing

Each primitive should be tested with:

- **Default Parameters**: Ensure the primitive renders correctly with default values
- **Boundary Values**: Test minimum and maximum reasonable values
- **Special Cases**: Test cases like zero dimensions, negative values

### Export Testing

For each primitive and parameter set:

- **Export Formats**: Test export to 3MF, AMF, CSG, and SCAD formats
- **Format Validation**: Ensure exported files meet format specifications
- **Metadata Preservation**: Verify that parametric properties are preserved

### Integration Testing

Test the full pipeline:

- **Natural Language → Parameters**: Test parameter extraction
- **Parameters → OpenSCAD Code**: Test code generation
- **OpenSCAD Code → Export**: Test file export
- **Export → Printer**: Test compatibility with printer software

## Validation Criteria

Exported models should meet these criteria:

- **Manifold**: Models should be watertight (no holes in the mesh)
- **Valid Format**: Files should validate against format specifications
- **Metadata**: Should contain relevant model metadata
- **Render Performance**: Models should render efficiently

## Implementation

The `PrimitiveTester` class implements this testing approach:

```python
# Example usage
tester = PrimitiveTester(code_generator, cad_exporter)
results = tester.test_all_primitives()

# Test specific primitives
cube_results = tester.test_primitive("cube")
```

## Printer Compatibility Tests

Before sending to physical printers:

1. Import exports into PrusaSlicer or Bambu Studio
2. Check for import warnings or errors
3. Verify that models slice correctly
4. Test prints with simple examples
