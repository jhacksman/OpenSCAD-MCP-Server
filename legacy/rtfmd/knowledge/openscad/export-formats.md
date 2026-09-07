# OpenSCAD Export Formats

<metadata>
  author: devin-ai-integration
  timestamp: 2025-03-21T12:00:00Z
  version: 1.0.0
  tags: [openscad, export-formats, 3d-printing, prusa, bambu]
</metadata>

## Overview

OpenSCAD supports exporting 3D models in various formats, each with different capabilities for preserving parametric properties and metadata. This document focuses on formats suitable for Prusa and Bambu printers, with an emphasis on alternatives to STL.

## Recommended Formats

### 3MF (3D Manufacturing Format)

3MF is a modern replacement for STL that addresses many of its limitations:

- **Metadata Support**: Includes model information, materials, colors
- **Compact Size**: More efficient encoding than STL
- **Multiple Objects**: Can contain multiple parts in a single file
- **Printer Compatibility**: Widely supported by Prusa and Bambu printers
- **Implementation**: ZIP archive containing XML files

```openscad
// Export to 3MF from command line
// openscad -o model.3mf model.scad
```

### AMF (Additive Manufacturing File Format)

AMF is another modern format that supports:

- **Material Information**: Material properties and colors
- **Curved Surfaces**: Better representation than STL's triangles
- **Metadata**: Design information and parameters
- **Implementation**: XML-based format

```openscad
// Export to AMF from command line
// openscad -o model.amf model.scad
```

### CSG (Constructive Solid Geometry)

CSG is OpenSCAD's native format:

- **Fully Parametric**: Preserves all construction operations
- **Editable**: Can be reopened and modified in OpenSCAD
- **Implementation**: OpenSCAD's internal representation

```openscad
// Export to CSG from command line
// openscad -o model.csg model.scad
```

### SCAD (OpenSCAD Source Code)

The original SCAD file preserves all parametric properties:

- **Complete Parameterization**: All variables and relationships
- **Code Structure**: Modules, functions, and comments
- **Implementation**: Text file with OpenSCAD code

## Printer Compatibility

### Prusa Printers

Prusa printers work well with:

- **3MF**: Full support in PrusaSlicer
- **AMF**: Good support for materials and colors
- **STL**: Supported but with limitations

### Bambu Printers

Bambu printers work best with:

- **3MF**: Preferred format for Bambu Lab software
- **AMF**: Well supported
- **STL**: Basic support

## Implementation Notes

When implementing export functionality:

1. Use OpenSCAD's command-line interface for reliable exports
2. Add metadata to 3MF and AMF files for better organization
3. Test exported files with actual printer software
4. Validate files before sending to printers
