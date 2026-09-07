# Decision: Export Format Selection

<metadata>
  author: devin-ai-integration
  timestamp: 2025-03-21T12:00:00Z
  version: 1.0.0
  tags: [export-formats, 3d-printing, decision, prusa, bambu]
</metadata>

## Context

The OpenSCAD MCP Server needs to export 3D models in formats that:
1. Preserve parametric properties
2. Support metadata
3. Are compatible with Prusa and Bambu printers
4. Avoid limitations of STL format

<exploration>
We evaluated multiple export formats:
- STL: Traditional format but lacks metadata
- CSG: OpenSCAD's native format, fully parametric
- SCAD: Source code, fully parametric
- 3MF: Modern format with metadata support
- AMF: XML-based format with metadata
- DXF/SVG: 2D formats for laser cutting
</exploration>

## Decision

We will use **3MF as the primary export format** with AMF as a secondary option. 
CSG and SCAD will be supported for users who want to modify the models in OpenSCAD.

<mental-model>
The ideal export format should:
- Maintain all design parameters
- Include metadata about the model
- Be widely supported by popular slicers
- Have a clean, standardized specification
- Support multiple objects and materials
</mental-model>

## Rationale

<pattern-recognition>
Modern 3D printing workflows favor formats that preserve more information than just geometry. The industry is shifting from STL to more capable formats like 3MF.
</pattern-recognition>

- **3MF** is supported by both Prusa and Bambu printer software
- **3MF** includes support for metadata, colors, and materials
- **3MF** has a cleaner specification than STL
- **AMF** offers similar advantages but with less widespread adoption
- **CSG/SCAD** formats maintain full parametric properties but only within OpenSCAD

<trade-off>
We considered making STL an option for broader compatibility, but this would compromise our goal of preserving parametric properties. The benefits of 3MF outweigh the minor compatibility issues that might arise.
</trade-off>

## Consequences

**Positive:**
- Better preservation of model information
- Improved compatibility with modern printer software
- Future-proof approach as 3MF adoption increases

**Negative:**
- Slightly more complex implementation than STL
- May require validation to ensure proper format compliance

<technical-debt>
We will need to implement validation for 3MF and AMF files to ensure they meet specifications. This adds complexity but is necessary for reliability.
</technical-debt>

<knowledge-refs>
- [OpenSCAD Export Formats](/rtfmd/knowledge/openscad/export-formats.md)
- [OpenSCAD Basics](/rtfmd/knowledge/openscad/openscad-basics.md)
</knowledge-refs>
