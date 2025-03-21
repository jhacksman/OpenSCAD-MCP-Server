# OpenSCAD Basics

<metadata>
  author: devin-ai-integration
  timestamp: 2025-03-21T01:30:00Z
  version: 1.0.0
  tags: [openscad, 3d-modeling, csg, parametric-design]
</metadata>

## Overview

OpenSCAD is a programmer's solid 3D CAD modeler that uses a scripting language to define 3D objects. Unlike traditional CAD software that focuses on interactive modeling, OpenSCAD emphasizes programmatic and parametric design.

## Key Concepts

### Constructive Solid Geometry (CSG)

OpenSCAD uses CSG operations to create complex models by combining simpler primitives:

- **Union**: Combines multiple objects (`union() { ... }`)
- **Difference**: Subtracts one object from another (`difference() { ... }`)
- **Intersection**: Creates an object from the overlapping portions of other objects (`intersection() { ... }`)

### Primitive Shapes

OpenSCAD provides several built-in primitive shapes:

- **Cube**: `cube([width, depth, height], center=true/false)`
- **Sphere**: `sphere(r=radius, $fn=segments)`
- **Cylinder**: `cylinder(h=height, r=radius, center=true/false, $fn=segments)`
- **Polyhedron**: For complex shapes with defined faces

### Transformations

Objects can be transformed using:

- **Translate**: `translate([x, y, z]) { ... }`
- **Rotate**: `rotate([x_deg, y_deg, z_deg]) { ... }`
- **Scale**: `scale([x, y, z]) { ... }`
- **Mirror**: `mirror([x, y, z]) { ... }`

### Parametric Design

OpenSCAD excels at parametric design:

- Variables can define dimensions and relationships
- Modules can create reusable components with parameters
- Mathematical expressions can define complex relationships

## Command Line Usage

OpenSCAD can be run headless using command-line options:

- Generate STL: `openscad -o output.stl input.scad`
- Pass parameters: `openscad -D "width=10" -D "height=20" -o output.stl input.scad`
- Generate PNG preview: `openscad --camera=0,0,0,0,0,0,50 --imgsize=800,600 -o preview.png input.scad`

## Best Practices

- Use modules for reusable components
- Parameterize designs for flexibility
- Use descriptive variable names
- Comment code for clarity
- Organize complex designs hierarchically
- Use $fn judiciously for performance
- Ensure models are manifold (watertight) for 3D printing
