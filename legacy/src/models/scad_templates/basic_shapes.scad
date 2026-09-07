// Basic Shapes Library for OpenSCAD MCP Server
// Contains reusable modules for common geometric patterns

// Basic cube with optional center parameter
module parametric_cube(width=10, depth=10, height=10, center=false) {
    cube([width, depth, height], center=center);
}

// Basic sphere with customizable segments
module parametric_sphere(radius=10, segments=32) {
    $fn = segments;
    sphere(r=radius);
}

// Basic cylinder with customizable segments
module parametric_cylinder(radius=10, height=20, center=false, segments=32) {
    $fn = segments;
    cylinder(h=height, r=radius, center=center);
}

// Hollow box with customizable wall thickness
module hollow_box(width=30, depth=20, height=15, thickness=2) {
    difference() {
        cube([width, depth, height]);
        translate([thickness, thickness, thickness])
        cube([width - 2*thickness, depth - 2*thickness, height - thickness]);
    }
}

// Rounded box with customizable corner radius
module rounded_box(width=30, depth=20, height=15, radius=3, segments=32) {
    $fn = segments;
    hull() {
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
    }
}

// Rounded hollow box (container with rounded corners)
module rounded_container(width=30, depth=20, height=15, radius=3, thickness=2, segments=32) {
    $fn = segments;
    difference() {
        rounded_box(width, depth, height, radius, segments);
        translate([thickness, thickness, thickness])
        rounded_box(
            width - 2*thickness, 
            depth - 2*thickness, 
            height - thickness + 0.01, // Slight overlap to ensure clean difference
            radius - thickness > 0 ? radius - thickness : 0.1,
            segments
        );
    }
}

// Tube (hollow cylinder)
module tube(outer_radius=10, inner_radius=8, height=20, center=false, segments=32) {
    $fn = segments;
    difference() {
        cylinder(h=height, r=outer_radius, center=center);
        cylinder(h=height+0.01, r=inner_radius, center=center);
    }
}

// Cone
module cone(bottom_radius=10, top_radius=0, height=20, center=false, segments=32) {
    $fn = segments;
    cylinder(h=height, r1=bottom_radius, r2=top_radius, center=center);
}

// Wedge (triangular prism)
module wedge(width=20, depth=20, height=10) {
    polyhedron(
        points=[
            [0,0,0], [width,0,0], [width,depth,0], [0,depth,0],
            [0,0,height], [width,0,0], [0,depth,0]
        ],
        faces=[
            [0,1,2,3], // bottom
            [4,5,1,0], // front
            [4,0,3,6], // left
            [6,3,2,5], // back
            [4,6,5], // top
        ]
    );
}

// Rounded cylinder (cylinder with rounded top and bottom)
module rounded_cylinder(radius=10, height=20, corner_radius=2, center=false, segments=32) {
    $fn = segments;
    hull() {
        translate([0, 0, corner_radius])
        cylinder(h=height - 2*corner_radius, r=radius, center=center);
        
        translate([0, 0, corner_radius])
        rotate_extrude()
        translate([radius - corner_radius, 0, 0])
        circle(r=corner_radius);
        
        translate([0, 0, height - corner_radius])
        rotate_extrude()
        translate([radius - corner_radius, 0, 0])
        circle(r=corner_radius);
    }
}

// Torus (donut shape)
module torus(outer_radius=20, inner_radius=5, segments=32) {
    $fn = segments;
    rotate_extrude()
    translate([outer_radius - inner_radius, 0, 0])
    circle(r=inner_radius);
}

// Hexagonal prism
module hexagonal_prism(radius=10, height=20, center=false) {
    cylinder(h=height, r=radius, $fn=6, center=center);
}

// Text with customizable parameters
module text_3d(text="OpenSCAD", size=10, height=3, font="Liberation Sans", halign="center", valign="center") {
    linear_extrude(height=height)
    text(text=text, size=size, font=font, halign=halign, valign=valign);
}
