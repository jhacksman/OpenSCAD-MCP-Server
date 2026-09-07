"""Small, explicit parametric templates; no inferred geometry or AI fallback."""

import json
import math
import re

DEFAULTS = {
    "cube": {"width": 10, "depth": 10, "height": 10, "center": False},
    "sphere": {"radius": 10, "segments": 48},
    "cylinder": {"radius": 10, "height": 20, "center": False, "segments": 48},
    "box": {"width": 30, "depth": 20, "height": 15, "thickness": 2},
    "rounded_box": {
        "width": 30,
        "depth": 20,
        "height": 15,
        "radius": 3,
        "segments": 32,
    },
    "tube": {
        "outer_radius": 10,
        "inner_radius": 8,
        "height": 20,
        "center": False,
        "segments": 48,
    },
    "cone": {
        "bottom_radius": 10,
        "top_radius": 0,
        "height": 20,
        "center": False,
        "segments": 48,
    },
    "torus": {"major_radius": 15, "minor_radius": 5, "segments": 48},
    "prism": {"width": 20, "depth": 20, "height": 10},
    "hexagonal_prism": {"radius": 10, "height": 20, "center": False},
    "text": {"text": "OpenSCAD", "size": 10, "height": 3},
}

BODIES = {
    "cube": "cube([width, depth, height], center=center);",
    "sphere": "sphere(r=radius, $fn=segments);",
    "cylinder": "cylinder(h=height, r=radius, center=center, $fn=segments);",
    "box": """difference() {
    cube([width, depth, height]);
    translate([thickness, thickness, thickness])
        cube([width-2*thickness, depth-2*thickness, height]);
}""",
    "rounded_box": """hull() {
    for (x=[radius, width-radius], y=[radius, depth-radius], z=[radius, height-radius])
        translate([x,y,z]) sphere(r=radius, $fn=segments);
}""",
    "tube": """difference() {
    cylinder(h=height, r=outer_radius, center=center, $fn=segments);
    translate([0,0,center ? 0 : -0.01])
        cylinder(h=height+0.02, r=inner_radius, center=center, $fn=segments);
}""",
    "cone": "cylinder(h=height, r1=bottom_radius, r2=top_radius, center=center, $fn=segments);",
    "torus": """rotate_extrude($fn=segments)
    translate([major_radius,0,0]) circle(r=minor_radius, $fn=segments);""",
    "prism": "linear_extrude(height=height) polygon([[0,0],[width,0],[0,depth]]);",
    "hexagonal_prism": "cylinder(h=height, r=radius, center=center, $fn=6);",
    "text": 'linear_extrude(height=height) text(text, size=size, halign="center", valign="center");',
}


def parameters_for(shape: str, values: dict | None = None) -> dict:
    if shape not in DEFAULTS:
        raise ValueError(
            f"Unsupported model type {shape!r}. Supported: {', '.join(DEFAULTS)}"
        )
    values = values or {}
    unknown = values.keys() - DEFAULTS[shape].keys()
    if unknown:
        raise ValueError(
            f"Unsupported parameters for {shape}: {', '.join(sorted(unknown))}"
        )
    p = DEFAULTS[shape] | values
    for key, value in p.items():
        if key == "center":
            if type(value) is not bool:
                raise ValueError("center must be a boolean")
        elif key == "text":
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError("text must contain 1–200 characters")
        elif key == "segments":
            if type(value) is not int or not 12 <= value <= 256:
                raise ValueError("segments must be an integer between 12 and 256")
        elif (
            type(value) not in (int, float)
            or value < 0
            or value > 10000
            or not math.isfinite(value)
            or (value == 0 and key != "top_radius")
        ):
            raise ValueError(
                f"{key} must be a finite positive number, at most 10000 mm"
            )
    if shape == "box" and (
        2 * p["thickness"] >= min(p["width"], p["depth"])
        or p["thickness"] >= p["height"]
    ):
        raise ValueError("thickness leaves no interior in the box")
    if shape == "rounded_box" and 2 * p["radius"] > min(
        p["width"], p["depth"], p["height"]
    ):
        raise ValueError("radius exceeds half the smallest box dimension")
    if shape == "tube" and p["inner_radius"] >= p["outer_radius"]:
        raise ValueError("inner_radius must be smaller than outer_radius")
    if shape == "torus" and p["minor_radius"] >= p["major_radius"]:
        raise ValueError("minor_radius must be smaller than major_radius")
    return p


def generate_source(shape: str, values: dict | None = None) -> tuple[str, dict]:
    p = parameters_for(shape, values)
    source = "// Dimensions in millimeters. Edit the parameters below.\n"
    source += "\n".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)};" for key, value in p.items()
    )
    return source + "\n\n" + BODIES[shape] + "\n", p


# Restricted description syntax is intentionally separate from arbitrary SCAD.
SHAPES = [
    ("rounded_box", r"rounded[ _]box"),
    ("box", r"hollow box|container|tray|box"),
    ("hexagonal_prism", r"hexagonal[ _]prism"),
    ("tube", r"tube|pipe"),
    ("cube", r"cube|cuboid|block"),
    ("sphere", r"sphere|ball"),
    ("cylinder", r"cylinder|rod"),
    ("cone", r"cone"),
    ("torus", r"torus|donut"),
    ("prism", r"triangular prism|prism|wedge"),
]
UNITS = {
    None: 1,
    "mm": 1,
    "cm": 10,
    "m": 1000,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
}
NUMBER = r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(mm|cm|m|inches|inch|in)?\b"


def parse_description(
    description: str, shape: str | None = None, existing: dict | None = None
) -> tuple[str, dict]:
    if not description.strip() or len(description) > 2000:
        raise ValueError("description must contain 1–2000 characters")
    if re.search(r"\b(ft|feet|foot)\b", description, re.I):
        raise ValueError("Use mm, cm, m, or inches")
    if re.search(
        r"\b(hole|holes|cutout|subtract|union|difference|intersection)\b",
        description,
        re.I,
    ):
        raise ValueError("Use create_model_from_scad for composite geometry")
    if shape is None:
        for candidate, pattern in SHAPES:
            match = re.search(rf"\b(?:{pattern})\b", description, re.I)
            if match:
                remainder = description[: match.start()] + description[match.end() :]
                if any(
                    re.search(rf"\b(?:{other})\b", remainder, re.I)
                    for _, other in SHAPES
                ):
                    raise ValueError("Use create_model_from_scad for multiple shapes")
                shape = candidate
                break
        else:
            raise ValueError(
                "No supported primitive found. Use explicit model_type/parameters or create_model_from_scad for complex models."
            )
    if shape not in DEFAULTS:
        raise ValueError("This model cannot be edited with a primitive description")
    values = dict(existing or {})
    found = False
    consumed_numbers = set()
    for key in DEFAULTS[shape]:
        if key in ("text", "center"):
            continue
        label = key.replace("_", "[ _]")
        aliases = {
            "width": "width|wide",
            "height": "height|high|tall",
            "depth": "depth|deep|long",
            "thickness": "thickness|thick",
        }.get(key, label)
        patterns = [
            rf"\b(?:{label})\s*(?:=|:|of|to)?\s*{NUMBER}",
            rf"{NUMBER}\s*(?:{aliases})\b",
        ]
        matches = [
            m for pattern in patterns for m in re.finditer(pattern, description, re.I)
        ]
        if matches:
            match = max(matches, key=lambda m: m.start())
            if key == "segments" and match.group(2):
                raise ValueError("segments is a count, not a length")
            value = (
                float(match.group(1))
                * UNITS[match.group(2).lower() if match.group(2) else None]
            )
            values[key] = (
                int(value) if key == "segments" and value.is_integer() else value
            )
            consumed_numbers.update(m.span(1) for m in matches)
            found = True
    if "radius" in DEFAULTS[shape]:
        diameter = re.search(
            rf"\bdiameter\s*(?:=|:|of|to)?\s*{NUMBER}", description, re.I
        )
        if diameter:
            values["radius"] = (
                float(diameter[1])
                * UNITS[diameter[2].lower() if diameter[2] else None]
                / 2
            )
            consumed_numbers.add(diameter.span(1))
            found = True
    if "center" in DEFAULTS[shape] and re.search(
        r"\bcent(er|re)(ed)?\b", description, re.I
    ):
        values["center"] = not bool(
            re.search(r"\b(?:not|un)[ -]?cent(er|re)(ed)?\b", description, re.I)
        )
        found = True
    if existing is not None and not found:
        raise ValueError(
            "No parameter changes recognized. Use e.g. 'height 30 mm' or explicit parameters."
        )
    number_spans = {
        m.span() for m in re.finditer(r"-?(?:\d+(?:\.\d*)?|\.\d+)", description)
    }
    if number_spans - consumed_numbers:
        raise ValueError(
            "Dimensions were not understood. Name each parameter, e.g. 'width 30 mm height 20 mm'."
        )
    return shape, parameters_for(shape, values)
