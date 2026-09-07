import json

import pytest

from openscad_mcp.geometry import generate_source, parameters_for, parse_description


@pytest.mark.parametrize(
    ("description", "shape", "expected"),
    [
        (
            "hollow box width 30 mm depth 20 mm height 15 mm thickness 2 mm",
            "box",
            {"width": 30, "thickness": 2},
        ),
        ("rounded box width 40 mm", "rounded_box", {"width": 40}),
        (
            "tube outer_radius 12 inner_radius 6 height 30",
            "tube",
            {"outer_radius": 12, "inner_radius": 6},
        ),
        (
            "cube 2 cm wide 1 inch high depth 5 mm",
            "cube",
            {"width": 20, "height": 25.4, "depth": 5},
        ),
        ("sphere diameter 30 mm", "sphere", {"radius": 15}),
        ("cube centered width 12", "cube", {"center": True, "width": 12}),
        ("cube not centered width 12", "cube", {"center": False}),
    ],
)
def test_descriptions(description, shape, expected):
    actual_shape, parameters = parse_description(description)
    assert actual_shape == shape
    assert expected.items() <= parameters.items()


@pytest.mark.parametrize(
    "description",
    [
        "a rabbit",
        "cube 20x30x40",
        "cube width -3 mm",
        "cube height 0",
        "cube width 3 feet",
    ],
)
def test_reject_unrecognized_or_invalid_descriptions(description):
    with pytest.raises(ValueError):
        parse_description(description)


@pytest.mark.parametrize(
    ("shape", "values"),
    [
        ("cube", {"width": float("nan")}),
        ("cube", {"width": float("inf")}),
        ("cube", {"width": True}),
        ("cube", {"center": "false"}),
        ("cube", {"radius": 3}),
        ("tube", {"inner_radius": 12}),
        ("box", {"thickness": 11}),
        ("rounded_box", {"radius": 20}),
        ("sphere", {"segments": 2}),
        ("torus", {"minor_radius": 20}),
        ("custom", {}),
    ],
)
def test_reject_invalid_parameters(shape, values):
    with pytest.raises(ValueError):
        parameters_for(shape, values)


def test_edit_preserves_unspecified_dimensions():
    existing = parameters_for("cube", {"width": 23, "center": True})
    _, result = parse_description("height 32 mm", "cube", existing)
    assert result == existing | {"height": 32}
    assert existing["height"] == 10
    with pytest.raises(ValueError, match="No parameter changes"):
        parse_description("make it prettier", "cube", existing)


def test_source_is_self_contained_and_strings_are_escaped():
    value = 'say "hello";\n cube(999); //'
    source, _ = generate_source("text", {"text": value})
    assert f"text = {json.dumps(value)};" in source
    assert "include <" not in source


@pytest.mark.parametrize(
    "description",
    [
        "cube width 20 radius 3",
        "cube with a hole",
        "cube and sphere",
        "sphere segments 24 mm",
    ],
)
def test_does_not_ignore_unsupported_dimensions_or_operations(description):
    with pytest.raises(ValueError):
        parse_description(description)


def test_huge_number_is_validation_error():
    with pytest.raises(ValueError):
        parameters_for("cube", {"width": 10**400})
