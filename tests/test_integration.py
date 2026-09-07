import hashlib
import math
import shutil
import zipfile
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from openscad_mcp.engine import OpenSCADError
from openscad_mcp.geometry import DEFAULTS, generate_source
from openscad_mcp.service import ModelNotFound, ModelService

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("shape", list(DEFAULTS))
def test_every_primitive_is_real_closed_geometry(tmp_path, engine, shape):
    source, _ = generate_source(shape)
    scad = tmp_path / f"{shape}.scad"
    scad.write_text(source)
    stl = engine.export(scad, tmp_path / f"{shape}.stl")
    mesh = trimesh.load_mesh(stl)
    assert len(mesh.faces) >= 4
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    if shape == "cube":
        assert mesh.extents.tolist() == pytest.approx([10, 10, 10])
        assert mesh.volume == pytest.approx(1000)
    if shape == "box":
        assert mesh.volume == pytest.approx(30 * 20 * 15 - 26 * 16 * 13)
    if shape == "tube":
        assert mesh.volume == pytest.approx(math.pi * (100 - 64) * 20, rel=0.01)
    if shape == "prism":
        assert mesh.volume == pytest.approx(2000)


def test_full_model_lifecycle(service, tmp_path):
    model = service.create(
        "hollow box width 30 mm depth 20 mm height 15 mm thickness 2 mm"
    )
    model_id = model["model_id"]
    hashes = []
    for path in model["preview_files"].values():
        with Image.open(path) as image:
            assert image.size == (800, 600)
            assert image.convert("RGB").getextrema() != ((0, 0),) * 3
            assert len(image.getcolors(800 * 600)) >= 2
            # A filled silhouette must occupy a meaningful part of the image.
            colors = image.convert("RGB").getcolors(800 * 600)
            assert max(count for count, _ in colors) < 800 * 600 * 0.99
        hashes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    assert len(set(hashes)) == 4
    for format in ("scad", "stl", "csg", "3mf"):
        exported = Path(service.export(model_id, format)["model_file"])
        assert exported.stat().st_size > 100
        if format == "3mf":
            with zipfile.ZipFile(exported) as archive:
                assert any(name.endswith(".model") for name in archive.namelist())
        if format == "scad":
            # The download works outside the source checkout, without template includes.
            portable = tmp_path / "elsewhere" / "portable.scad"
            portable.parent.mkdir()
            shutil.copyfile(exported, portable)
            service.engine.export(portable, portable.with_suffix(".stl"))
    changed = service.modify(model_id, "height 25 mm")
    assert changed["parameters"]["height"] == 25
    assert changed["parameters"]["thickness"] == 2
    assert trimesh.load_mesh(changed["model_file"]).extents.tolist() == pytest.approx(
        [30, 20, 25]
    )
    reloaded = ModelService(service.root, service.engine).get(model_id)
    assert reloaded == changed
    with pytest.raises(OpenSCADError):
        service.modify(model_id, scad_code="this is invalid SCAD !!!")
    assert service.get(model_id) == changed
    assert (
        len(list((service.root / model_id).iterdir())) == 3
    )  # two revisions and manifest


@pytest.mark.parametrize(
    "source",
    [
        "nonsense !!!",
        "",
        "// no geometry",
        "unknown_module();",
        'assert(false, "broken"); cube(2);',
    ],
)
def test_invalid_source_never_creates_success(service, source):
    with pytest.raises((ValueError, OpenSCADError)):
        service.from_scad(source)
    assert not list(service.root.iterdir())


def test_custom_geometry(service):
    model = service.from_scad(
        "difference() { cube([40,30,8]); translate([20,15,-1]) cylinder(h=10,r=5,$fn=48); }"
    )
    mesh = trimesh.load_mesh(model["model_file"])
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(40 * 30 * 8 - math.pi * 25 * 8, rel=0.002)
    with pytest.raises(ValueError, match="scad_code"):
        service.modify(model["model_id"], parameters={"width": 2})


@pytest.mark.parametrize("model_id", ["../../etc/passwd", "bad", "a" * 32])
def test_unknown_ids(service, model_id):
    with pytest.raises(ModelNotFound):
        service.get(model_id)
