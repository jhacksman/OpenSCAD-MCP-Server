import pytest

from openscad_mcp.engine import OpenSCAD
from openscad_mcp.service import ModelService


@pytest.fixture
def engine():
    # Missing OpenSCAD is a failure: a green suite must include actual rendering.
    result = OpenSCAD(timeout=120)
    result.version()
    return result


@pytest.fixture
def service(tmp_path, engine):
    return ModelService(tmp_path / "models", engine)
