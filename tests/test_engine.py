import subprocess
from pathlib import Path

import pytest

from openscad_mcp.engine import OpenSCAD, OpenSCADError


def test_missing_executable():
    with pytest.raises(OpenSCADError, match="Cannot launch"):
        OpenSCAD("/no/such/openscad").version()


@pytest.mark.parametrize("mode", ["timeout", "empty", "warning", "error", "nonzero"])
def test_failures_do_not_replace_existing_outputs(tmp_path, monkeypatch, mode):
    destination = tmp_path / "model.stl"
    destination.write_bytes(b"previous valid revision")

    def run(command, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        output = Path(command[command.index("-o") + 1])
        if mode != "empty":
            output.write_bytes(b"bad output")
        message = {
            "warning": "WARNING: Ignoring unknown module",
            "error": "ERROR: Assertion failed",
        }.get(mode, "")
        return subprocess.CompletedProcess(
            command, 1 if mode == "nonzero" else 0, "", message
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(OpenSCADError):
        OpenSCAD("openscad").export(tmp_path / "source.scad", destination)
    assert destination.read_bytes() == b"previous valid revision"
    assert not list(tmp_path.glob(".render-*"))
