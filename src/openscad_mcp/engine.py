"""OpenSCAD subprocess execution with verified, atomically replaced outputs."""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

VIEWS = {
    "perspective": "0,0,0,55,0,25,100",
    "front": "0,0,0,90,0,0,100",
    "top": "0,0,0,0,0,0,100",
    "right": "0,0,0,90,0,90,100",
}
FORMATS = ("scad", "stl", "csg", "3mf")


class OpenSCADError(RuntimeError):
    pass


class OpenSCAD:
    def __init__(self, executable: str | None = None, timeout: float = 120):
        requested = executable or os.environ.get("OPENSCAD_EXECUTABLE")
        self.executable = requested or shutil.which("openscad")
        if not self.executable:
            for candidate in (
                "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
                "/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD",
            ):
                if Path(candidate).is_file():
                    self.executable = candidate
                    break
        if not self.executable:
            raise OpenSCADError(
                "OpenSCAD not found. Install it or set OPENSCAD_EXECUTABLE to its executable."
            )
        self.timeout = timeout

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                [self.executable, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise OpenSCADError(
                f"OpenSCAD exceeded the {self.timeout:g}s timeout"
            ) from exc
        except OSError as exc:
            raise OpenSCADError(f"Cannot launch OpenSCAD: {exc}") from exc
        diagnostics = result.stderr + result.stdout
        # Some OpenSCAD versions report errors (including failed assertions) with exit code zero.
        if result.returncode or re.search(r"(?:ERROR|WARNING):", diagnostics):
            raise OpenSCADError(f"OpenSCAD failed: {diagnostics[-6000:].strip()}")
        return result

    def version(self) -> str:
        result = self._run(["--version"])
        return (result.stdout + result.stderr).strip()

    def export(self, source: Path, destination: Path, view: str | None = None) -> Path:
        suffix = destination.suffix.lower()
        if view is not None and view not in VIEWS:
            raise ValueError(f"Unknown view: {view}")
        if suffix not in {".stl", ".csg", ".3mf", ".png"}:
            raise ValueError(f"Unsupported output extension: {suffix}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".render-", dir=destination.parent
        ) as temporary:
            output = Path(temporary) / destination.name
            args = ["-o", str(output)]
            if suffix == ".png":
                args += [
                    "--render",
                    "--autocenter",
                    "--viewall",
                    "--imgsize=800,600",
                    f"--camera={VIEWS[view or 'perspective']}",
                ]
            args.append(str(source.resolve()))
            self._run(args)
            if not output.is_file() or output.stat().st_size == 0:
                raise OpenSCADError("OpenSCAD produced no geometry/output")
            if suffix == ".png" and not output.read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            ):
                raise OpenSCADError("OpenSCAD did not produce a PNG image")
            os.replace(output, destination)
        return destination
