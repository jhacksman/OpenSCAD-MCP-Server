"""Persistent models with immutable revisions and commit-on-success edits."""

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from .engine import FORMATS, VIEWS, OpenSCAD
from .geometry import generate_source, parse_description


class ModelNotFound(ValueError):
    pass


class ModelService:
    def __init__(self, output_dir: str | Path, engine: OpenSCAD | None = None):
        self.root = Path(output_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine or OpenSCAD()
        # One process owns this store. Serialize renders/edits so updates cannot race.
        self.lock = threading.RLock()

    def _directory(self, model_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", model_id):
            raise ModelNotFound("Invalid model ID")
        return self.root / model_id

    def _load(self, model_id: str) -> dict:
        try:
            return json.loads((self._directory(model_id) / "model.json").read_text())
        except FileNotFoundError as exc:
            raise ModelNotFound(f"Model {model_id} not found") from exc

    def _save(self, model_id: str, metadata: dict):
        directory = self._directory(model_id)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, delete=False, suffix=".json"
        ) as stream:
            temporary = Path(stream.name)
            json.dump(metadata, stream, indent=2)
        try:
            os.replace(temporary, directory / "model.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _build(
        self, model_id: str, source: str, shape: str, parameters: dict, description: str
    ) -> dict:
        directory = self._directory(model_id)
        directory.mkdir(exist_ok=True)
        revision = uuid.uuid4().hex
        target = directory / revision
        target.mkdir()
        try:
            scad = target / "model.scad"
            scad.write_text(source, encoding="utf-8")
            self.engine.export(scad, target / "model.stl")
            for view in VIEWS:
                self.engine.export(scad, target / f"{view}.png", view)
            data = {
                "model_id": model_id,
                "revision": revision,
                "model_type": shape,
                "parameters": parameters,
                "description": description,
            }
            self._save(model_id, data)
        except Exception:
            shutil.rmtree(target)
            if not (directory / "model.json").exists():
                directory.rmdir()
            raise
        return self._public(data)

    def _public(self, data: dict) -> dict:
        model_id = data["model_id"]
        directory = self._directory(model_id) / data["revision"]
        return data | {
            "scad_file": str(directory / "model.scad"),
            "model_file": str(directory / "model.stl"),
            "preview_url": f"/ui/preview/{model_id}",
            "previews": {view: f"/preview/{view}/{model_id}" for view in VIEWS},
            "preview_files": {view: str(directory / f"{view}.png") for view in VIEWS},
            "download_url": f"/download/{model_id}?format=stl",
            "supported_formats": list(FORMATS),
        }

    def create(
        self,
        description: str = "",
        model_type: str | None = None,
        parameters: dict | None = None,
    ) -> dict:
        if model_type is None:
            model_type, extracted = parse_description(description)
            parameters = extracted | (parameters or {})
        source, parameters = generate_source(model_type, parameters)
        with self.lock:
            return self._build(
                uuid.uuid4().hex, source, model_type, parameters, description
            )

    def from_scad(self, source: str, description: str = "") -> dict:
        if not source.strip() or len(source.encode()) > 100_000:
            raise ValueError("SCAD source must contain 1–100000 bytes")
        with self.lock:
            return self._build(uuid.uuid4().hex, source, "custom", {}, description)

    def modify(
        self,
        model_id: str,
        modifications: str = "",
        parameters: dict | None = None,
        scad_code: str | None = None,
    ) -> dict:
        with self.lock:
            data = self._load(model_id)
            if scad_code is not None:
                if modifications or parameters:
                    raise ValueError("Supply scad_code alone when replacing source")
                if not scad_code.strip() or len(scad_code.encode()) > 100_000:
                    raise ValueError("SCAD source must contain 1–100000 bytes")
                return self._build(
                    model_id, scad_code, "custom", {}, data["description"]
                )
            if data["model_type"] == "custom":
                raise ValueError("Supply scad_code to replace a custom model's source")
            if not modifications and not parameters:
                raise ValueError("Supply modifications, parameters, or scad_code")
            values = data["parameters"]
            if modifications:
                _, values = parse_description(modifications, data["model_type"], values)
            source, values = generate_source(
                data["model_type"], values | (parameters or {})
            )
            return self._build(
                model_id, source, data["model_type"], values, data["description"]
            )

    def get(self, model_id: str) -> dict:
        return self._public(self._load(model_id))

    def export(self, model_id: str, format: str = "stl") -> dict:
        if format not in FORMATS:
            raise ValueError(f"Unsupported format. Choose from {', '.join(FORMATS)}")
        with self.lock:
            data = self._load(model_id)
            directory = self._directory(model_id) / data["revision"]
            output = directory / f"model.{format}"
            if not output.exists():
                self.engine.export(directory / "model.scad", output)
            return {
                "model_id": model_id,
                "format": format,
                "model_file": str(output),
                "download_url": f"/download/{model_id}?format={format}",
            }

    def preview(self, model_id: str, view: str) -> Path:
        if view not in VIEWS:
            raise ModelNotFound("Unknown preview view")
        data = self._load(model_id)
        return self._directory(model_id) / data["revision"] / f"{view}.png"
