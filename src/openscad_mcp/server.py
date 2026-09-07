"""Official MCP transports plus local preview/download HTTP routes."""

import argparse
import asyncio
import html
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, validate_call
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .engine import OpenSCAD, OpenSCADError
from .geometry import DEFAULTS
from .service import ModelConflict, ModelNotFound, ModelService


def create_server(service: ModelService) -> tuple[FastMCP, FastAPI]:
    mcp = FastMCP(
        "OpenSCAD",
        stateless_http=True,
        json_response=True,
        instructions="Create parametric primitives with explicit dimensions in mm, or supply OpenSCAD source for complex models. Read the returned parameters. Only SCAD preserves editable source. This local server executes trusted SCAD with the current user's file access.",
    )

    async def create_3d_model(
        description: str = "",
        model_type: str | None = None,
        parameters: dict | None = None,
    ) -> dict:
        """Create a primitive, STL and four PNG views. Prefer model_type and parameters; descriptions only recognize named primitive dimensions. Use get_capabilities for defaults."""
        return await asyncio.to_thread(
            service.create, description, model_type, parameters
        )

    async def create_model_from_scad(scad_code: str, description: str = "") -> dict:
        """Compile trusted, self-contained 3D OpenSCAD source into STL and four PNG previews. No image reconstruction or implicit AI generation."""
        return await asyncio.to_thread(service.from_scad, scad_code, description)

    async def modify_3d_model(
        model_id: str,
        modifications: str = "",
        parameters: dict | None = None,
        scad_code: str | None = None,
        expected_revision: str | None = None,
    ) -> dict:
        """Edit primitive dimensions or replace source using scad_code. Pass expected_revision from get_model/get_model_source to reject stale edits. Failed edits leave the previous revision intact."""
        return await asyncio.to_thread(
            service.modify,
            model_id,
            modifications,
            parameters,
            scad_code,
            expected_revision,
        )

    async def export_model(model_id: str, format: str = "stl") -> dict:
        """Export scad (editable source), stl/3mf (meshes), or csg (evaluated geometry tree)."""
        return await asyncio.to_thread(service.export, model_id, format)

    async def get_model(model_id: str) -> dict:
        """Read a persisted model's parameters and local artifact paths."""
        return await asyncio.to_thread(service.get, model_id)

    async def get_model_source(model_id: str) -> dict:
        """Read the saved UTF-8 SCAD source and revision over MCP, including inside Docker. Edit scad_code and pass its revision as expected_revision to modify_3d_model."""
        return await asyncio.to_thread(service.source, model_id)

    async def get_model_preview(model_id: str, view: str = "perspective") -> Image:
        """Return an actual PNG image to the MCP client. Views: perspective, front, top, right."""
        path = await asyncio.to_thread(service.preview, model_id, view)
        return Image(path=str(path))

    async def get_capabilities() -> dict:
        """Report OpenSCAD version and supported primitives with defaults (all dimensions in mm)."""
        version = await asyncio.to_thread(service.engine.version)
        return {
            "version": __version__,
            "openscad": version,
            "primitives": DEFAULTS,
            "transports": ["stdio", "streamable-http"],
            "mcp_path": "/mcp",
        }

    functions = [
        create_3d_model,
        create_model_from_scad,
        modify_3d_model,
        export_model,
        get_model,
        get_model_source,
        get_capabilities,
    ]
    rest_tools = {}
    for function in functions:
        mcp.add_tool(function)
        rest_tools[function.__name__] = validate_call(
            config=ConfigDict(extra="forbid")
        )(function)
    mcp.add_tool(get_model_preview)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="OpenSCAD MCP Server", version=__version__, lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"]
    )

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit

            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or parsed.hostname not in (
                "localhost",
                "127.0.0.1",
                "::1",
            ):
                return JSONResponse(
                    {"detail": "Only local browser origins are allowed"},
                    status_code=403,
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ModelNotFound)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ModelConflict)
    async def conflict(request, exc):
        return JSONResponse(
            {"detail": str(exc), "current_revision": exc.current_revision},
            status_code=409,
        )

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(OpenSCADError)
    async def render_failed(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/")
    async def root():
        return {
            "name": "OpenSCAD MCP Server",
            "version": __version__,
            "mcp_endpoint": "/mcp",
            "tools": list(rest_tools) + ["get_model_preview"],
        }

    @app.get("/health")
    async def health():
        return await get_capabilities()

    class ToolCall(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tool_name: str
        tool_params: dict = Field(default_factory=dict)

    @app.post("/tool_call")
    async def tool_call(body: ToolCall):
        if body.tool_name not in rest_tools:
            return JSONResponse({"detail": "Unknown tool"}, status_code=404)
        try:
            return await rest_tools[body.tool_name](**body.tool_params)
        except ValidationError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/preview/{view}/{model_id}")
    async def preview(view: str, model_id: str):
        path = await asyncio.to_thread(service.preview, model_id, view)
        return FileResponse(
            path, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.get("/download/{model_id}")
    async def download(model_id: str, format: str = "stl"):
        data = await asyncio.to_thread(service.export, model_id, format)
        return FileResponse(data["model_file"], filename=f"{model_id}.{format}")

    @app.get("/ui/preview/{model_id}", response_class=HTMLResponse)
    async def preview_page(model_id: str):
        data = await asyncio.to_thread(service.get, model_id)
        views = "".join(
            f'<figure><img width="400" src="{url}" alt="{view}"><figcaption>{view}</figcaption></figure>'
            for view, url in data["previews"].items()
        )
        links = " · ".join(
            f'<a href="/download/{model_id}?format={fmt}">{fmt.upper()}</a>'
            for fmt in data["supported_formats"]
        )
        return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>OpenSCAD model</title><body><h1>{html.escape(data["description"] or data["model_type"])}</h1><pre>{html.escape(json.dumps(data["parameters"], indent=2))}</pre><nav>{links}</nav><main style="display:flex;flex-wrap:wrap">{views}</main></body></html>'

    app.mount("/", mcp_app)
    return mcp, app


def main():
    parser = argparse.ArgumentParser(description="Local OpenSCAD MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "OPENSCAD_OUTPUT_DIR", str(Path.home() / ".local/share/openscad-mcp/models")
        ),
    )
    parser.add_argument("--openscad", default=os.environ.get("OPENSCAD_EXECUTABLE"))
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    logging.basicConfig(level=logging.INFO)
    try:
        engine = OpenSCAD(args.openscad, args.timeout)
        engine.version()
        mcp, app = create_server(ModelService(args.output_dir, engine))
    except OpenSCADError as exc:
        parser.exit(1, f"{exc}\n")
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
