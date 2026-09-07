"""Exercise the built image through real stdio MCP, without network or a GPU."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import trimesh
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image


def docker(*args, check=True):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check
    )


def data(result):
    assert not result.isError, result
    return result.structuredContent or json.loads(result.content[0].text)


@asynccontextmanager
async def session_for(image, volume):
    name = f"openscad-test-{uuid.uuid4().hex}"
    params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--network",
            "none",
            "-v",
            f"{volume}:/data",
            image,
        ],
        env=dict(os.environ),
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert docker("exec", name, "id", "-u").stdout.strip() == "10001"
                yield session, name
    finally:
        docker("rm", "-f", name, check=False)


async def check(image):
    volume = f"openscad-test-{uuid.uuid4().hex}"
    docker("volume", "create", volume)
    try:
        async with session_for(image, volume) as (session, name):
            assert len((await session.list_tools()).tools) == 7
            capabilities = data(await session.call_tool("get_capabilities", {}))
            model = data(
                await session.call_tool(
                    "create_3d_model",
                    {
                        "model_type": "box",
                        "parameters": {
                            "width": 40,
                            "depth": 30,
                            "height": 20,
                            "thickness": 2,
                        },
                    },
                )
            )
            model_id = model["model_id"]
            preview_hashes = []
            for view in ("perspective", "front", "top", "right"):
                result = await session.call_tool(
                    "get_model_preview", {"model_id": model_id, "view": view}
                )
                assert not result.isError, result
                block = next(block for block in result.content if block.type == "image")
                png = base64.b64decode(block.data)
                with Image.open(io.BytesIO(png)) as rendered:
                    assert rendered.size == (800, 600)
                    assert len(rendered.getcolors(800 * 600)) >= 2
                preview_hashes.append(hashlib.sha256(png).hexdigest())
            assert len(set(preview_hashes)) == 4
            with tempfile.TemporaryDirectory() as temporary:
                for format in ("stl", "scad", "csg", "3mf"):
                    exported = data(
                        await session.call_tool(
                            "export_model", {"model_id": model_id, "format": format}
                        )
                    )
                    local = Path(temporary) / f"model.{format}"
                    docker("cp", f"{name}:{exported['model_file']}", str(local))
                    assert local.stat().st_size > 0
                    if format == "stl":
                        mesh = trimesh.load_mesh(local)
                        assert mesh.is_watertight
                        assert mesh.extents.tolist() == [40, 30, 20]
                        assert abs(mesh.volume - (40 * 30 * 20 - 36 * 26 * 18)) < 0.01
                    elif format == "3mf":
                        with zipfile.ZipFile(local) as archive:
                            assert any(p.endswith(".model") for p in archive.namelist())
                modified = data(
                    await session.call_tool(
                        "modify_3d_model",
                        {"model_id": model_id, "parameters": {"height": 25}},
                    )
                )
                assert modified["parameters"]["height"] == 25
                failed = await session.call_tool(
                    "modify_3d_model",
                    {"model_id": model_id, "scad_code": "invalid SCAD !!!"},
                )
                assert failed.isError
                assert (
                    data(await session.call_tool("get_model", {"model_id": model_id}))[
                        "revision"
                    ]
                    == modified["revision"]
                )
        # A fresh container reuses the named volume and recovers the successful edit.
        async with session_for(image, volume) as (session, name):
            restored = data(
                await session.call_tool("get_model", {"model_id": model_id})
            )
            assert restored["parameters"]["height"] == 25
            assert restored["revision"] == modified["revision"]
            preview = await session.call_tool(
                "get_model_preview", {"model_id": model_id}
            )
            assert not preview.isError
        print(
            json.dumps(
                {
                    "image": image,
                    "openscad": capabilities["openscad"],
                    "result": "passed",
                    "checks": [
                        "nonroot",
                        "network disabled",
                        "MCP handshake and tools",
                        "geometry and volume",
                        "four PNG views",
                        "four exports",
                        "edit",
                        "failed-edit rollback",
                        "container restart persistence",
                    ],
                },
                indent=2,
            )
        )
    finally:
        docker("volume", "rm", volume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="openscad-mcp:local")
    asyncio.run(check(parser.parse_args().image))
