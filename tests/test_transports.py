"""Real subprocesses, real MCP clients, real HTTP downloads, real OpenSCAD."""

import asyncio
import base64
import io
import json
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import trimesh
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from PIL import Image

pytestmark = pytest.mark.integration


def tool_data(result):
    assert not result.isError, result
    return result.structuredContent or json.loads(result.content[0].text)


@asynccontextmanager
async def http_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (tmp_path / "server.log").open("w+")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openscad_mcp",
            "--transport",
            "http",
            "--port",
            str(port),
            "--output-dir",
            str(tmp_path / "models"),
        ],
        cwd=tmp_path,
        stdout=log,
        stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(base_url=base, timeout=180) as client:
            for _ in range(150):
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail(log.read())
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Server failed to become healthy")
            yield base, client
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


async def test_stdio_end_to_end(tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "openscad_mcp", "--output-dir", str(tmp_path / "models")],
        cwd=str(tmp_path),
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "create_3d_model",
                "create_model_from_scad",
                "modify_3d_model",
                "get_model",
                "get_model_source",
                "get_model_preview",
                "get_capabilities",
                "export_model",
            }
            caps = tool_data(await session.call_tool("get_capabilities", {}))
            assert "OpenSCAD" in caps["openscad"]
            model = tool_data(
                await session.call_tool(
                    "create_3d_model",
                    {
                        "model_type": "cube",
                        "parameters": {
                            "width": 12,
                            "depth": 15,
                            "height": 9,
                            "center": True,
                        },
                    },
                )
            )
            model_id = model["model_id"]
            mesh = trimesh.load_mesh(model["model_file"])
            assert mesh.volume == pytest.approx(1620)
            assert mesh.bounds.mean(axis=0).tolist() == pytest.approx([0, 0, 0])
            preview = await session.call_tool(
                "get_model_preview", {"model_id": model_id}
            )
            assert not preview.isError
            image = next(block for block in preview.content if block.type == "image")
            with Image.open(io.BytesIO(base64.b64decode(image.data))) as png:
                assert png.size == (800, 600)
            source = tool_data(
                await session.call_tool("get_model_source", {"model_id": model_id})
            )
            assert source["scad_code"] == Path(model["scad_file"]).read_text()
            assert source["revision"] == model["revision"]
            modified = tool_data(
                await session.call_tool(
                    "modify_3d_model",
                    {
                        "model_id": model_id,
                        "parameters": {"height": 18},
                        "expected_revision": source["revision"],
                    },
                )
            )
            assert trimesh.load_mesh(modified["model_file"]).volume == pytest.approx(
                3240
            )
            stale = await session.call_tool(
                "modify_3d_model",
                {
                    "model_id": model_id,
                    "parameters": {"height": 99},
                    "expected_revision": source["revision"],
                },
            )
            assert stale.isError
            assert "Model changed" in stale.content[0].text
            exported = tool_data(
                await session.call_tool(
                    "export_model", {"model_id": model_id, "format": "3mf"}
                )
            )
            assert Path(exported["model_file"]).read_bytes().startswith(b"PK")
            assert (
                await session.call_tool(
                    "create_model_from_scad", {"scad_code": "bad SCAD !!!"}
                )
            ).isError
            assert (
                await session.call_tool("get_model", {"model_id": "../../etc/passwd"})
            ).isError
    # New process must recover metadata and artifacts from disk.
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            persisted = tool_data(
                await session.call_tool("get_model", {"model_id": model_id})
            )
            assert persisted["parameters"]["height"] == 18
            saved_source = tool_data(
                await session.call_tool("get_model_source", {"model_id": model_id})
            )
            assert saved_source["revision"] == persisted["revision"]
            assert "height = 18;" in saved_source["scad_code"]


async def test_http_mcp_and_downloads(tmp_path):
    async with http_server(tmp_path) as (base, client):
        assert (await client.get("/")).json()["mcp_endpoint"] == "/mcp"
        async with streamable_http_client(base + "/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert len((await session.list_tools()).tools) == 8
                model = tool_data(
                    await session.call_tool(
                        "create_model_from_scad",
                        {
                            "scad_code": "difference() { cube([30,20,8]); translate([15,10,-1]) cylinder(h=10,r=4,$fn=48); }",
                            "description": '<script>alert("x")</script>',
                        },
                    )
                )
                source = tool_data(
                    await session.call_tool(
                        "get_model_source", {"model_id": model["model_id"]}
                    )
                )
                assert source["scad_code"].startswith("difference()")
                changed = tool_data(
                    await session.call_tool(
                        "modify_3d_model",
                        {
                            "model_id": model["model_id"],
                            "scad_code": source["scad_code"].replace(
                                "[30,20,8]", "[36,20,8]"
                            ),
                            "expected_revision": source["revision"],
                        },
                    )
                )
                fetched = await client.post(
                    "/tool_call",
                    json={
                        "tool_name": "get_model_source",
                        "tool_params": {"model_id": model["model_id"]},
                    },
                )
                assert fetched.status_code == 200
                assert "[36,20,8]" in fetched.json()["scad_code"]
                conflict = await client.post(
                    "/tool_call",
                    json={
                        "tool_name": "modify_3d_model",
                        "tool_params": {
                            "model_id": model["model_id"],
                            "scad_code": "cube(100);",
                            "expected_revision": source["revision"],
                        },
                    },
                )
                assert conflict.status_code == 409
                assert conflict.json()["current_revision"] == changed["revision"]
                page = await client.get(model["preview_url"])
                assert page.status_code == 200
                assert "<script>" not in page.text
                assert "&lt;script&gt;" in page.text
                for path in model["previews"].values():
                    result = await client.get(path)
                    assert result.status_code == 200
                    assert result.headers["content-type"] == "image/png"
                    Image.open(io.BytesIO(result.content)).verify()
                for format in ("stl", "scad", "csg", "3mf"):
                    response = await client.get(
                        f"/download/{model['model_id']}", params={"format": format}
                    )
                    assert response.status_code == 200, response.text
                    assert len(response.content) > 0
                    if format == "stl":
                        mesh = trimesh.load_mesh(
                            io.BytesIO(response.content), file_type="stl"
                        )
                        assert mesh.extents.tolist() == pytest.approx([36, 20, 8])
                    if format == "scad":
                        assert response.text.startswith("difference()")
                invalid = await session.call_tool(
                    "export_model", {"model_id": model["model_id"], "format": "step"}
                )
                assert invalid.isError
        response = await client.post(
            "/tool_call",
            json={
                "tool_name": "create_3d_model",
                "tool_params": {"description": "sphere radius 7 mm"},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["parameters"]["radius"] == 7
        for payload, status in [
            ({"tool_name": "unknown"}, 404),
            ({"tool_name": "modify_3d_model", "tool_params": {}}, 422),
            (
                {
                    "tool_name": "create_3d_model",
                    "tool_params": {"model_type": "cube", "unknown": 1},
                },
                422,
            ),
            (
                {
                    "tool_name": "create_model_from_scad",
                    "tool_params": {"scad_code": "bad !!!"},
                },
                422,
            ),
            ({"tool_name": "get_model", "tool_params": {"model_id": "bad"}}, 404),
            (
                {"tool_name": "get_model_source", "tool_params": {"model_id": "bad"}},
                404,
            ),
        ]:
            assert (await client.post("/tool_call", json=payload)).status_code == status
        assert (
            await client.post(
                "/tool_call",
                content="{broken",
                headers={"Content-Type": "application/json"},
            )
        ).status_code == 422
        assert (
            await client.get("/", headers={"Host": "evil.example"})
        ).status_code == 400
        assert (
            await client.post(
                "/tool_call",
                json={"tool_name": "get_capabilities"},
                headers={"Origin": "https://evil.example"},
            )
        ).status_code == 403
