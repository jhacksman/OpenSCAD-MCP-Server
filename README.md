# OpenSCAD MCP Server

A local MCP server that turns OpenSCAD source or explicit primitive dimensions into editable SCAD, STL geometry, and four PNG previews. The connected assistant can write SCAD for complex designs; this server compiles and renders it with the installed OpenSCAD executable.

**Status:** the supported implementation is in `src/openscad_mcp`. The previous image generation, CUDA reconstruction, remote processing, and printer code is archived in [`legacy/`](legacy/README.md). It is unfinished and is not part of the installed server. See the [audit and machine test results](docs/audit.md).

## Install

Requires Python 3.11+ and OpenSCAD. No API keys, CUDA, Open3D, or separate image renderer are required.

Install OpenSCAD using the [official downloads](https://openscad.org/downloads.html). On macOS, the current native Apple Silicon/Intel build is available with:

```sh
brew install --cask openscad@snapshot
openscad --version
```

On Debian/Ubuntu, `sudo apt-get install openscad` installs the distribution package. Linux PNG rendering may need an X display; use `xvfb-run -a` around the server or tests on a headless machine. Windows users can set `OPENSCAD_EXECUTABLE` to the full path of `openscad.exe`.

```sh
git clone https://github.com/jhacksman/OpenSCAD-MCP-Server.git
cd OpenSCAD-MCP-Server
uv sync --locked
uv run openscad-mcp --help
```

Alternatively, install into a virtual environment with `python -m pip install .` and run `openscad-mcp`. `requirements.txt` installs this same package. `uv.lock` pins the tested dependency resolution; the project metadata permits compatible updates.

On an external filesystem that cannot create symlinks, put the environment on the internal disk:

```sh
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/openscad-mcp-venv"
uv sync --locked
```

## Docker and Jetson

A headless stdio image is available through the root `Dockerfile`. See [Docker setup and verification](docs/docker.md) for build commands, MCP client configuration, persistent model storage, and Glama listing requirements. The image needs no GPU.

For the hardware question in issue #19, see [Jetson Orin Nano and CUDA](docs/jetson.md). The supported server does not use CUDA; the archived reconstruction prototype is not restored by this Docker support.

## Connect an MCP client

Stdio is the default. Configure the client to launch the server process, using absolute paths:

```json
{
  "mcpServers": {
    "openscad": {
      "command": "/absolute/path/to/venv/bin/openscad-mcp",
      "args": ["--output-dir", "/absolute/path/to/models"]
    }
  }
}
```

For clients supporting Streamable HTTP:

```sh
uv run openscad-mcp --transport http --port 8000
```

The MCP endpoint is **`http://127.0.0.1:8000/mcp`**. `/` is server information; `/health` reports the OpenSCAD version and primitive defaults. HTTP also serves preview pages and downloads. In stdio mode, use returned absolute file paths or `get_model_preview`; there is no HTTP listener.

## Tools

| Tool | Purpose |
| --- | --- |
| `get_capabilities` | OpenSCAD version and supported primitives with default parameters |
| `create_3d_model` | Create a primitive using `model_type` and `parameters`, or a limited description |
| `create_model_from_scad` | Compile trusted `scad_code` for a custom 3D model |
| `modify_3d_model` | Change primitive parameters or replace a model's `scad_code` |
| `get_model` | Read a saved model and its artifact paths |
| `get_model_preview` | Return a PNG image as MCP image content |
| `export_model` | Export `scad`, `stl`, `csg`, or `3mf` |

Example arguments to `create_3d_model`:

```json
{
  "model_type": "box",
  "parameters": {"width": 40, "depth": 30, "height": 20, "thickness": 2}
}
```

Dimensions are in millimeters. Supported types: `cube` (also rectangular blocks), `sphere`, `cylinder`, `box` (open hollow box), `rounded_box` (solid), `tube`, `cone`, `torus`, `prism` (right triangular cross-section), `hexagonal_prism`, and `text`. Call `get_capabilities` for each type's exact parameter names. A torus uses the radius to the tube center (`major_radius`) and the tube radius (`minor_radius`).

The optional description parser recognizes named dimensions such as `hollow box width 40 mm depth 30 mm height 20 mm thickness 2 mm` or `cube 2 cm wide 1 inch high`. It supports mm, cm, m, and inches; omitted units mean mm. It is a small parser, not a general natural-language model. Unspecified dimensions use the returned defaults. Prefer explicit parameters for precision. For arbitrary objects, have the assistant write self-contained SCAD and call `create_model_from_scad`.

```json
{
  "scad_code": "difference() { cube([40,30,8]); translate([20,15,-1]) cylinder(h=10,r=5,$fn=48); }",
  "description": "Mounting plate with a through hole"
}
```

To modify a primitive:

```json
{"model_id": "ID_FROM_CREATION", "parameters": {"height": 25}}
```

Only **SCAD** retains editable source parameters. CSG is an evaluated geometry tree. STL and 3MF are meshes; they do not preserve the design's parametric relationships. AMF (removed in current OpenSCAD builds), STEP, OBJ, 2D exports, image reconstruction, and printing are not offered.

## HTTP smoke test

With the HTTP server running:

```sh
curl --fail http://127.0.0.1:8000/tool_call \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"create_3d_model","tool_params":{"model_type":"box","parameters":{"width":40,"depth":30,"height":20,"thickness":2}}}'
```

Open the returned `preview_url` on the same server. Download `/download/MODEL_ID?format=stl` or `?format=scad`. `/tool_call` is a convenience JSON API, separate from the actual MCP protocol at `/mcp`. Image retrieval over this API uses `/preview/VIEW/MODEL_ID`.

## Behavior and limits

- Creation succeeds only after a real STL and all four PNGs have been generated. Renderer failures, empty output, and OpenSCAD warnings/errors are reported as failures; no placeholder geometry or preview is substituted.
- Edits create a new revision and update the manifest only after successful rendering. Earlier revisions remain on disk. Models survive server restarts.
- Default storage is `~/.local/share/openscad-mcp/models`. Override with `--output-dir` or `OPENSCAD_OUTPUT_DIR`. Each render subprocess has a 120-second timeout, configurable with `--timeout`. A model requires five subprocesses, so the total request can take longer.
- Run one server process per output directory. Renders and edits are serialized. There is no automatic disk cleanup, distributed job queue, or multi-process store coordination.
- This is a trusted local tool. Custom SCAD executes with the current user's file access, including OpenSCAD `import`, `include`, and `use`; it is not sandboxed. HTTP binds to loopback and rejects non-local Host/Origin headers. There is no authentication or supported public/LAN deployment.
- Tested locally on macOS arm64 and in Ubuntu 24.04 CI under Xvfb. Windows and other platform/display combinations are unverified. No hardware printer or remote GPU was exercised.

## Development and verification

```sh
uv sync --locked --group dev
uv run ruff check src/openscad_mcp tests scripts
uv run ruff format --check src/openscad_mcp tests scripts
uv run pytest -q
```

The full suite requires OpenSCAD and fails if it is unavailable. Tests launch real stdio and HTTP server processes, use the official MCP client, download exports, decode PNGs, inspect mesh topology/dimensions/volume, test persistence and failed edits, and exercise invalid requests. `uv run pytest -m 'not integration'` runs the unit checks only and does not establish end-to-end correctness.

## License

[MIT](LICENSE), copyright 2026 Jack Hacksman.
