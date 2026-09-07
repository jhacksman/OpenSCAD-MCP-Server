# Docker

The root `Dockerfile` builds a headless **stdio MCP** server. It includes OpenSCAD, Xvfb, software OpenGL, and Liberation fonts. The runtime uses an unprivileged user (UID 10001); no host display, GPU, CUDA, or API keys are needed. Python dependencies come from `uv.lock`. Debian packages and the Python base image receive upstream updates at build time.

## Build and connect

```sh
docker build -t openscad-mcp:local .
docker run --rm -i --network none \
  -v openscad-models:/data \
  openscad-mcp:local
```

The process waits for MCP messages on stdin. Use `-i` and **do not use `-t`**, which would mix terminal output with protocol messages. Configure a local MCP client to run:

```json
{
  "mcpServers": {
    "openscad": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--network", "none",
        "-v", "openscad-models:/data",
        "openscad-mcp:local"
      ]
    }
  }
}
```

The named volume preserves models across container restarts. A fresh named volume receives the image's `/data` ownership. If you use a bind mount or an existing volume, make sure UID 10001 can write to it. Run only one server per volume.

Returned paths such as `/data/MODEL_ID/REVISION/model.stl` are **inside the container**. `get_model_preview` returns the image directly over MCP. To copy an export to the host while the container is running, find its name with `docker ps`, then use the exact `model_file` returned by `export_model`:

```sh
docker cp CONTAINER_NAME:/data/MODEL_ID/REVISION/model.stl ./model.stl
```

The image does not expose an HTTP port. For browser previews and HTTP clients, use the native `--transport http` setup in the root README. Container port publishing does not make the server's loopback-only HTTP listener accessible.

Custom SCAD can access files within the container and any mounted directories. Mount only the model data and assets you intend it to read. Networking is not required for generation; the example disables it.

## Test the image

Install development dependencies on the host, build the image, and run:

```sh
uv sync --locked --group dev
uv run python scripts/test_docker.py --image openscad-mcp:local
```

This starts disposable containers with networking disabled and an isolated named volume. It uses the actual MCP protocol, verifies non-root execution, creates a hollow box, checks mesh dimensions and volume, decodes all four views, retrieves every supported export, edits the model, rejects invalid SCAD without losing the good revision, and verifies persistence in a fresh container. The script removes its containers and test volume afterward.

On macOS, a Docker-compatible Linux VM is required. The local test used a dedicated Colima profile with Docker context `colima-openscad-mcp`; this does not require changing your default Docker context:

```sh
colima start --profile openscad-mcp --vm-type vz --cpu 4 --memory 6 --disk 20 --activate=false
DOCKER_CONTEXT=colima-openscad-mcp docker build -t openscad-mcp:local .
DOCKER_CONTEXT=colima-openscad-mcp uv run python scripts/test_docker.py --image openscad-mcp:local
```

## Glama listing

`Dockerfile`, the MIT `LICENSE`, and `glama.json` identifying `jhacksman` as maintainer address the repository prerequisites discussed in issue #14. Glama documents the maintainer file and ownership synchronization in [What is glama.json?](https://glama.ai/blog/2025-07-08-what-is-glamajson).

A successful local image test is not a Glama deployment. After merging, sync/claim the existing listing through its owner account and run Glama's build/tool checks. No external account permissions, hosted deployment, or listing score changes are implied by these repository files.
