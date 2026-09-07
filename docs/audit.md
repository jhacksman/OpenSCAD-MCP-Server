# Refresh audit — 2026-09-06

Reviewed upstream `6948c2d` on macOS 26.5.2 arm64. The starting workspace was empty; the repository was cloned and work isolated on `refresh/local-e2e`.

## Verdict

The original checkout could not substantiate its advertised end-to-end workflow. The supported server has been replaced with a smaller implementation that directly compiles and renders OpenSCAD, exposes actual MCP transports, and verifies its outputs. Image reconstruction and printer features remain archived, not certified.

## Confirmed problems and disposition

| Original problem | Evidence | Change |
| --- | --- | --- |
| Startup failure | The official installed `mcp` package raises `ImportError` for `MCPServer`; `main.py` also references a missing SAM module and undefined classes | Use official `FastMCP`; clean entrypoint and package |
| Missing MCP endpoint | `/tool_call` was a custom HTTP dispatcher; issue #18 asks where to connect | Real stdio and Streamable HTTP at `/mcp`, tested with official clients |
| Source/file confusion | `CodeGenerator.generate_code` returns a path, which `main.py` writes as SCAD source | One source-generation interface and self-contained SCAD |
| Fake outputs | CUDA converter writes a single hardcoded triangle; renderers substitute placeholders | Archive those paths; fail when actual rendering fails |
| Unfounded reconstruction | Identical synthetic camera poses and placeholder mesh conversion; tests mock outputs | Remove reconstruction from the supported tools and claims |
| Incorrect dimensions and type selection | Original shape matching selects cube before hollow/rounded box; unit handling and edit defaults are unreliable | Explicit primitive parameters, validation, limited parser with unit tests |
| False export promises | Mesh formats and evaluated CSG described as preserving parametric properties | Only SCAD is described as editable source; test portable SCAD and mesh exports |
| Broken current AMF support | Native OpenSCAD 2026.09.05 rejects `.amf` with “Invalid suffix amf” | Do not advertise AMF |
| Installation overload | Unconditional Open3D/CUDA/image dependencies; issue #20 reports M3 installation failure | Three direct runtime dependencies; lockfile and clean-wheel installation |
| State and error handling | In-memory models, blocking async routes, missing timeouts, permissive CORS | Persistent revision manifests; worker-thread dispatch; timeouts; loopback HTTP; explicit errors |

Related upstream reports: [startup #17](https://github.com/jhacksman/OpenSCAD-MCP-Server/issues/17), [endpoint #18](https://github.com/jhacksman/OpenSCAD-MCP-Server/issues/18), [Apple Silicon install #20](https://github.com/jhacksman/OpenSCAD-MCP-Server/issues/20). No claims are made that the archived CUDA/Jetson workflow is fixed.

## Machine and dependency evidence

- macOS 26.5.2, arm64; Python 3.12.9.
- OpenSCAD 2026.09.05, installed through the official Homebrew `openscad@snapshot` cask. The old `openscad` cask is disabled; the [official downloads page](https://openscad.org/downloads.html) recommends the snapshot on macOS.
- MCP SDK 1.29.1, FastAPI 0.141.1, Uvicorn 0.52.4. Exact resolved dependencies are in `uv.lock`.
- The MEDIA volume does not support the virtual environment's links. The development environment is `/Users/jackhacksman/.cache/openscad-mcp-venv` on the internal disk. A separate wheel-check environment is `/Users/jackhacksman/.cache/openscad-mcp-wheel-check`.

## Verification

`pytest -q`: **59 passed in 54.19 seconds**, no skipped tests. `ruff check`, `ruff format --check`, and `git diff --check` pass. Wheel and source-distribution builds succeed. A clean environment installs the wheel, exposes the CLI, and successfully creates a cylinder through stdio MCP from `/tmp`, then retrieves it after restart.

The suite exercises:

- All 11 supported primitives through the actual OpenSCAD executable; closed meshes, winding consistency, positive volume, and representative analytic dimensions/volumes.
- Four real PNG views, image decoding and distinct outputs.
- SCAD, STL, CSG and 3MF exports; exported SCAD compiles from a different directory without template files.
- Parameter edits, preservation of unrelated dimensions, state reload, and rollback after invalid source.
- Empty geometry, malformed SCAD, unknown modules and failed assertions.
- Real stdio subprocess initialization, tool discovery, generation, image content, modification, export, errors and persistence after restart.
- Real HTTP subprocess initialization and MCP calls, browser HTML, image and file downloads, legacy JSON API validation, escaping, bad model IDs, and Host/Origin rejection.
- Fault injection for timeouts, warnings, nonzero exit, zero-exit errors and empty output. These tests complement real rendering; they do not replace it.

A separately generated mounting plate was inspected in PNG files and the browser preview page. Its STL is watertight, has 1,052 triangles and measures exactly **60 × 40 × 6 mm**. Measured volume is **13,722.5055 mm³** versus **13,721.4160 mm³** analytically (0.00794% difference from faceted circular holes). All four offered exports exist.

Local artifacts are retained under `output/manual-e2e/`, indexed by `output/manual-e2e-result.json`. These generated files are intentionally gitignored.

## Scope and remaining limits

This is a breaking replacement of the non-working prototype, not a repair of every archived experiment. The original source, scripts, and historical notes are preserved under `legacy/`; tracked bytecode was removed. The installable wheel contains only the supported package.

This macOS machine was exercised locally. The Ubuntu 24.04 CI run also passed all 59 tests in 14.71 seconds under Xvfb, plus lint and formatting ([run 34081686256](https://github.com/jhacksman/OpenSCAD-MCP-Server/actions/runs/34081686256)). Windows, other Linux/display configurations, remote GPUs, external image providers and hardware printers are not certified here. Custom SCAD is trusted code with the current user's filesystem access. One server process owns an output directory; edits are serialized and old revisions are retained. No public server deployment, mesh repair, slicing, or physical printing is claimed.
