# Issue follow-up — 2026-09-07

PR #21 is merged into `main`. Issues #17 (startup/demo), #18 (MCP endpoint), and #20 (Open3D installation) received replies with the merged fix, current commands, scope, and test evidence, and were closed.

## Additional local verification

- Python **3.13.2**, macOS arm64: **59 tests passed in 22.00 seconds**, with actual OpenSCAD and both MCP transports. This verifies the Python 3.13 family on this Mac; it does not reproduce the reporter's exact M3/Conda/Python 3.13.11 installation.
- Built the root Dockerfile in a dedicated Colima Linux arm64 VM. The image runs Python 3.13, Debian Bookworm's OpenSCAD 2021.01, Xvfb, and software OpenGL.
- `scripts/test_docker.py` passed against the built image with **networking disabled**, no GPU, and UID **10001**. It exercised tool discovery, actual geometry and analytic volume, all four PNG views, SCAD/STL/CSG/3MF exports, modification, failed-edit rollback, and persistence after replacing the container.
- The Docker test cleans up its disposable containers and named volume. The image is retained locally as `openscad-mcp:issue14`; the dedicated Docker context is `colima-openscad-mcp`.
- `glama.json` validates against Glama's published JSON schema. The owner selected MIT; the root license and package license metadata have been added.

CI now includes native Python 3.12/3.13 tests and real Docker builds/tests on amd64 and arm64 runners. Remote results are tracked on the follow-up pull request.

## Remaining external checks

The Jetson question is answered in [Jetson Orin Nano and CUDA](jetson.md). General CUDA capability and successful ARM64 container testing do not establish physical Jetson or archived reconstruction support.

The repository changes address the missing Dockerfile and license in #14. The [Glama owner listing](https://glama.ai/mcp/servers/jhacksman/OpenSCAD-MCP-Server) still needs to sync the merged revision and pass Glama's own build/tool checks. No Glama deployment or score improvement is claimed by the local test results.
