# Archived prototype — unsupported

This directory preserves the previous implementation and its historical notes/tests for reference. It is **not installed, imported, or exposed** by the supported server. The old imports and entrypoints are not maintained. Do not use the original README here as setup instructions.

The September 2026 review found:

- `src/main.py` imports nonexistent MCP SDK classes, a removed SAM module, and several undefined components. Its `/tool_call` route is not an MCP transport.
- `CodeGenerator.generate_code` returns a filename, while the entrypoint treats it as source text. It also points at the wrong template directory.
- CUDA reconstruction synthesizes uncalibrated camera matrices and its OBJ conversion writes a dummy triangle. The provided mock tests do not establish working reconstruction.
- Printer discovery returns simulated printer entries.
- Renderer fallbacks produce placeholder images after errors.
- The original installation pulls heavy unrelated packages (including Open3D) into the base server, obstructing Apple Silicon installation.
- Parametric export claims incorrectly include mesh formats and evaluated CSG. Several helpers silently ignore unsupported operations or fall back to a cube.

The root README documents the replacement implementation. Historical tracked Python bytecode was removed rather than preserved. Reviving an experiment requires a separate implementation and genuine integration tests; passing the archived mocks would not establish support.
