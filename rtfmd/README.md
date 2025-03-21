# Reasoning Trace Framework for OpenSCAD MCP Server

This directory contains the Reasoning Trace Framework (RTF) documentation for the OpenSCAD MCP Server project. The RTF provides insight into the design decisions, mental models, and reasoning processes behind the implementation.

## Directory Structure

- `/rtfmd/files/` - Shadow file system mirroring the actual source code structure
  - Contains `.md` files with the same names as their corresponding source files
  - Each file documents the reasoning behind the implementation

- `/rtfmd/knowledge/` - Domain knowledge documentation
  - `/openscad/` - Knowledge about OpenSCAD and 3D modeling
  - `/ai/` - Knowledge about AI and natural language processing
  - `/nlp/` - Knowledge about natural language parameter extraction

- `/rtfmd/decisions/` - Architectural decision records
  - Documents major design decisions and their rationales

## How to Use This Documentation

1. Start with the `/rtfmd/files/src/main.py.md` file to understand the overall architecture
2. Explore specific components through their corresponding `.md` files
3. Refer to the knowledge directory for domain-specific information
4. Review the decisions directory for major architectural decisions

## Tags Used

- `<metadata>` - File metadata including author, timestamp, version, etc.
- `<exploration>` - Documents the exploration process and alternatives considered
- `<mental-model>` - Explains the mental model used in the implementation
- `<pattern-recognition>` - Identifies design patterns used
- `<trade-off>` - Documents trade-offs considered and choices made
- `<domain-knowledge>` - References to domain knowledge required
- `<technical-debt>` - Acknowledges technical debt and future improvements
- `<knowledge-refs>` - References to related knowledge documents

## Contributing

When modifying the codebase, please update the corresponding RTF documentation to reflect your reasoning and design decisions.
