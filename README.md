# OpenSCAD MCP Server

A Model Context Protocol (MCP) server that enables users to describe 3D objects in natural language, collects design specifications, and generates OpenSCAD code to create parametric 3D models.

## Features

- **Natural Language Processing**: Extract design parameters from natural language descriptions
- **OpenSCAD Integration**: Generate parametric 3D models using OpenSCAD
- **Preview Generation**: Create multi-angle preview images of models
- **Parametric Export**: Export models in formats that preserve parametric properties (CSG, AMF, 3MF, SCAD)
- **3D Printer Discovery**: Optional network printer discovery and direct printing

## Architecture

The server is built using the Python MCP SDK and follows a modular architecture:

```
openscad-mcp-server/
├── src/
│   ├── main.py                  # Main application
│   ├── openscad_wrapper/        # OpenSCAD CLI wrapper
│   ├── nlp/                     # Natural language processing
│   ├── models/                  # OpenSCAD code generation
│   ├── visualization/           # Preview generation and web interface
│   ├── utils/                   # Utility functions
│   └── printer_discovery/       # 3D printer discovery
├── scad/                        # Generated OpenSCAD files
├── output/                      # Output files (models, previews)
├── templates/                   # Web interface templates
└── static/                      # Static files for web interface
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jhacksman/OpenSCAD-MCP-Server.git
   cd OpenSCAD-MCP-Server
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install OpenSCAD:
   - Ubuntu/Debian: `sudo apt-get install openscad`
   - macOS: `brew install openscad`
   - Windows: Download from [openscad.org](https://openscad.org/downloads.html)

## Usage

1. Start the server:
   ```
   python src/main.py
   ```

2. The server will start on http://localhost:8000

3. Use the MCP tools to interact with the server:

   - **create_3d_model**: Create a 3D model from a natural language description
     ```json
     {
       "description": "Create a cube with width 30mm, height 20mm, and depth 15mm"
     }
     ```

   - **modify_3d_model**: Modify an existing 3D model
     ```json
     {
       "model_id": "your-model-id",
       "modifications": "Make it taller and add rounded corners"
     }
     ```

   - **export_model**: Export a model to a specific format
     ```json
     {
       "model_id": "your-model-id",
       "format": "csg"  // or "amf", "3mf", "scad", etc.
     }
     ```

   - **discover_printers**: Discover 3D printers on the network
     ```json
     {}
     ```

   - **print_model**: Print a model on a connected printer
     ```json
     {
       "model_id": "your-model-id",
       "printer_id": "your-printer-id"
     }
     ```

## Supported Export Formats

The server supports exporting models in various formats that preserve parametric properties:

- **CSG**: OpenSCAD CSG format (preserves all parametric properties)
- **SCAD**: OpenSCAD source code (fully parametric)
- **AMF**: Additive Manufacturing File Format (preserves some metadata)
- **3MF**: 3D Manufacturing Format (modern replacement for STL with metadata)
- **DXF**: Drawing Exchange Format (for 2D designs)
- **SVG**: Scalable Vector Graphics (for 2D designs)

## Web Interface

The server provides a web interface for previewing models:

- Access the preview page at `/ui/preview/{model_id}`
- View multi-angle previews of the model
- Download the model in the exported format

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
