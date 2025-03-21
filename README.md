# OpenSCAD MCP Server

A Model Context Protocol (MCP) server that enables users to generate 3D models from text descriptions or images, with a focus on creating parametric 3D models using multi-view reconstruction and OpenSCAD.

## Features

- **AI Image Generation**: Generate images from text descriptions using Google Gemini or Venice.ai APIs
- **Multi-View Image Generation**: Create multiple views of the same 3D object for reconstruction
- **Image Approval Workflow**: Review and approve/deny generated images before reconstruction
- **3D Reconstruction**: Convert approved multi-view images into 3D models using CUDA Multi-View Stereo
- **OpenSCAD Integration**: Generate parametric 3D models using OpenSCAD
- **Parametric Export**: Export models in formats that preserve parametric properties (CSG, AMF, 3MF, SCAD)
- **3D Printer Discovery**: Optional network printer discovery and direct printing

## Architecture

The server is built using the Python MCP SDK and follows a modular architecture:

```
openscad-mcp-server/
├── src/
│   ├── main.py                  # Main application
│   ├── ai/                      # AI integrations
│   │   ├── gemini_api.py        # Google Gemini API for image generation
│   │   └── venice_api.py        # Venice.ai API for image generation (optional)
│   ├── models/                  # 3D model generation
│   │   ├── cuda_mvs.py          # CUDA Multi-View Stereo integration
│   │   └── code_generator.py    # OpenSCAD code generation
│   ├── workflow/                # Workflow components
│   │   ├── image_approval.py    # Image approval mechanism
│   │   └── multi_view_to_model_pipeline.py  # Complete pipeline
│   ├── openscad_wrapper/        # OpenSCAD CLI wrapper
│   ├── visualization/           # Preview generation and web interface
│   ├── utils/                   # Utility functions
│   └── printer_discovery/       # 3D printer discovery
├── scad/                        # Generated OpenSCAD files
├── output/                      # Output files (models, previews)
│   ├── images/                  # Generated images
│   ├── multi_view/              # Multi-view images
│   ├── approved_images/         # Approved images for reconstruction
│   └── models/                  # Generated 3D models
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

5. Install CUDA Multi-View Stereo:
   ```
   git clone https://github.com/fixstars/cuda-multi-view-stereo.git
   cd cuda-multi-view-stereo
   mkdir build && cd build
   cmake ..
   make
   ```

6. Set up API keys:
   - Create a `.env` file in the root directory
   - Add your API keys:
     ```
     GEMINI_API_KEY=your-gemini-api-key
     VENICE_API_KEY=your-venice-api-key  # Optional
     ```

## Usage

1. Start the server:
   ```
   python src/main.py
   ```

2. The server will start on http://localhost:8000

3. Use the MCP tools to interact with the server:

   - **generate_image_gemini**: Generate an image using Google Gemini API
     ```json
     {
       "prompt": "A low-poly rabbit with black background",
       "model": "gemini-2.0-flash-exp-image-generation"
     }
     ```

   - **generate_multi_view_images**: Generate multiple views of the same 3D object
     ```json
     {
       "prompt": "A low-poly rabbit",
       "num_views": 4
     }
     ```

   - **create_3d_model_from_images**: Create a 3D model from approved multi-view images
     ```json
     {
       "image_ids": ["view_1", "view_2", "view_3", "view_4"],
       "output_name": "rabbit_model"
     }
     ```

   - **create_3d_model_from_text**: Complete pipeline from text to 3D model
     ```json
     {
       "prompt": "A low-poly rabbit",
       "num_views": 4
     }
     ```

   - **export_model**: Export a model to a specific format
     ```json
     {
       "model_id": "your-model-id",
       "format": "obj"  // or "stl", "ply", "scad", etc.
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

## Image Generation Options

The server supports multiple image generation options:

1. **Google Gemini API** (Default): Uses the Gemini 2.0 Flash Experimental model for high-quality image generation
   - Supports multi-view generation with consistent style
   - Requires a Google Gemini API key

2. **Venice.ai API** (Optional): Alternative image generation service
   - Supports various models including flux-dev and fluently-xl
   - Requires a Venice.ai API key

3. **User-Provided Images**: Skip image generation and use your own images
   - Upload images directly to the server
   - Useful for working with existing photographs or renders

## Multi-View Workflow

The server implements a multi-view workflow for 3D reconstruction:

1. **Image Generation**: Generate multiple views of the same 3D object
2. **Image Approval**: Review and approve/deny each generated image
3. **3D Reconstruction**: Convert approved images into a 3D model using CUDA MVS
4. **Model Refinement**: Optionally refine the model using OpenSCAD

## Supported Export Formats

The server supports exporting models in various formats:

- **OBJ**: Wavefront OBJ format (standard 3D model format)
- **STL**: Standard Triangle Language (for 3D printing)
- **PLY**: Polygon File Format (for point clouds and meshes)
- **SCAD**: OpenSCAD source code (for parametric models)
- **CSG**: OpenSCAD CSG format (preserves all parametric properties)
- **AMF**: Additive Manufacturing File Format (preserves some metadata)
- **3MF**: 3D Manufacturing Format (modern replacement for STL with metadata)

## Web Interface

The server provides a web interface for:

- Generating and approving multi-view images
- Previewing 3D models from different angles
- Downloading models in various formats

Access the interface at http://localhost:8000/ui/

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
