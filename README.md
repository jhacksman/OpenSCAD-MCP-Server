# OpenSCAD MCP Server

A Model Context Protocol (MCP) server that enables users to generate 3D models from text descriptions or images, with a focus on creating parametric 3D models using multi-view reconstruction and OpenSCAD.

## Features

- **AI Image Generation**: Generate images from text descriptions using Google Gemini or Venice.ai APIs
- **Multi-View Image Generation**: Create multiple views of the same 3D object for reconstruction
- **Image Approval Workflow**: Review and approve/deny generated images before reconstruction
- **3D Reconstruction**: Convert approved multi-view images into 3D models using CUDA Multi-View Stereo
- **Remote Processing**: Process computationally intensive tasks on remote servers within your LAN
- **OpenSCAD Integration**: Generate parametric 3D models using OpenSCAD
- **Parametric Export**: Export models in formats that preserve parametric properties (CSG, AMF, 3MF, SCAD)
- **3D Printer Discovery**: Optional network printer discovery and direct printing

## Architecture

The server is built using the Python MCP SDK and follows a modular architecture:

```
openscad-mcp-server/
├── src/
│   ├── main.py                  # Main application
│   ├── main_remote.py           # Remote CUDA MVS server
│   ├── ai/                      # AI integrations
│   │   ├── gemini_api.py        # Google Gemini API for image generation
│   │   └── venice_api.py        # Venice.ai API for image generation (optional)
│   ├── models/                  # 3D model generation
│   │   ├── cuda_mvs.py          # CUDA Multi-View Stereo integration
│   │   └── code_generator.py    # OpenSCAD code generation
│   ├── workflow/                # Workflow components
│   │   ├── image_approval.py    # Image approval mechanism
│   │   └── multi_view_to_model_pipeline.py  # Complete pipeline
│   ├── remote/                  # Remote processing
│   │   ├── cuda_mvs_client.py   # Client for remote CUDA MVS processing
│   │   ├── cuda_mvs_server.py   # Server for remote CUDA MVS processing
│   │   ├── connection_manager.py # Remote connection management
│   │   └── error_handling.py    # Error handling for remote processing
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
     REMOTE_CUDA_MVS_API_KEY=your-remote-api-key  # For remote processing
     ```

## Remote Processing Setup

The server supports remote processing of computationally intensive tasks, particularly CUDA Multi-View Stereo reconstruction. This allows you to offload processing to more powerful machines within your LAN.

### Server Setup (on the machine with CUDA GPU)

1. Install CUDA Multi-View Stereo on the server machine:
   ```
   git clone https://github.com/fixstars/cuda-multi-view-stereo.git
   cd cuda-multi-view-stereo
   mkdir build && cd build
   cmake ..
   make
   ```

2. Start the remote CUDA MVS server:
   ```
   python src/main_remote.py
   ```

3. The server will automatically advertise itself on the local network using Zeroconf.

### Client Configuration

1. Configure remote processing in your `.env` file:
   ```
   REMOTE_CUDA_MVS_ENABLED=True
   REMOTE_CUDA_MVS_USE_LAN_DISCOVERY=True
   REMOTE_CUDA_MVS_API_KEY=your-shared-secret-key
   ```

2. Alternatively, you can specify a server URL directly:
   ```
   REMOTE_CUDA_MVS_ENABLED=True
   REMOTE_CUDA_MVS_USE_LAN_DISCOVERY=False
   REMOTE_CUDA_MVS_SERVER_URL=http://server-ip:8765
   REMOTE_CUDA_MVS_API_KEY=your-shared-secret-key
   ```

### Remote Processing Features

- **Automatic Server Discovery**: Find CUDA MVS servers on your local network
- **Job Management**: Upload images, track job status, and download results
- **Fault Tolerance**: Automatic retries, circuit breaker pattern, and error tracking
- **Authentication**: Secure API key authentication for all remote operations
- **Health Monitoring**: Continuous server health checks and status reporting

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

   - **discover_remote_cuda_mvs_servers**: Find CUDA MVS servers on your network
     ```json
     {
       "timeout": 5
     }
     ```

   - **get_remote_job_status**: Check the status of a remote processing job
     ```json
     {
       "server_id": "server-id",
       "job_id": "job-id"
     }
     ```

   - **download_remote_model_result**: Download a completed model from a remote server
     ```json
     {
       "server_id": "server-id",
       "job_id": "job-id",
       "output_name": "model-name"
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
   - Can be processed locally or on a remote server within your LAN
4. **Model Refinement**: Optionally refine the model using OpenSCAD

## Remote Processing Workflow

The remote processing workflow allows you to offload computationally intensive tasks to more powerful machines:

1. **Server Discovery**: Automatically discover CUDA MVS servers on your network
2. **Image Upload**: Upload approved multi-view images to the remote server
3. **Job Processing**: Process the images on the remote server using CUDA MVS
4. **Status Tracking**: Monitor the job status and progress
5. **Result Download**: Download the completed 3D model when processing is finished

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
