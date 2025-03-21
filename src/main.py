import os
import logging
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from mcp import MCPServer, MCPTool, MCPToolCall, MCPToolCallResult

# Import configuration
from src.config import *

# Import components
from src.nlp.parameter_extractor import ParameterExtractor
from src.models.code_generator import CodeGenerator
from src.openscad_wrapper.wrapper import OpenSCADWrapper
from src.utils.cad_exporter import CADExporter
from src.visualization.headless_renderer import HeadlessRenderer
from src.printer_discovery.printer_discovery import PrinterDiscovery, PrinterInterface
from src.ai.venice_api import VeniceImageGenerator
from src.ai.sam_segmentation import SAMSegmenter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="OpenSCAD MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("scad", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("output/models", exist_ok=True)
os.makedirs("output/preview", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize components
parameter_extractor = ParameterExtractor()
code_generator = CodeGenerator("scad", "output")
openscad_wrapper = OpenSCADWrapper("scad", "output")
cad_exporter = CADExporter()
headless_renderer = HeadlessRenderer()
printer_discovery = PrinterDiscovery()

# Initialize AI components
venice_generator = VeniceImageGenerator(VENICE_API_KEY, IMAGES_DIR)

# SAM2 segmenter will be initialized on first use to avoid loading the model unnecessarily
sam_segmenter = None

def get_sam_segmenter():
    """
    Get or initialize the SAM2 segmenter.
    
    Returns:
        SAMSegmenter instance
    """
    global sam_segmenter
    if sam_segmenter is None:
        logger.info("Initializing SAM2 segmenter")
        sam_segmenter = SAMSegmenter(
            model_type=SAM2_MODEL_TYPE,
            checkpoint_path=SAM2_CHECKPOINT_PATH,
            use_gpu=SAM2_USE_GPU,
            output_dir=MASKS_DIR
        )
    return sam_segmenter

# Store models in memory
models = {}
printers = {}

# Create MCP server
mcp_server = MCPServer()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Create model preview template
with open("templates/preview.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenSCAD Model Preview</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .preview-image {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
        }
        .preview-image img {
            max-width: 100%;
            height: auto;
        }
        .preview-image h3 {
            margin-top: 10px;
            margin-bottom: 5px;
            color: #555;
        }
        .parameters {
            margin-top: 20px;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #eee;
        }
        .parameters h2 {
            margin-top: 0;
            color: #333;
        }
        .parameters table {
            width: 100%;
            border-collapse: collapse;
        }
        .parameters table th, .parameters table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .parameters table th {
            background-color: #f2f2f2;
        }
        .actions {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        .actions button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .actions button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>OpenSCAD Model Preview: {{ model_id }}</h1>
        
        <div class="parameters">
            <h2>Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {% for key, value in parameters.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="preview-container">
            {% for view, image_path in previews.items() %}
            <div class="preview-image">
                <h3>{{ view|title }} View</h3>
                <img src="{{ image_path }}" alt="{{ view }} view">
            </div>
            {% endfor %}
        </div>
        
        <div class="actions">
            <button onclick="window.location.href='/download/{{ model_id }}'">Download Model</button>
        </div>
    </div>
</body>
</html>
    """)

# Define MCP tools
@mcp_server.tool
def create_3d_model(description: str) -> Dict[str, Any]:
    """
    Create a 3D model from a natural language description.
    
    Args:
        description: Natural language description of the 3D model
        
    Returns:
        Dictionary with model information
    """
    # Extract parameters from description
    model_type, parameters = parameter_extractor.extract_parameters(description)
    
    # Generate a unique model ID
    model_id = str(uuid.uuid4())
    
    # Generate OpenSCAD code
    scad_code = code_generator.generate_code(model_type, parameters)
    
    # Save the SCAD file
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    
    # Generate preview images
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, parameters)
    
    # Export to parametric format (CSG by default)
    success, model_file, error = cad_exporter.export_model(
        scad_file, 
        "csg",
        parameters,
        metadata={
            "description": description,
            "model_type": model_type,
        }
    )
    
    # Store model information
    models[model_id] = {
        "id": model_id,
        "type": model_type,
        "parameters": parameters,
        "description": description,
        "scad_file": scad_file,
        "model_file": model_file if success else None,
        "previews": previews,
        "format": "csg"
    }
    
    # Create response
    response = {
        "model_id": model_id,
        "model_type": model_type,
        "parameters": parameters,
        "preview_url": f"/ui/preview/{model_id}",
        "supported_formats": cad_exporter.get_supported_formats()
    }
    
    return response

@mcp_server.tool
def modify_3d_model(model_id: str, modifications: str) -> Dict[str, Any]:
    """
    Modify an existing 3D model.
    
    Args:
        model_id: ID of the model to modify
        modifications: Natural language description of the modifications
        
    Returns:
        Dictionary with updated model information
    """
    # Check if model exists
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    
    # Get existing model information
    model_info = models[model_id]
    
    # Extract parameters from modifications
    _, new_parameters = parameter_extractor.extract_parameters(
        modifications, 
        model_type=model_info["type"],
        existing_parameters=model_info["parameters"]
    )
    
    # Generate OpenSCAD code with updated parameters
    scad_code = code_generator.generate_code(model_info["type"], new_parameters)
    
    # Save the SCAD file
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    
    # Generate preview images
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, new_parameters)
    
    # Export to parametric format (same as original)
    success, model_file, error = cad_exporter.export_model(
        scad_file, 
        model_info["format"],
        new_parameters,
        metadata={
            "description": model_info["description"] + " | " + modifications,
            "model_type": model_info["type"],
        }
    )
    
    # Update model information
    models[model_id] = {
        "id": model_id,
        "type": model_info["type"],
        "parameters": new_parameters,
        "description": model_info["description"] + " | " + modifications,
        "scad_file": scad_file,
        "model_file": model_file if success else None,
        "previews": previews,
        "format": model_info["format"]
    }
    
    # Create response
    response = {
        "model_id": model_id,
        "model_type": model_info["type"],
        "parameters": new_parameters,
        "preview_url": f"/ui/preview/{model_id}",
        "supported_formats": cad_exporter.get_supported_formats()
    }
    
    return response

@mcp_server.tool
def export_model(model_id: str, format: str = "csg") -> Dict[str, Any]:
    """
    Export a 3D model to a specific format.
    
    Args:
        model_id: ID of the model to export
        format: Format to export to (csg, stl, obj, etc.)
        
    Returns:
        Dictionary with export information
    """
    # Check if model exists
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if format is supported
    supported_formats = cad_exporter.get_supported_formats()
    if format not in supported_formats:
        raise ValueError(f"Format {format} not supported. Supported formats: {', '.join(supported_formats)}")
    
    # Export model
    success, model_file, error = cad_exporter.export_model(
        model_info["scad_file"],
        format,
        model_info["parameters"],
        metadata={
            "description": model_info["description"],
            "model_type": model_info["type"],
        }
    )
    
    if not success:
        raise ValueError(f"Failed to export model: {error}")
    
    # Update model information
    models[model_id]["model_file"] = model_file
    models[model_id]["format"] = format
    
    # Create response
    response = {
        "model_id": model_id,
        "format": format,
        "model_file": model_file,
        "download_url": f"/download/{model_id}"
    }
    
    return response

@mcp_server.tool
def discover_printers() -> Dict[str, Any]:
    """
    Discover 3D printers on the network.
    
    Returns:
        Dictionary with discovered printers
    """
    # Discover printers
    discovered_printers = printer_discovery.discover_printers()
    
    # Store printers
    for printer in discovered_printers:
        printers[printer["id"]] = printer
    
    # Create response
    response = {
        "printers": discovered_printers
    }
    
    return response

@mcp_server.tool
def connect_to_printer(printer_id: str) -> Dict[str, Any]:
    """
    Connect to a 3D printer.
    
    Args:
        printer_id: ID of the printer to connect to
        
    Returns:
        Dictionary with connection information
    """
    # Check if printer exists
    if printer_id not in printers:
        raise ValueError(f"Printer with ID {printer_id} not found")
    
    # Get printer information
    printer_info = printers[printer_id]
    
    # Connect to printer
    printer_interface = PrinterInterface(printer_info)
    success, error = printer_interface.connect()
    
    if not success:
        raise ValueError(f"Failed to connect to printer: {error}")
    
    # Update printer information
    printers[printer_id]["connected"] = True
    printers[printer_id]["interface"] = printer_interface
    
    # Create response
    response = {
        "printer_id": printer_id,
        "connected": True,
        "printer_info": printer_info
    }
    
    return response

@mcp_server.tool
def print_model(model_id: str, printer_id: str) -> Dict[str, Any]:
    """
    Print a 3D model on a connected printer.
    
    Args:
        model_id: ID of the model to print
        printer_id: ID of the printer to print on
        
    Returns:
        Dictionary with print job information
    """
    # Check if model exists
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    
    # Check if printer exists
    if printer_id not in printers:
        raise ValueError(f"Printer with ID {printer_id} not found")
    
    # Check if printer is connected
    if not printers[printer_id].get("connected", False):
        raise ValueError(f"Printer with ID {printer_id} is not connected")
    
    # Get model and printer information
    model_info = models[model_id]
    printer_info = printers[printer_id]
    
    # Check if model has been exported to a printable format
    if not model_info.get("model_file"):
        raise ValueError(f"Model with ID {model_id} has not been exported")
    
    # Print model
    printer_interface = printer_info["interface"]
    job_id, error = printer_interface.print_model(model_info["model_file"])
    
    if not job_id:
        raise ValueError(f"Failed to print model: {error}")
    
    # Create response
    response = {
        "model_id": model_id,
        "printer_id": printer_id,
        "job_id": job_id,
        "status": "printing"
    }
    
    return response

@mcp_server.tool
def get_printer_status(printer_id: str) -> Dict[str, Any]:
    """
    Get the status of a printer.
    
    Args:
        printer_id: ID of the printer to get status for
        
    Returns:
        Dictionary with printer status
    """
    # Check if printer exists
    if printer_id not in printers:
        raise ValueError(f"Printer with ID {printer_id} not found")
    
    # Check if printer is connected
    if not printers[printer_id].get("connected", False):
        raise ValueError(f"Printer with ID {printer_id} is not connected")
    
    # Get printer information
    printer_info = printers[printer_id]
    
    # Get printer status
    printer_interface = printer_info["interface"]
    status = printer_interface.get_status()
    
    # Create response
    response = {
        "printer_id": printer_id,
        "status": status
    }
    
    return response

@mcp_server.tool
def cancel_print_job(printer_id: str, job_id: str) -> Dict[str, Any]:
    """
    Cancel a print job.
    
    Args:
        printer_id: ID of the printer
        job_id: ID of the print job to cancel
        
    Returns:
        Dictionary with cancellation information
    """
    # Check if printer exists
    if printer_id not in printers:
        raise ValueError(f"Printer with ID {printer_id} not found")
    
    # Check if printer is connected
    if not printers[printer_id].get("connected", False):
        raise ValueError(f"Printer with ID {printer_id} is not connected")
    
    # Get printer information
    printer_info = printers[printer_id]
    
    # Cancel print job
    printer_interface = printer_info["interface"]
    success, error = printer_interface.cancel_job(job_id)
    
    if not success:
        raise ValueError(f"Failed to cancel print job: {error}")
    
    # Create response
    response = {
        "printer_id": printer_id,
        "job_id": job_id,
        "status": "cancelled"
    }
    
    return response

# Add Venice.ai image generation tool
@mcp_server.tool
def generate_image(prompt: str, model: str = "fluently-xl") -> Dict[str, Any]:
    """
    Generate an image using Venice.ai's image generation models.
    
    Args:
        prompt: Text description for image generation
        model: Model to use (default: fluently-xl). Options include:
            - "fluently-xl" (fastest, 2.30s): Quick generation with good quality
            - "flux-dev" (high quality): Detailed, premium image quality
            - "flux-dev-uncensored": Uncensored version of flux-dev model
            - "stable-diffusion-3.5": Standard stable diffusion model
            - "pony-realism": Specialized for realistic outputs
            - "lustify-sdxl": Artistic stylization model
            
            You can also use natural language like:
            - "fastest model", "quick generation", "efficient"
            - "high quality", "detailed", "premium quality"
            - "realistic", "photorealistic"
            - "artistic", "stylized", "creative"
        
    Returns:
        Dictionary with image information
    """
    # Generate a unique image ID
    image_id = str(uuid.uuid4())
    
    # Generate image
    result = venice_generator.generate_image(prompt, model)
    
    # Create response
    response = {
        "image_id": image_id,
        "prompt": prompt,
        "model": model,
        "image_path": result.get("local_path"),
        "image_url": result.get("image_url")
    }
    
    return response

# Add SAM2 segmentation tool
@mcp_server.tool
def segment_image(image_path: str, points: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """
    Segment objects in an image using SAM2 (Segment Anything Model 2).
    
    Args:
        image_path: Path to the input image
        points: Optional list of (x, y) points to guide segmentation
               If not provided, automatic segmentation will be used
        
    Returns:
        Dictionary with segmentation masks and metadata
    """
    # Get or initialize SAM2 segmenter
    sam_segmenter = get_sam_segmenter()
    
    # Generate a unique segmentation ID
    segmentation_id = str(uuid.uuid4())
    
    try:
        # Perform segmentation
        if points:
            result = sam_segmenter.segment_image(image_path, points)
        else:
            result = sam_segmenter.segment_with_auto_points(image_path)
        
        # Create response
        response = {
            "segmentation_id": segmentation_id,
            "image_path": image_path,
            "mask_paths": result.get("mask_paths", []),
            "num_masks": result.get("num_masks", 0),
            "points_used": points if points else result.get("points", [])
        }
        
        return response
    except Exception as e:
        logger.error(f"Error segmenting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error segmenting image: {str(e)}")


# FastAPI routes
@app.post("/tool_call")
async def handle_tool_call(request: Request) -> JSONResponse:
    """
    Handle a tool call from a client.
    
    Args:
        request: FastAPI request object
        
    Returns:
        JSON response with tool call result
    """
    # Parse request
    data = await request.json()
    
    # Check if tool name is provided
    if "tool_name" not in data:
        raise HTTPException(status_code=400, detail="Tool name is required")
    
    # Check if tool exists
    tool_name = data["tool_name"]
    if tool_name not in mcp_server.tools:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    # Get tool parameters
    tool_params = data.get("tool_params", {})
    
    # Call tool
    try:
        result = mcp_server.tools[tool_name](**tool_params)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui/preview/{model_id}")
async def preview_model(request: Request, model_id: str) -> Response:
    """
    Render a preview page for a model.
    
    Args:
        request: FastAPI request object
        model_id: ID of the model to preview
        
    Returns:
        HTML response with model preview
    """
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Render template
    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "model_id": model_id,
            "parameters": model_info["parameters"],
            "previews": model_info["previews"]
        }
    )

@app.get("/preview/{view}/{model_id}")
async def get_preview(view: str, model_id: str) -> FileResponse:
    """
    Get a preview image for a model.
    
    Args:
        view: View to get preview for
        model_id: ID of the model
        
    Returns:
        Image file response
    """
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if preview exists
    if view not in model_info["previews"]:
        raise HTTPException(status_code=404, detail=f"Preview for view {view} not found")
    
    # Return preview image
    return FileResponse(model_info["previews"][view])

@app.get("/download/{model_id}")
async def download_model(model_id: str) -> FileResponse:
    """
    Download a model file.
    
    Args:
        model_id: ID of the model to download
        
    Returns:
        Model file response
    """
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if model file exists
    if not model_info.get("model_file"):
        raise HTTPException(status_code=404, detail=f"Model file for model with ID {model_id} not found")
    
    # Return model file
    return FileResponse(
        model_info["model_file"],
        filename=f"{model_id}.{model_info['format']}"
    )

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint.
    
    Returns:
        Dictionary with server information
    """
    return {
        "name": "OpenSCAD MCP Server",
        "version": "1.0.0",
        "description": "MCP server for OpenSCAD",
        "tools": list(mcp_server.tools.keys())
    }

# Run server
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
