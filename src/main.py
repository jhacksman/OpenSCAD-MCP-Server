import os
import logging
import uuid
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from mcp import MCPServer, MCPTool, MCPToolCall, MCPToolCallResult

from src.nlp.parameter_extractor import ParameterExtractor
from src.models.code_generator import CodeGenerator
from src.openscad_wrapper.wrapper import OpenSCADWrapper
from src.utils.cad_exporter import CADExporter
from src.visualization.headless_renderer import HeadlessRenderer
from src.printer_discovery.printer_discovery import PrinterDiscovery, PrinterInterface

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
code_generator = CodeGenerator()
openscad_wrapper = OpenSCADWrapper("scad", "output")
cad_exporter = CADExporter()
headless_renderer = HeadlessRenderer()
printer_discovery = PrinterDiscovery()

# Store models in memory
models = {}
printers = {}

# Create MCP server
mcp_server = MCPServer()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Create a simple HTML template for previewing models
with open("templates/preview.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>OpenSCAD Model Preview</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
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
        format: Format to export to (csg, amf, 3mf, scad, etc.)
        
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
    if format.lower() not in supported_formats:
        raise ValueError(f"Format {format} not supported. Supported formats: {', '.join(supported_formats)}")
    
    # Export to the specified format
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
        raise ValueError(f"Failed to export model to {format}: {error}")
    
    # Update model information
    models[model_id]["model_file"] = model_file
    models[model_id]["format"] = format
    
    # Create response
    response = {
        "model_id": model_id,
        "format": format,
        "download_url": f"/download/{model_id}",
        "format_description": cad_exporter.get_format_description(format)
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
        "printers": discovered_printers,
        "count": len(discovered_printers)
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
        "name": printer_info["name"],
        "type": printer_info["type"],
        "status": "connected"
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
        Dictionary with print information
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
    
    # Check if model has been exported
    if not model_info.get("model_file"):
        raise ValueError(f"Model with ID {model_id} has not been exported")
    
    # Print model
    printer_interface = printer_info["interface"]
    job_id, error = printer_interface.print_model(model_info["model_file"])
    
    if not job_id:
        raise ValueError(f"Failed to print model: {error}")
    
    # Create response
    response = {
        "job_id": job_id,
        "model_id": model_id,
        "printer_id": printer_id,
        "status": "printing"
    }
    
    return response

@mcp_server.tool
def get_printer_status(printer_id: str) -> Dict[str, Any]:
    """
    Get the status of a printer.
    
    Args:
        printer_id: ID of the printer
        
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
        "name": printer_info["name"],
        "type": printer_info["type"],
        "status": status
    }
    
    return response

@mcp_server.tool
def cancel_print_job(printer_id: str, job_id: str) -> Dict[str, Any]:
    """
    Cancel a print job.
    
    Args:
        printer_id: ID of the printer
        job_id: ID of the print job
        
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
        "job_id": job_id,
        "printer_id": printer_id,
        "status": "cancelled"
    }
    
    return response

# Define FastAPI routes
@app.post("/mcp/tools/{tool_name}")
async def handle_tool_call(tool_name: str, request: Request):
    """Handle MCP tool calls."""
    try:
        # Parse request body
        body = await request.json()
        
        # Create tool call
        tool_call = MCPToolCall(
            tool_name=tool_name,
            tool_args=body
        )
        
        # Execute tool call
        result = await mcp_server.execute_tool_call(tool_call)
        
        # Return result
        return JSONResponse(content=result.tool_result)
    except Exception as e:
        logger.error(f"Error handling tool call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui/preview/{model_id}")
async def preview_model(model_id: str, request: Request):
    """Render model preview page."""
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Create preview URLs
    preview_urls = {}
    for view, path in model_info["previews"].items():
        preview_urls[view] = f"/preview/{model_id}/{view}"
    
    # Render template
    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "model_id": model_id,
            "parameters": model_info["parameters"],
            "previews": preview_urls
        }
    )

@app.get("/preview/{model_id}/{view}")
async def get_preview(model_id: str, view: str):
    """Get model preview image."""
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
async def download_model(model_id: str):
    """Download model file."""
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if model file exists
    if not model_info.get("model_file"):
        raise HTTPException(status_code=404, detail=f"Model file not found")
    
    # Return model file
    return FileResponse(
        model_info["model_file"],
        filename=f"{model_id}.{model_info['format']}"
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "OpenSCAD MCP Server",
        "version": "1.0.0",
        "description": "MCP server for OpenSCAD",
        "tools": list(mcp_server.tools.keys())
    }

# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
