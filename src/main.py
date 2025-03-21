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
gemini_generator = GeminiImageGenerator(GEMINI_API_KEY, IMAGES_DIR)
cuda_mvs = CUDAMultiViewStereo(CUDA_MVS_PATH, MODELS_DIR, use_gpu=CUDA_MVS_USE_GPU)
image_approval = ImageApprovalTool(APPROVED_IMAGES_DIR)

# Initialize remote processing components if enabled
remote_connection_manager = None
if REMOTE_CUDA_MVS["ENABLED"]:
    logger.info("Initializing remote CUDA MVS connection manager")
    remote_connection_manager = CUDAMVSConnectionManager(
        api_key=REMOTE_CUDA_MVS["API_KEY"],
        discovery_port=REMOTE_CUDA_MVS["DISCOVERY_PORT"],
        use_lan_discovery=REMOTE_CUDA_MVS["USE_LAN_DISCOVERY"],
        server_url=REMOTE_CUDA_MVS["SERVER_URL"] if REMOTE_CUDA_MVS["SERVER_URL"] else None
    )

# Initialize workflow pipeline
multi_view_pipeline = MultiViewToModelPipeline(
    gemini_generator=gemini_generator,
    cuda_mvs=cuda_mvs,
    approval_tool=image_approval,
    output_dir=OUTPUT_DIR
)

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
approved_images = {}
remote_jobs = {}

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


# Add Google Gemini image generation tool
@mcp_server.tool
def generate_image_gemini(prompt: str, model: str = GEMINI_MODEL) -> Dict[str, Any]:
    """
    Generate an image using Google Gemini's image generation models.
    
    Args:
        prompt: Text description for image generation
        model: Model to use (default: gemini-2.0-flash-exp-image-generation)
        
    Returns:
        Dictionary with image information
    """
    # Generate a unique image ID
    image_id = str(uuid.uuid4())
    
    # Generate image
    result = gemini_generator.generate_image(prompt, model)
    
    # Create response
    response = {
        "image_id": image_id,
        "prompt": prompt,
        "model": model,
        "image_path": result.get("local_path"),
        "image_url": f"/images/{os.path.basename(result.get('local_path', ''))}"
    }
    
    return response


# Add multi-view image generation tool
@mcp_server.tool
def generate_multi_view_images(prompt: str, num_views: int = 4) -> Dict[str, Any]:
    """
    Generate multiple views of the same 3D object using Google Gemini.
    
    Args:
        prompt: Text description of the 3D object
        num_views: Number of views to generate (default: 4)
        
    Returns:
        Dictionary with multi-view image information
    """
    # Validate number of views
    if num_views < MULTI_VIEW_PIPELINE["MIN_NUM_VIEWS"]:
        raise ValueError(f"Number of views must be at least {MULTI_VIEW_PIPELINE['MIN_NUM_VIEWS']}")
    
    if num_views > MULTI_VIEW_PIPELINE["MAX_NUM_VIEWS"]:
        raise ValueError(f"Number of views cannot exceed {MULTI_VIEW_PIPELINE['MAX_NUM_VIEWS']}")
    
    # Generate a unique multi-view ID
    multi_view_id = str(uuid.uuid4())
    
    # Generate multi-view images
    results = gemini_generator.generate_multiple_views(prompt, num_views)
    
    # Create response
    response = {
        "multi_view_id": multi_view_id,
        "prompt": prompt,
        "num_views": num_views,
        "views": [
            {
                "view_id": result.get("view_id", f"view_{i+1}"),
                "view_index": result.get("view_index", i+1),
                "view_direction": result.get("view_direction", ""),
                "image_path": result.get("local_path"),
                "image_url": f"/images/{os.path.basename(result.get('local_path', ''))}"
            }
            for i, result in enumerate(results)
        ],
        "approval_required": IMAGE_APPROVAL["ENABLED"] and not IMAGE_APPROVAL["AUTO_APPROVE"]
    }
    
    # Store multi-view information for approval
    if IMAGE_APPROVAL["ENABLED"]:
        approved_images[multi_view_id] = {
            "multi_view_id": multi_view_id,
            "prompt": prompt,
            "num_views": num_views,
            "views": response["views"],
            "approved_views": [] if not IMAGE_APPROVAL["AUTO_APPROVE"] else [view["view_id"] for view in response["views"]],
            "rejected_views": [],
            "approval_complete": IMAGE_APPROVAL["AUTO_APPROVE"]
        }
    
    return response


# Add image approval tool
@mcp_server.tool
def approve_image(multi_view_id: str, view_id: str) -> Dict[str, Any]:
    """
    Approve an image for 3D model generation.
    
    Args:
        multi_view_id: ID of the multi-view set
        view_id: ID of the view to approve
        
    Returns:
        Dictionary with approval information
    """
    # Check if multi-view ID exists
    if multi_view_id not in approved_images:
        raise ValueError(f"Multi-view set with ID {multi_view_id} not found")
    
    # Get multi-view information
    multi_view_info = approved_images[multi_view_id]
    
    # Check if view ID exists
    view_exists = False
    for view in multi_view_info["views"]:
        if view["view_id"] == view_id:
            view_exists = True
            break
    
    if not view_exists:
        raise ValueError(f"View with ID {view_id} not found in multi-view set {multi_view_id}")
    
    # Check if view is already approved
    if view_id in multi_view_info["approved_views"]:
        return {
            "multi_view_id": multi_view_id,
            "view_id": view_id,
            "status": "already_approved",
            "approved_views": multi_view_info["approved_views"],
            "rejected_views": multi_view_info["rejected_views"],
            "approval_complete": multi_view_info["approval_complete"]
        }
    
    # Remove from rejected views if present
    if view_id in multi_view_info["rejected_views"]:
        multi_view_info["rejected_views"].remove(view_id)
    
    # Add to approved views
    multi_view_info["approved_views"].append(view_id)
    
    # Check if approval is complete
    if len(multi_view_info["approved_views"]) >= IMAGE_APPROVAL["MIN_APPROVED_IMAGES"]:
        multi_view_info["approval_complete"] = True
    
    # Create response
    response = {
        "multi_view_id": multi_view_id,
        "view_id": view_id,
        "status": "approved",
        "approved_views": multi_view_info["approved_views"],
        "rejected_views": multi_view_info["rejected_views"],
        "approval_complete": multi_view_info["approval_complete"]
    }
    
    return response


# Add image rejection tool
@mcp_server.tool
def reject_image(multi_view_id: str, view_id: str) -> Dict[str, Any]:
    """
    Reject an image for 3D model generation.
    
    Args:
        multi_view_id: ID of the multi-view set
        view_id: ID of the view to reject
        
    Returns:
        Dictionary with rejection information
    """
    # Check if multi-view ID exists
    if multi_view_id not in approved_images:
        raise ValueError(f"Multi-view set with ID {multi_view_id} not found")
    
    # Get multi-view information
    multi_view_info = approved_images[multi_view_id]
    
    # Check if view ID exists
    view_exists = False
    for view in multi_view_info["views"]:
        if view["view_id"] == view_id:
            view_exists = True
            break
    
    if not view_exists:
        raise ValueError(f"View with ID {view_id} not found in multi-view set {multi_view_id}")
    
    # Check if view is already rejected
    if view_id in multi_view_info["rejected_views"]:
        return {
            "multi_view_id": multi_view_id,
            "view_id": view_id,
            "status": "already_rejected",
            "approved_views": multi_view_info["approved_views"],
            "rejected_views": multi_view_info["rejected_views"],
            "approval_complete": multi_view_info["approval_complete"]
        }
    
    # Remove from approved views if present
    if view_id in multi_view_info["approved_views"]:
        multi_view_info["approved_views"].remove(view_id)
    
    # Add to rejected views
    multi_view_info["rejected_views"].append(view_id)
    
    # Check if approval is complete
    if len(multi_view_info["approved_views"]) >= IMAGE_APPROVAL["MIN_APPROVED_IMAGES"]:
        multi_view_info["approval_complete"] = True
    else:
        multi_view_info["approval_complete"] = False
    
    # Create response
    response = {
        "multi_view_id": multi_view_id,
        "view_id": view_id,
        "status": "rejected",
        "approved_views": multi_view_info["approved_views"],
        "rejected_views": multi_view_info["rejected_views"],
        "approval_complete": multi_view_info["approval_complete"]
    }
    
    return response


# Add 3D model generation from approved images tool
@mcp_server.tool
def create_3d_model_from_images(multi_view_id: str, output_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a 3D model from approved multi-view images.
    
    Args:
        multi_view_id: ID of the multi-view set
        output_name: Optional name for the output model
        
    Returns:
        Dictionary with model information
    """
    # Check if multi-view ID exists
    if multi_view_id not in approved_images:
        raise ValueError(f"Multi-view set with ID {multi_view_id} not found")
    
    # Get multi-view information
    multi_view_info = approved_images[multi_view_id]
    
    # Check if approval is complete
    if not multi_view_info["approval_complete"]:
        raise ValueError(f"Approval for multi-view set {multi_view_id} is not complete")
    
    # Check if there are enough approved images
    if len(multi_view_info["approved_views"]) < IMAGE_APPROVAL["MIN_APPROVED_IMAGES"]:
        raise ValueError(f"Not enough approved images. Need at least {IMAGE_APPROVAL['MIN_APPROVED_IMAGES']}, but only have {len(multi_view_info['approved_views'])}")
    
    # Get approved image paths
    approved_image_paths = []
    for view in multi_view_info["views"]:
        if view["view_id"] in multi_view_info["approved_views"]:
            approved_image_paths.append(view["image_path"])
    
    # Generate a unique model ID
    model_id = str(uuid.uuid4())
    
    # Set output name if not provided
    if not output_name:
        output_name = f"model_{model_id[:8]}"
    
    # Create 3D model
    if REMOTE_CUDA_MVS["ENABLED"] and remote_connection_manager:
        # Use remote CUDA MVS processing
        servers = discover_remote_servers()
        
        if not servers:
            raise ValueError("No remote CUDA MVS servers found")
        
        # Use the first available server
        server_id = servers[0]["id"]
        
        # Upload images
        upload_result = upload_images_to_server(server_id, approved_image_paths)
        
        if not upload_result or "job_id" not in upload_result:
            raise ValueError("Failed to upload images to remote server")
        
        job_id = upload_result["job_id"]
        
        # Process images
        process_result = process_images_remotely(
            server_id,
            job_id,
            {
                "quality": REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
                "output_format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
            }
        )
        
        if not process_result:
            raise ValueError(f"Failed to process images for job {job_id}")
        
        # Wait for completion if requested
        if REMOTE_CUDA_MVS["WAIT_FOR_COMPLETION"]:
            import time
            
            while True:
                status = get_job_status(job_id)
                
                if not status:
                    raise ValueError(f"Failed to get status for job {job_id}")
                
                if status["status"] in ["completed", "failed", "cancelled"]:
                    break
                
                time.sleep(REMOTE_CUDA_MVS["POLL_INTERVAL"])
            
            if status["status"] == "completed":
                # Download model
                download_result = download_remote_model(job_id)
                
                if not download_result:
                    raise ValueError(f"Failed to download model for job {job_id}")
                
                # Store model information
                models[model_id] = {
                    "id": model_id,
                    "type": "cuda_mvs_remote",
                    "parameters": {
                        "multi_view_id": multi_view_id,
                        "prompt": multi_view_info["prompt"],
                        "num_views": len(approved_image_paths),
                        "quality": REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
                        "output_format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
                    },
                    "description": f"3D model generated from {len(approved_image_paths)} views of '{multi_view_info['prompt']}'",
                    "model_file": download_result.get("model_path"),
                    "point_cloud_file": download_result.get("point_cloud_path"),
                    "previews": {},  # Will be generated later
                    "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"],
                    "remote_job_id": job_id
                }
                
                # Create response
                response = {
                    "model_id": model_id,
                    "multi_view_id": multi_view_id,
                    "status": "completed",
                    "model_path": download_result.get("model_path"),
                    "point_cloud_path": download_result.get("point_cloud_path"),
                    "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
                }
            else:
                # Store job information
                remote_jobs[job_id] = {
                    "model_id": model_id,
                    "multi_view_id": multi_view_id,
                    "server_id": server_id,
                    "job_id": job_id,
                    "status": status["status"],
                    "message": status.get("message", "")
                }
                
                # Create response
                response = {
                    "model_id": model_id,
                    "multi_view_id": multi_view_id,
                    "status": status["status"],
                    "message": status.get("message", ""),
                    "job_id": job_id
                }
        else:
            # Store job information
            remote_jobs[job_id] = {
                "model_id": model_id,
                "multi_view_id": multi_view_id,
                "server_id": server_id,
                "job_id": job_id,
                "status": "processing"
            }
            
            # Create response
            response = {
                "model_id": model_id,
                "multi_view_id": multi_view_id,
                "status": "processing",
                "job_id": job_id,
                "server_id": server_id
            }
    else:
        # Use local CUDA MVS processing
        result = cuda_mvs.process_images(
            approved_image_paths,
            output_name=output_name,
            quality=REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
            output_format=REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
        )
        
        # Store model information
        models[model_id] = {
            "id": model_id,
            "type": "cuda_mvs_local",
            "parameters": {
                "multi_view_id": multi_view_id,
                "prompt": multi_view_info["prompt"],
                "num_views": len(approved_image_paths),
                "quality": REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
                "output_format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
            },
            "description": f"3D model generated from {len(approved_image_paths)} views of '{multi_view_info['prompt']}'",
            "model_file": result.get("model_path"),
            "point_cloud_file": result.get("point_cloud_path"),
            "previews": {},  # Will be generated later
            "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
        }
        
        # Create response
        response = {
            "model_id": model_id,
            "multi_view_id": multi_view_id,
            "status": "completed",
            "model_path": result.get("model_path"),
            "point_cloud_path": result.get("point_cloud_path"),
            "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
        }
    
    return response


# Add complete pipeline tool (text to 3D model)
@mcp_server.tool
def create_3d_model_from_text(prompt: str, num_views: int = 4, wait_for_completion: bool = True) -> Dict[str, Any]:
    """
    Create a 3D model from a text description using the complete pipeline.
    
    Args:
        prompt: Text description of the 3D object
        num_views: Number of views to generate (default: 4)
        wait_for_completion: Whether to wait for remote processing to complete
        
    Returns:
        Dictionary with model information
    """
    # Generate multi-view images
    multi_view_result = generate_multi_view_images(prompt, num_views)
    
    multi_view_id = multi_view_result["multi_view_id"]
    
    # Auto-approve all images if enabled
    if IMAGE_APPROVAL["AUTO_APPROVE"]:
        for view in multi_view_result["views"]:
            approve_image(multi_view_id, view["view_id"])
    else:
        # Return multi-view result for manual approval
        return {
            "status": "awaiting_approval",
            "message": "Please approve or reject each image before proceeding",
            "multi_view_id": multi_view_id,
            "views": multi_view_result["views"]
        }
    
    # Create 3D model from approved images
    model_result = create_3d_model_from_images(multi_view_id)
    
    # If remote processing is not waiting for completion, return job information
    if not wait_for_completion and model_result.get("status") == "processing":
        return model_result
    
    # Return model information
    return model_result


# Add remote CUDA MVS server discovery tool
@mcp_server.tool
def discover_remote_cuda_mvs_servers() -> Dict[str, Any]:
    """
    Discover remote CUDA MVS servers on the network.
    
    Returns:
        Dictionary with discovered servers
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    servers = discover_remote_servers()
    
    return {
        "servers": servers,
        "count": len(servers)
    }


# Add remote job status tool
@mcp_server.tool
def get_remote_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a remote CUDA MVS processing job.
    
    Args:
        job_id: ID of the job to get status for
        
    Returns:
        Dictionary with job status
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    # Check if job exists
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Get status from server
    status = get_job_status(job_id)
    
    if not status:
        raise ValueError(f"Failed to get status for job with ID {job_id}")
    
    # Update job information
    job_info["status"] = status.get("status", job_info["status"])
    job_info["progress"] = status.get("progress", 0)
    job_info["message"] = status.get("message", "")
    
    return job_info


# Add remote model download tool
@mcp_server.tool
def download_remote_model_result(job_id: str) -> Dict[str, Any]:
    """
    Download a processed model from a remote CUDA MVS server.
    
    Args:
        job_id: ID of the job to download model for
        
    Returns:
        Dictionary with model information
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    # Check if job exists
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Check if job is completed
    if job_info["status"] != "completed":
        raise ValueError(f"Job with ID {job_id} is not completed (status: {job_info['status']})")
    
    # Download model
    result = download_remote_model(job_id)
    
    if not result:
        raise ValueError(f"Failed to download model for job with ID {job_id}")
    
    # Update job information
    job_info["model_path"] = result.get("model_path")
    job_info["point_cloud_path"] = result.get("point_cloud_path")
    job_info["downloaded"] = True
    
    # Update model information if available
    if "model_id" in job_info and job_info["model_id"] in models:
        model_id = job_info["model_id"]
        models[model_id]["model_file"] = result.get("model_path")
        models[model_id]["point_cloud_file"] = result.get("point_cloud_path")
    
    return {
        "job_id": job_id,
        "model_path": result.get("model_path"),
        "point_cloud_path": result.get("point_cloud_path"),
        "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
    }


# Add remote job cancellation tool
@mcp_server.tool
def cancel_remote_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a remote CUDA MVS processing job.
    
    Args:
        job_id: ID of the job to cancel
        
    Returns:
        Dictionary with cancellation result
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    # Check if job exists
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Cancel job
    result = cancel_job(job_id)
    
    if not result:
        raise ValueError(f"Failed to cancel job with ID {job_id}")
    
    # Update job information
    if result.get("cancelled", False):
        job_info["status"] = "cancelled"
        job_info["message"] = "Job cancelled by user"
    
    return {
        "job_id": job_id,
        "cancelled": result.get("cancelled", False),
        "status": job_info["status"],
        "message": job_info.get("message", "")
    }


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
