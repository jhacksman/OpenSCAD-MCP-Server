"""
Main module for OpenSCAD MCP Server with remote CUDA MVS processing.

This module adds remote CUDA MVS processing capabilities to the MCP server.
"""

import os
import logging
from typing import Dict, Any, List, Optional

# Import remote processing components
from src.remote.cuda_mvs_client import CUDAMVSClient
from src.remote.connection_manager import CUDAMVSConnectionManager
from src.config import REMOTE_CUDA_MVS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize remote processing components if enabled
remote_connection_manager = None
remote_jobs = {}

def initialize_remote_processing():
    """
    Initialize remote CUDA MVS processing components.
    
    Returns:
        CUDAMVSConnectionManager instance if enabled, None otherwise
    """
    global remote_connection_manager
    
    if REMOTE_CUDA_MVS["ENABLED"]:
        logger.info("Initializing remote CUDA MVS connection manager")
        remote_connection_manager = CUDAMVSConnectionManager(
            api_key=REMOTE_CUDA_MVS["API_KEY"],
            discovery_port=REMOTE_CUDA_MVS["DISCOVERY_PORT"],
            use_lan_discovery=REMOTE_CUDA_MVS["USE_LAN_DISCOVERY"],
            server_url=REMOTE_CUDA_MVS["SERVER_URL"] if REMOTE_CUDA_MVS["SERVER_URL"] else None
        )
        return remote_connection_manager
    
    return None

def discover_remote_servers():
    """
    Discover remote CUDA MVS servers on the network.
    
    Returns:
        List of discovered servers
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return []
    
    return remote_connection_manager.discover_servers()

def get_server_status(server_id: str):
    """
    Get the status of a remote CUDA MVS server.
    
    Args:
        server_id: ID of the server to get status for
        
    Returns:
        Server status information
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return None
    
    return remote_connection_manager.get_server_status(server_id)

def upload_images_to_server(server_id: str, image_paths: List[str], job_id: Optional[str] = None):
    """
    Upload images to a remote CUDA MVS server.
    
    Args:
        server_id: ID of the server to upload to
        image_paths: List of image paths to upload
        job_id: Optional job ID to use
        
    Returns:
        Job information
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return None
    
    return remote_connection_manager.upload_images(server_id, image_paths, job_id)

def process_images_remotely(server_id: str, job_id: str, params: Dict[str, Any] = None):
    """
    Process uploaded images on a remote CUDA MVS server.
    
    Args:
        server_id: ID of the server to process on
        job_id: Job ID of the uploaded images
        params: Optional processing parameters
        
    Returns:
        Job status information
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return None
    
    # Set default parameters if not provided
    if params is None:
        params = {
            "quality": REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
            "output_format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
        }
    
    # Start processing
    result = remote_connection_manager.process_job(server_id, job_id, params)
    
    # Store job information
    if result and "job_id" in result:
        remote_jobs[result["job_id"]] = {
            "server_id": server_id,
            "job_id": result["job_id"],
            "status": result.get("status", "processing"),
            "params": params
        }
    
    return result

def get_job_status(job_id: str):
    """
    Get the status of a remote processing job.
    
    Args:
        job_id: ID of the job to get status for
        
    Returns:
        Job status information
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return None
    
    # Check if job exists
    if job_id not in remote_jobs:
        logger.warning(f"Job with ID {job_id} not found")
        return None
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Get status from server
    status = remote_connection_manager.get_job_status(job_info["server_id"], job_id)
    
    # Update job information
    if status:
        job_info["status"] = status.get("status", job_info["status"])
        job_info["progress"] = status.get("progress", 0)
        job_info["message"] = status.get("message", "")
    
    return job_info

def download_model(job_id: str, output_dir: Optional[str] = None):
    """
    Download a processed model from a remote CUDA MVS server.
    
    Args:
        job_id: ID of the job to download model for
        output_dir: Optional directory to save the model to
        
    Returns:
        Model information
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return None
    
    # Check if job exists
    if job_id not in remote_jobs:
        logger.warning(f"Job with ID {job_id} not found")
        return None
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(REMOTE_CUDA_MVS["OUTPUT_DIR"], job_id)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model
    result = remote_connection_manager.download_model(job_info["server_id"], job_id, output_dir)
    
    # Update job information
    if result:
        job_info["model_path"] = result.get("model_path")
        job_info["point_cloud_path"] = result.get("point_cloud_path")
        job_info["completed"] = True
    
    return result

def cancel_job(job_id: str):
    """
    Cancel a remote processing job.
    
    Args:
        job_id: ID of the job to cancel
        
    Returns:
        Cancellation result
    """
    if not remote_connection_manager:
        logger.warning("Remote CUDA MVS processing is not enabled")
        return None
    
    # Check if job exists
    if job_id not in remote_jobs:
        logger.warning(f"Job with ID {job_id} not found")
        return None
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Cancel job
    result = remote_connection_manager.cancel_job(job_info["server_id"], job_id)
    
    # Update job information
    if result and result.get("cancelled", False):
        job_info["status"] = "cancelled"
        job_info["message"] = "Job cancelled by user"
    
    return result

# MCP tool functions for remote processing

def discover_remote_cuda_mvs_servers():
    """
    MCP tool function to discover remote CUDA MVS servers.
    
    Returns:
        Dictionary with discovered servers
    """
    servers = discover_remote_servers()
    
    return {
        "servers": servers,
        "count": len(servers)
    }

def get_remote_server_status(server_id: str):
    """
    MCP tool function to get the status of a remote CUDA MVS server.
    
    Args:
        server_id: ID of the server to get status for
        
    Returns:
        Dictionary with server status
    """
    status = get_server_status(server_id)
    
    if not status:
        raise ValueError(f"Failed to get status for server with ID {server_id}")
    
    return status

def process_images_with_remote_cuda_mvs(
    server_id: str,
    image_paths: List[str],
    quality: str = REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
    output_format: str = REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"],
    wait_for_completion: bool = REMOTE_CUDA_MVS["WAIT_FOR_COMPLETION"]
):
    """
    MCP tool function to process images with remote CUDA MVS.
    
    Args:
        server_id: ID of the server to process on
        image_paths: List of image paths to process
        quality: Reconstruction quality (low, normal, high)
        output_format: Output format (obj, ply)
        wait_for_completion: Whether to wait for job completion
        
    Returns:
        Dictionary with job information
    """
    # Upload images
    upload_result = upload_images_to_server(server_id, image_paths)
    
    if not upload_result or "job_id" not in upload_result:
        raise ValueError("Failed to upload images to server")
    
    job_id = upload_result["job_id"]
    
    # Process images
    process_result = process_images_remotely(
        server_id,
        job_id,
        {
            "quality": quality,
            "output_format": output_format
        }
    )
    
    if not process_result:
        raise ValueError(f"Failed to process images for job {job_id}")
    
    # Wait for completion if requested
    if wait_for_completion:
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
            download_result = download_model(job_id)
            
            if not download_result:
                raise ValueError(f"Failed to download model for job {job_id}")
            
            return {
                "job_id": job_id,
                "status": "completed",
                "model_path": download_result.get("model_path"),
                "point_cloud_path": download_result.get("point_cloud_path")
            }
        else:
            return {
                "job_id": job_id,
                "status": status["status"],
                "message": status.get("message", "")
            }
    
    # Return job information without waiting
    return {
        "job_id": job_id,
        "status": "processing",
        "server_id": server_id
    }

def get_remote_job_status(job_id: str):
    """
    MCP tool function to get the status of a remote processing job.
    
    Args:
        job_id: ID of the job to get status for
        
    Returns:
        Dictionary with job status
    """
    status = get_job_status(job_id)
    
    if not status:
        raise ValueError(f"Failed to get status for job with ID {job_id}")
    
    return status

def download_remote_model(job_id: str, output_dir: Optional[str] = None):
    """
    MCP tool function to download a processed model from a remote CUDA MVS server.
    
    Args:
        job_id: ID of the job to download model for
        output_dir: Optional directory to save the model to
        
    Returns:
        Dictionary with model information
    """
    result = download_model(job_id, output_dir)
    
    if not result:
        raise ValueError(f"Failed to download model for job with ID {job_id}")
    
    return result

def cancel_remote_job(job_id: str):
    """
    MCP tool function to cancel a remote processing job.
    
    Args:
        job_id: ID of the job to cancel
        
    Returns:
        Dictionary with cancellation result
    """
    result = cancel_job(job_id)
    
    if not result:
        raise ValueError(f"Failed to cancel job with ID {job_id}")
    
    return result
