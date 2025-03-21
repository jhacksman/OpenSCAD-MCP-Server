"""
Server for remote CUDA Multi-View Stereo processing.

This module provides a server implementation that can be deployed on a machine
with CUDA capabilities to process multi-view images into 3D models remotely.
"""

import os
import sys
import json
import uuid
import logging
import argparse
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import shutil
import time

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
from pydantic import BaseModel, Field
from zeroconf import ServiceInfo, Zeroconf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8765
DEFAULT_HOST = "0.0.0.0"
DEFAULT_CUDA_MVS_PATH = "/opt/cuda-multi-view-stereo"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MAX_JOBS = 5
DEFAULT_MAX_IMAGES_PER_JOB = 50
DEFAULT_JOB_TIMEOUT = 3600  # 1 hour

# Models
class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str = "created"  # created, uploading, processing, completed, failed
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    num_images: int = 0
    images_uploaded: int = 0
    progress: float = 0.0
    error_message: Optional[str] = None
    model_id: Optional[str] = None
    output_dir: Optional[str] = None
    point_cloud_file: Optional[str] = None
    obj_file: Optional[str] = None
    processing_time: Optional[float] = None

class ServerConfig(BaseModel):
    """Server configuration model."""
    cuda_mvs_path: str = DEFAULT_CUDA_MVS_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_jobs: int = DEFAULT_MAX_JOBS
    max_images_per_job: int = DEFAULT_MAX_IMAGES_PER_JOB
    job_timeout: int = DEFAULT_JOB_TIMEOUT
    api_key: Optional[str] = None
    server_name: str = "CUDA MVS Server"
    advertise_service: bool = True
    gpu_info: Optional[str] = None

class JobRequest(BaseModel):
    """Job creation request model."""
    num_images: int

class ProcessRequest(BaseModel):
    """Process job request model."""
    reconstruction_quality: str = "normal"  # low, normal, high
    output_formats: List[str] = ["obj", "ply"]

class ServerInfo(BaseModel):
    """Server information model."""
    server_id: str
    name: str
    version: str = "1.0.0"
    status: str = "running"
    capabilities: Dict[str, Any]
    jobs: Dict[str, Any]
    uptime: float

# Server implementation
class CUDAMVSServer:
    """
    Server for remote CUDA Multi-View Stereo processing.
    
    This server:
    1. Accepts image uploads
    2. Processes images using CUDA MVS
    3. Provides 3D model downloads
    4. Advertises itself on the local network
    """
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the CUDA MVS server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.app = FastAPI(title="CUDA MVS Server", description="Remote CUDA Multi-View Stereo processing server")
        self.jobs: Dict[str, JobStatus] = {}
        self.server_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.zeroconf = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Detect GPU info
        self.detect_gpu_info()
        
        # Register routes
        self.register_routes()
        
        # Advertise service if enabled
        if config.advertise_service:
            self.advertise_service()
    
    def detect_gpu_info(self):
        """Detect GPU information."""
        if not self.config.gpu_info:
            try:
                # Try to get GPU info using nvidia-smi
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.config.gpu_info = result.stdout.strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                # If nvidia-smi fails, try lspci
                try:
                    result = subprocess.run(
                        ["lspci", "-v", "|", "grep", "-i", "vga"],
                        capture_output=True,
                        text=True,
                        shell=True
                    )
                    self.config.gpu_info = result.stdout.strip()
                except subprocess.SubprocessError:
                    self.config.gpu_info = "Unknown GPU"
    
    def register_routes(self):
        """Register API routes."""
        
        # Authentication dependency
        api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
        
        async def verify_api_key(api_key: str = Depends(api_key_header)):
            if self.config.api_key:
                if not api_key:
                    raise HTTPException(status_code=401, detail="API key required")
                
                # Check if the API key is in the format "Bearer <key>"
                if api_key.startswith("Bearer "):
                    api_key = api_key[7:]
                
                if api_key != self.config.api_key:
                    raise HTTPException(status_code=401, detail="Invalid API key")
            return True
        
        # Status endpoint
        @self.app.get("/api/status")
        async def get_status():
            return self.get_server_info()
        
        # Job management endpoints
        @self.app.post("/api/jobs", status_code=201, dependencies=[Depends(verify_api_key)])
        async def create_job(job_request: JobRequest):
            return self.create_job(job_request.num_images)
        
        @self.app.get("/api/jobs", dependencies=[Depends(verify_api_key)])
        async def list_jobs():
            return {"jobs": self.jobs}
        
        @self.app.get("/api/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
        async def get_job(job_id: str):
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            return self.jobs[job_id]
        
        @self.app.delete("/api/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
        async def cancel_job(job_id: str):
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Cancel the job
            job = self.jobs[job_id]
            if job.status in ["created", "uploading", "processing"]:
                job.status = "cancelled"
                job.updated_at = time.time()
                
                # Clean up job directory
                job_dir = os.path.join(self.config.output_dir, job_id)
                if os.path.exists(job_dir):
                    shutil.rmtree(job_dir)
                
                return {"status": "success", "message": "Job cancelled"}
            else:
                return {"status": "error", "message": f"Cannot cancel job in {job.status} state"}
        
        # Image upload endpoint
        @self.app.post("/api/jobs/{job_id}/images", dependencies=[Depends(verify_api_key)])
        async def upload_image(
            job_id: str,
            file: UploadFile = File(...),
            metadata: str = Form(None)
        ):
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            
            # Check if job is in a valid state for uploads
            if job.status not in ["created", "uploading"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot upload images to job in {job.status} state"
                )
            
            # Check if we've reached the maximum number of images
            if job.images_uploaded >= job.num_images:
                raise HTTPException(
                    status_code=400,
                    detail=f"Maximum number of images ({job.num_images}) already uploaded"
                )
            
            # Update job status
            job.status = "uploading"
            job.updated_at = time.time()
            
            # Create job directory if it doesn't exist
            job_dir = os.path.join(self.config.output_dir, job_id, "images")
            os.makedirs(job_dir, exist_ok=True)
            
            # Parse metadata
            image_metadata = {}
            if metadata:
                try:
                    image_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid metadata format: {metadata}")
            
            # Save the file
            image_index = job.images_uploaded
            file_extension = os.path.splitext(file.filename)[1]
            image_path = os.path.join(job_dir, f"image_{image_index:04d}{file_extension}")
            
            with open(image_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Update job status
            job.images_uploaded += 1
            job.updated_at = time.time()
            
            return {
                "status": "success",
                "job_id": job_id,
                "image_index": image_index,
                "image_path": image_path,
                "images_uploaded": job.images_uploaded,
                "total_images": job.num_images
            }
        
        # Process job endpoint
        @self.app.post("/api/jobs/{job_id}/process", status_code=202, dependencies=[Depends(verify_api_key)])
        async def process_job(
            job_id: str,
            process_request: ProcessRequest,
            background_tasks: BackgroundTasks
        ):
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            
            # Check if job is in a valid state for processing
            if job.status != "uploading":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot process job in {job.status} state"
                )
            
            # Check if all images have been uploaded
            if job.images_uploaded < job.num_images:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not all images uploaded ({job.images_uploaded}/{job.num_images})"
                )
            
            # Update job status
            job.status = "processing"
            job.updated_at = time.time()
            job.progress = 0.0
            
            # Start processing in the background
            background_tasks.add_task(
                self.process_job_task,
                job_id,
                process_request.reconstruction_quality,
                process_request.output_formats
            )
            
            return {
                "status": "success",
                "job_id": job_id,
                "message": "Processing started"
            }
        
        # Model download endpoint
        @self.app.get("/api/jobs/{job_id}/model", dependencies=[Depends(verify_api_key)])
        async def download_model(job_id: str, format: str = "obj"):
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            
            # Check if job is completed
            if job.status != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Job is not completed. Current status: {job.status}"
                )
            
            # Check if the requested format is available
            if format == "obj" and job.obj_file:
                return FileResponse(job.obj_file, filename=f"{job.model_id}.obj")
            elif format == "ply" and job.point_cloud_file:
                return FileResponse(job.point_cloud_file, filename=f"{job.model_id}.ply")
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model in {format} format not available"
                )
    
    def create_job(self, num_images: int) -> Dict[str, Any]:
        """
        Create a new job.
        
        Args:
            num_images: Number of images to be uploaded
            
        Returns:
            Dictionary with job information
        """
        # Check if we've reached the maximum number of jobs
        active_jobs = sum(1 for job in self.jobs.values() if job.status in ["created", "uploading", "processing"])
        if active_jobs >= self.config.max_jobs:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum number of active jobs ({self.config.max_jobs}) reached"
            )
        
        # Check if the number of images is valid
        if num_images <= 0 or num_images > self.config.max_images_per_job:
            raise HTTPException(
                status_code=400,
                detail=f"Number of images must be between 1 and {self.config.max_images_per_job}"
            )
        
        # Create a new job
        job_id = str(uuid.uuid4())
        job = JobStatus(
            job_id=job_id,
            num_images=num_images
        )
        
        # Add job to the list
        self.jobs[job_id] = job
        
        # Create job directory
        job_dir = os.path.join(self.config.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        return job.dict()
    
    async def process_job_task(
        self,
        job_id: str,
        reconstruction_quality: str,
        output_formats: List[str]
    ):
        """
        Process a job in the background.
        
        Args:
            job_id: ID of the job to process
            reconstruction_quality: Quality of the reconstruction (low, normal, high)
            output_formats: List of output formats to generate
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        job_dir = os.path.join(self.config.output_dir, job_id)
        images_dir = os.path.join(job_dir, "images")
        output_dir = os.path.join(job_dir, "output")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a model ID
        model_id = f"model_{job_id[:8]}"
        job.model_id = model_id
        
        # Update job status
        job.status = "processing"
        job.progress = 0.0
        job.updated_at = time.time()
        
        try:
            # Start timing
            start_time = time.time()
            
            # Run CUDA MVS
            await self.run_cuda_mvs(
                job,
                images_dir,
                output_dir,
                model_id,
                reconstruction_quality
            )
            
            # Convert output formats if needed
            if "obj" in output_formats and not job.obj_file:
                # Convert PLY to OBJ if needed
                ply_file = job.point_cloud_file
                if ply_file and os.path.exists(ply_file):
                    obj_file = os.path.join(output_dir, f"{model_id}.obj")
                    await self.convert_ply_to_obj(ply_file, obj_file)
                    job.obj_file = obj_file
            
            # Calculate processing time
            job.processing_time = time.time() - start_time
            
            # Update job status
            job.status = "completed"
            job.progress = 100.0
            job.updated_at = time.time()
            job.output_dir = output_dir
            
            logger.info(f"Job {job_id} completed successfully in {job.processing_time:.2f} seconds")
            
        except Exception as e:
            # Update job status
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = time.time()
            
            logger.error(f"Job {job_id} failed: {e}")
    
    async def run_cuda_mvs(
        self,
        job: JobStatus,
        images_dir: str,
        output_dir: str,
        model_id: str,
        reconstruction_quality: str
    ):
        """
        Run CUDA MVS on the uploaded images.
        
        Args:
            job: Job status object
            images_dir: Directory containing the images
            output_dir: Directory to save the output
            model_id: ID of the model
            reconstruction_quality: Quality of the reconstruction
        """
        # Check if CUDA MVS is installed
        cuda_mvs_executable = os.path.join(self.config.cuda_mvs_path, "build", "app_patch_match_mvs")
        if not os.path.exists(cuda_mvs_executable):
            raise FileNotFoundError(f"CUDA MVS executable not found at {cuda_mvs_executable}")
        
        # Create a list of image paths
        image_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(images_dir, file))
        
        if not image_files:
            raise ValueError("No valid image files found")
        
        # Sort image files to ensure consistent order
        image_files.sort()
        
        # Create a camera parameter file
        camera_params_file = os.path.join(output_dir, "cameras.txt")
        await self.generate_camera_params(image_files, camera_params_file)
        
        # Set quality parameters
        if reconstruction_quality == "low":
            num_iterations = 3
            max_resolution = 1024
        elif reconstruction_quality == "normal":
            num_iterations = 5
            max_resolution = 2048
        elif reconstruction_quality == "high":
            num_iterations = 7
            max_resolution = 4096
        else:
            num_iterations = 5
            max_resolution = 2048
        
        # Prepare output files
        point_cloud_file = os.path.join(output_dir, f"{model_id}.ply")
        
        # Build the command
        cmd = [
            cuda_mvs_executable,
            "--input_folder", images_dir,
            "--camera_file", camera_params_file,
            "--output_folder", output_dir,
            "--output_file", point_cloud_file,
            "--num_iterations", str(num_iterations),
            "--max_resolution", str(max_resolution)
        ]
        
        # Run the command
        logger.info(f"Running CUDA MVS: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress
        while True:
            if process.poll() is not None:
                break
            
            # Read output line by line
            output = process.stdout.readline()
            if output:
                # Try to parse progress information
                if "Progress:" in output:
                    try:
                        progress_str = output.split("Progress:")[1].strip().rstrip("%")
                        progress = float(progress_str)
                        job.progress = progress
                        job.updated_at = time.time()
                    except (ValueError, IndexError):
                        pass
            
            # Sleep briefly to avoid CPU spinning
            await asyncio.sleep(0.1)
        
        # Get the final output
        stdout, stderr = process.communicate()
        
        # Check if the process was successful
        if process.returncode != 0:
            raise RuntimeError(f"CUDA MVS failed with error: {stderr}")
        
        # Check if the output file was created
        if not os.path.exists(point_cloud_file):
            raise FileNotFoundError(f"Output file not created: {point_cloud_file}")
        
        # Update job with the output file
        job.point_cloud_file = point_cloud_file
    
    async def generate_camera_params(self, image_files: List[str], output_file: str):
        """
        Generate camera parameters for CUDA MVS.
        
        Args:
            image_files: List of image files
            output_file: Output file for camera parameters
        """
        # For now, use a simple camera model with default parameters
        # In a real implementation, this would use structure from motion
        # to estimate camera parameters from the images
        
        with open(output_file, "w") as f:
            f.write(f"# Camera parameters for {len(image_files)} images\n")
            f.write("# Format: image_name width height fx fy cx cy\n")
            
            for i, image_file in enumerate(image_files):
                # Get image dimensions
                from PIL import Image
                with Image.open(image_file) as img:
                    width, height = img.size
                
                # Use default camera parameters
                fx = width * 1.2  # Focal length x
                fy = height * 1.2  # Focal length y
                cx = width / 2  # Principal point x
                cy = height / 2  # Principal point y
                
                # Write camera parameters
                f.write(f"{os.path.basename(image_file)} {width} {height} {fx} {fy} {cx} {cy}\n")
    
    async def convert_ply_to_obj(self, ply_file: str, obj_file: str):
        """
        Convert PLY file to OBJ format.
        
        Args:
            ply_file: Input PLY file
            obj_file: Output OBJ file
        """
        try:
            import open3d as o3d
            
            # Load the PLY file
            mesh = o3d.io.read_triangle_mesh(ply_file)
            
            # Save as OBJ
            o3d.io.write_triangle_mesh(obj_file, mesh)
            
            logger.info(f"Converted {ply_file} to {obj_file}")
            
        except ImportError:
            # If open3d is not available, use a subprocess
            try:
                # Try using meshlab
                subprocess.run(
                    ["meshlabserver", "-i", ply_file, "-o", obj_file],
                    check=True,
                    capture_output=True
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                # If meshlab is not available, try using assimp
                try:
                    subprocess.run(
                        ["assimp", "export", ply_file, obj_file],
                        check=True,
                        capture_output=True
                    )
                except (subprocess.SubprocessError, FileNotFoundError):
                    raise RuntimeError("No suitable tool found to convert PLY to OBJ")
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information.
        
        Returns:
            Dictionary with server information
        """
        # Count active jobs
        active_jobs = sum(1 for job in self.jobs.values() if job.status in ["created", "uploading", "processing"])
        
        # Get server capabilities
        capabilities = {
            "max_jobs": self.config.max_jobs,
            "max_images_per_job": self.config.max_images_per_job,
            "job_timeout": self.config.job_timeout,
            "supported_formats": ["obj", "ply"],
            "gpu_info": self.config.gpu_info
        }
        
        # Get job summary
        job_summary = {
            "total": len(self.jobs),
            "active": active_jobs,
            "completed": sum(1 for job in self.jobs.values() if job.status == "completed"),
            "failed": sum(1 for job in self.jobs.values() if job.status == "failed")
        }
        
        return {
            "server_id": self.server_id,
            "name": self.config.server_name,
            "version": "1.0.0",
            "status": "running",
            "capabilities": capabilities,
            "jobs": job_summary,
            "uptime": time.time() - self.start_time
        }
    
    def advertise_service(self):
        """Advertise the server on the local network using Zeroconf."""
        try:
            import socket
            from zeroconf import ServiceInfo, Zeroconf
            
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Create service info
            capabilities = {
                "max_jobs": self.config.max_jobs,
                "max_images_per_job": self.config.max_images_per_job,
                "supported_formats": ["obj", "ply"],
                "gpu_info": self.config.gpu_info
            }
            
            service_info = ServiceInfo(
                "_cudamvs._tcp.local.",
                f"{self.server_id}._cudamvs._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=DEFAULT_PORT,
                properties={
                    b"name": self.config.server_name.encode("utf-8"),
                    b"capabilities": json.dumps(capabilities).encode("utf-8")
                }
            )
            
            # Register service
            self.zeroconf = Zeroconf()
            self.zeroconf.register_service(service_info)
            
            logger.info(f"Advertising CUDA MVS service on {local_ip}:{DEFAULT_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to advertise service: {e}")
    
    def run(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Run the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        uvicorn.run(self.app, host=host, port=port)
    
    def cleanup(self):
        """Clean up resources."""
        if self.zeroconf:
            self.zeroconf.close()

# Main entry point
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CUDA MVS Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--cuda-mvs-path", default=DEFAULT_CUDA_MVS_PATH, help="Path to CUDA MVS installation")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS, help="Maximum number of concurrent jobs")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES_PER_JOB, help="Maximum images per job")
    parser.add_argument("--job-timeout", type=int, default=DEFAULT_JOB_TIMEOUT, help="Job timeout in seconds")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--server-name", default="CUDA MVS Server", help="Server name")
    parser.add_argument("--no-advertise", action="store_true", help="Don't advertise service on the network")
    
    args = parser.parse_args()
    
    # Create server configuration
    config = ServerConfig(
        cuda_mvs_path=args.cuda_mvs_path,
        output_dir=args.output_dir,
        max_jobs=args.max_jobs,
        max_images_per_job=args.max_images,
        job_timeout=args.job_timeout,
        api_key=args.api_key,
        server_name=args.server_name,
        advertise_service=not args.no_advertise
    )
    
    # Create and run server
    server = CUDAMVSServer(config)
    
    try:
        server.run(host=args.host, port=args.port)
    finally:
        server.cleanup()

if __name__ == "__main__":
    # Add asyncio import for async/await support
    import asyncio
    main()
