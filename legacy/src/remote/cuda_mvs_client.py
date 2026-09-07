"""
Client for remote CUDA Multi-View Stereo processing.

This module provides a client to connect to a remote CUDA MVS server
within the LAN for processing multi-view images into 3D models.
"""

import os
import json
import logging
import requests
import base64
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CUDAMVSClient:
    """
    Client for connecting to a remote CUDA Multi-View Stereo server.
    
    This client handles:
    1. Discovering available CUDA MVS servers on the LAN
    2. Uploading images to the server
    3. Requesting 3D reconstruction
    4. Downloading the resulting 3D models
    5. Monitoring job status
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "output/models",
        discovery_port: int = 8765,
        connection_timeout: int = 10,
        upload_chunk_size: int = 1024 * 1024,  # 1MB chunks
    ):
        """
        Initialize the CUDA MVS client.
        
        Args:
            server_url: URL of the CUDA MVS server (if known)
            api_key: API key for authentication (if required)
            output_dir: Directory to save downloaded models
            discovery_port: Port used for server discovery
            connection_timeout: Timeout for server connections in seconds
            upload_chunk_size: Chunk size for file uploads in bytes
        """
        self.server_url = server_url
        self.api_key = api_key
        self.output_dir = output_dir
        self.discovery_port = discovery_port
        self.connection_timeout = connection_timeout
        self.upload_chunk_size = upload_chunk_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize session for connection pooling
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def discover_servers(self) -> List[Dict[str, Any]]:
        """
        Discover CUDA MVS servers on the local network.
        
        Returns:
            List of dictionaries containing server information:
            [
                {
                    "server_id": "unique-server-id",
                    "name": "CUDA MVS Server 1",
                    "url": "http://192.168.1.100:8765",
                    "capabilities": {
                        "max_images": 50,
                        "max_resolution": 4096,
                        "supported_formats": ["jpg", "png"],
                        "gpu_info": "NVIDIA RTX 4090 24GB"
                    },
                    "status": "available"
                },
                ...
            ]
        """
        import socket
        import json
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
        
        discovered_servers = []
        
        class CUDAMVSListener(ServiceListener):
            def add_service(self, zc, type_, name):
                info = zc.get_service_info(type_, name)
                if info:
                    server_info = {
                        "server_id": name.split('.')[0],
                        "name": info.properties.get(b'name', b'Unknown').decode('utf-8'),
                        "url": f"http://{socket.inet_ntoa(info.addresses[0])}:{info.port}",
                        "capabilities": json.loads(info.properties.get(b'capabilities', b'{}').decode('utf-8')),
                        "status": "available"
                    }
                    discovered_servers.append(server_info)
                    logger.info(f"Discovered CUDA MVS server: {server_info['name']} at {server_info['url']}")
        
        try:
            zeroconf = Zeroconf()
            listener = CUDAMVSListener()
            browser = ServiceBrowser(zeroconf, "_cudamvs._tcp.local.", listener)
            
            # Wait for discovery (non-blocking in production code)
            import time
            time.sleep(2)  # Give some time for discovery
            
            zeroconf.close()
            return discovered_servers
        except Exception as e:
            logger.error(f"Error discovering CUDA MVS servers: {e}")
            return []
    
    def test_connection(self, server_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Test connection to a CUDA MVS server.
        
        Args:
            server_url: URL of the server to test (uses self.server_url if None)
            
        Returns:
            Dictionary with connection status and server information
        """
        url = server_url or self.server_url
        if not url:
            return {"status": "error", "message": "No server URL provided"}
        
        try:
            response = self.session.get(
                f"{url}/api/status",
                timeout=self.connection_timeout
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "server_info": response.json(),
                    "latency_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                return {
                    "status": "error",
                    "message": f"Server returned status code {response.status_code}",
                    "details": response.text
                }
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def upload_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Upload images to the CUDA MVS server.
        
        Args:
            image_paths: List of paths to images to upload
            
        Returns:
            Dictionary with upload status and job information
        """
        if not self.server_url:
            return {"status": "error", "message": "No server URL configured"}
        
        # Create a new job
        try:
            response = self.session.post(
                f"{self.server_url}/api/jobs",
                json={"num_images": len(image_paths)},
                timeout=self.connection_timeout
            )
            
            if response.status_code != 201:
                return {
                    "status": "error",
                    "message": f"Failed to create job: {response.status_code}",
                    "details": response.text
                }
            
            job_info = response.json()
            job_id = job_info["job_id"]
            
            # Upload each image
            for i, image_path in enumerate(image_paths):
                # Check if file exists
                if not os.path.exists(image_path):
                    return {
                        "status": "error",
                        "message": f"Image file not found: {image_path}"
                    }
                
                # Get file size for progress tracking
                file_size = os.path.getsize(image_path)
                
                # Prepare upload
                with open(image_path, "rb") as f:
                    files = {
                        "file": (os.path.basename(image_path), f, "image/jpeg" if image_path.endswith(".jpg") else "image/png")
                    }
                    
                    metadata = {
                        "image_index": i,
                        "total_images": len(image_paths),
                        "filename": os.path.basename(image_path)
                    }
                    
                    response = self.session.post(
                        f"{self.server_url}/api/jobs/{job_id}/images",
                        files=files,
                        data={"metadata": json.dumps(metadata)},
                        timeout=None  # No timeout for uploads
                    )
                    
                    if response.status_code != 200:
                        return {
                            "status": "error",
                            "message": f"Failed to upload image {i+1}/{len(image_paths)}: {response.status_code}",
                            "details": response.text
                        }
                
                logger.info(f"Uploaded image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Start processing
            response = self.session.post(
                f"{self.server_url}/api/jobs/{job_id}/process",
                timeout=self.connection_timeout
            )
            
            if response.status_code != 202:
                return {
                    "status": "error",
                    "message": f"Failed to start processing: {response.status_code}",
                    "details": response.text
                }
            
            return {
                "status": "success",
                "job_id": job_id,
                "message": f"Uploaded {len(image_paths)} images and started processing",
                "job_url": f"{self.server_url}/api/jobs/{job_id}"
            }
            
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Upload error: {str(e)}"}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a CUDA MVS job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Dictionary with job status information
        """
        if not self.server_url:
            return {"status": "error", "message": "No server URL configured"}
        
        try:
            response = self.session.get(
                f"{self.server_url}/api/jobs/{job_id}",
                timeout=self.connection_timeout
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "job_info": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to get job status: {response.status_code}",
                    "details": response.text
                }
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def download_model(self, job_id: str, output_format: str = "obj") -> Dict[str, Any]:
        """
        Download a processed 3D model from the CUDA MVS server.
        
        Args:
            job_id: ID of the job to download
            output_format: Format of the model to download (obj, ply, etc.)
            
        Returns:
            Dictionary with download status and local file path
        """
        if not self.server_url:
            return {"status": "error", "message": "No server URL configured"}
        
        # Check job status first
        status_result = self.get_job_status(job_id)
        if status_result["status"] != "success":
            return status_result
        
        job_info = status_result["job_info"]
        if job_info["status"] != "completed":
            return {
                "status": "error",
                "message": f"Job is not completed yet. Current status: {job_info['status']}",
                "job_info": job_info
            }
        
        # Download the model
        try:
            response = self.session.get(
                f"{self.server_url}/api/jobs/{job_id}/model?format={output_format}",
                stream=True,
                timeout=None  # No timeout for downloads
            )
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Failed to download model: {response.status_code}",
                    "details": response.text
                }
            
            # Create a unique filename
            model_id = job_info.get("model_id", str(uuid.uuid4()))
            output_path = os.path.join(self.output_dir, f"{model_id}.{output_format}")
            
            # Save the file
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded model to {output_path}")
            
            return {
                "status": "success",
                "model_id": model_id,
                "local_path": output_path,
                "format": output_format,
                "job_id": job_id
            }
            
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Download error: {str(e)}"}
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running CUDA MVS job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            Dictionary with cancellation status
        """
        if not self.server_url:
            return {"status": "error", "message": "No server URL configured"}
        
        try:
            response = self.session.delete(
                f"{self.server_url}/api/jobs/{job_id}",
                timeout=self.connection_timeout
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "Job cancelled successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to cancel job: {response.status_code}",
                    "details": response.text
                }
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def generate_model_from_images(
        self,
        image_paths: List[str],
        output_format: str = "obj",
        wait_for_completion: bool = True,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Complete workflow to generate a 3D model from images.
        
        Args:
            image_paths: List of paths to images
            output_format: Format of the output model
            wait_for_completion: Whether to wait for job completion
            poll_interval: Interval in seconds to poll for job status
            
        Returns:
            Dictionary with job status and model information if completed
        """
        # Upload images and start processing
        upload_result = self.upload_images(image_paths)
        if upload_result["status"] != "success":
            return upload_result
        
        job_id = upload_result["job_id"]
        
        # If not waiting for completion, return the job info
        if not wait_for_completion:
            return upload_result
        
        # Poll for job completion
        import time
        while True:
            status_result = self.get_job_status(job_id)
            if status_result["status"] != "success":
                return status_result
            
            job_info = status_result["job_info"]
            if job_info["status"] == "completed":
                # Download the model
                return self.download_model(job_id, output_format)
            elif job_info["status"] == "failed":
                return {
                    "status": "error",
                    "message": "Job processing failed",
                    "job_info": job_info
                }
            
            # Wait before polling again
            time.sleep(poll_interval)
            logger.info(f"Job {job_id} status: {job_info['status']}, progress: {job_info.get('progress', 0)}%")
