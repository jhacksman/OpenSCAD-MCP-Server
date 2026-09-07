"""
Connection manager for remote CUDA Multi-View Stereo processing.

This module provides functionality to discover, connect to, and manage
connections with remote CUDA MVS servers within the LAN.
"""

import os
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
import socket
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

from src.remote.cuda_mvs_client import CUDAMVSClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CUDAMVSConnectionManager:
    """
    Connection manager for remote CUDA MVS servers.
    
    This class handles:
    1. Discovering available CUDA MVS servers on the LAN
    2. Managing connections to multiple servers
    3. Load balancing across available servers
    4. Monitoring server health and status
    5. Automatic failover if a server becomes unavailable
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        discovery_port: int = 8765,
        connection_timeout: int = 10,
        health_check_interval: int = 60,
        auto_discover: bool = True
    ):
        """
        Initialize the connection manager.
        
        Args:
            api_key: API key for authentication (if required)
            discovery_port: Port used for server discovery
            connection_timeout: Timeout for server connections in seconds
            health_check_interval: Interval for health checks in seconds
            auto_discover: Whether to automatically discover servers on startup
        """
        self.api_key = api_key
        self.discovery_port = discovery_port
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        
        # Server tracking
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.clients: Dict[str, CUDAMVSClient] = {}
        self.server_lock = threading.RLock()
        
        # Health check thread
        self.health_check_thread = None
        self.health_check_stop_event = threading.Event()
        
        # Discovery
        self.zeroconf = None
        self.browser = None
        
        # Start discovery if enabled
        if auto_discover:
            self.start_discovery()
            
        # Start health check thread
        self.start_health_check()
    
    def start_discovery(self):
        """
        Start discovering CUDA MVS servers on the LAN.
        """
        if self.zeroconf is not None:
            return
        
        try:
            self.zeroconf = Zeroconf()
            listener = CUDAMVSServiceListener(self)
            self.browser = ServiceBrowser(self.zeroconf, "_cudamvs._tcp.local.", listener)
            logger.info("Started CUDA MVS server discovery")
        except Exception as e:
            logger.error(f"Error starting discovery: {e}")
    
    def stop_discovery(self):
        """
        Stop discovering CUDA MVS servers.
        """
        if self.zeroconf is not None:
            try:
                self.zeroconf.close()
                self.zeroconf = None
                self.browser = None
                logger.info("Stopped CUDA MVS server discovery")
            except Exception as e:
                logger.error(f"Error stopping discovery: {e}")
    
    def start_health_check(self):
        """
        Start the health check thread.
        """
        if self.health_check_thread is not None and self.health_check_thread.is_alive():
            return
        
        self.health_check_stop_event.clear()
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Started health check thread")
    
    def stop_health_check(self):
        """
        Stop the health check thread.
        """
        if self.health_check_thread is not None:
            self.health_check_stop_event.set()
            self.health_check_thread.join(timeout=5)
            self.health_check_thread = None
            logger.info("Stopped health check thread")
    
    def _health_check_loop(self):
        """
        Health check loop that runs in a separate thread.
        """
        while not self.health_check_stop_event.is_set():
            try:
                self.check_all_servers()
            except Exception as e:
                logger.error(f"Error in health check: {e}")
            
            # Wait for the next check interval or until stopped
            self.health_check_stop_event.wait(self.health_check_interval)
    
    def add_server(self, server_info: Dict[str, Any]):
        """
        Add a server to the manager.
        
        Args:
            server_info: Dictionary with server information
        """
        server_id = server_info.get("server_id")
        if not server_id:
            logger.error("Cannot add server without server_id")
            return
        
        with self.server_lock:
            # Check if server already exists
            if server_id in self.servers:
                # Update existing server info
                self.servers[server_id].update(server_info)
                logger.info(f"Updated server: {server_info.get('name')} at {server_info.get('url')}")
            else:
                # Add new server
                self.servers[server_id] = server_info
                
                # Create client for the server
                self.clients[server_id] = CUDAMVSClient(
                    server_url=server_info.get("url"),
                    api_key=self.api_key,
                    connection_timeout=self.connection_timeout
                )
                
                logger.info(f"Added server: {server_info.get('name')} at {server_info.get('url')}")
    
    def remove_server(self, server_id: str):
        """
        Remove a server from the manager.
        
        Args:
            server_id: ID of the server to remove
        """
        with self.server_lock:
            if server_id in self.servers:
                server_info = self.servers.pop(server_id)
                if server_id in self.clients:
                    del self.clients[server_id]
                logger.info(f"Removed server: {server_info.get('name')} at {server_info.get('url')}")
    
    def get_servers(self) -> List[Dict[str, Any]]:
        """
        Get a list of all servers.
        
        Returns:
            List of dictionaries with server information
        """
        with self.server_lock:
            return list(self.servers.values())
    
    def get_server(self, server_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Dictionary with server information or None if not found
        """
        with self.server_lock:
            return self.servers.get(server_id)
    
    def get_client(self, server_id: str) -> Optional[CUDAMVSClient]:
        """
        Get the client for a specific server.
        
        Args:
            server_id: ID of the server
            
        Returns:
            CUDAMVSClient instance or None if not found
        """
        with self.server_lock:
            return self.clients.get(server_id)
    
    def get_best_server(self) -> Optional[str]:
        """
        Get the ID of the best server to use based on availability and load.
        
        Returns:
            Server ID or None if no servers are available
        """
        with self.server_lock:
            available_servers = [
                server_id for server_id, server in self.servers.items()
                if server.get("status") == "available"
            ]
            
            if not available_servers:
                return None
            
            # For now, just return the first available server
            # In a more advanced implementation, this would consider
            # server load, capabilities, latency, etc.
            return available_servers[0]
    
    def check_server(self, server_id: str) -> Dict[str, Any]:
        """
        Check the status of a specific server.
        
        Args:
            server_id: ID of the server to check
            
        Returns:
            Dictionary with server status information
        """
        client = self.get_client(server_id)
        if not client:
            return {"status": "error", "message": f"Server {server_id} not found"}
        
        # Test connection
        result = client.test_connection()
        
        with self.server_lock:
            if server_id in self.servers:
                # Update server status
                if result["status"] == "success":
                    self.servers[server_id]["status"] = "available"
                    self.servers[server_id]["last_check"] = time.time()
                    self.servers[server_id]["latency_ms"] = result.get("latency_ms")
                    
                    # Update capabilities if available
                    if "server_info" in result and "capabilities" in result["server_info"]:
                        self.servers[server_id]["capabilities"] = result["server_info"]["capabilities"]
                else:
                    self.servers[server_id]["status"] = "unavailable"
                    self.servers[server_id]["last_check"] = time.time()
                    self.servers[server_id]["error"] = result.get("message")
        
        return result
    
    def check_all_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Check the status of all servers.
        
        Returns:
            Dictionary mapping server IDs to status information
        """
        results = {}
        
        with self.server_lock:
            server_ids = list(self.servers.keys())
        
        for server_id in server_ids:
            results[server_id] = self.check_server(server_id)
        
        return results
    
    def discover_servers(self) -> List[Dict[str, Any]]:
        """
        Manually discover CUDA MVS servers on the LAN.
        
        Returns:
            List of dictionaries containing server information
        """
        # Create a temporary client to discover servers
        client = CUDAMVSClient(
            api_key=self.api_key,
            discovery_port=self.discovery_port,
            connection_timeout=self.connection_timeout
        )
        
        discovered_servers = client.discover_servers()
        
        # Add discovered servers
        for server_info in discovered_servers:
            self.add_server(server_info)
        
        return discovered_servers
    
    def upload_images_to_best_server(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Upload images to the best available server.
        
        Args:
            image_paths: List of paths to images to upload
            
        Returns:
            Dictionary with upload status and job information
        """
        server_id = self.get_best_server()
        if not server_id:
            return {"status": "error", "message": "No available servers"}
        
        client = self.get_client(server_id)
        if not client:
            return {"status": "error", "message": f"Client for server {server_id} not found"}
        
        # Upload images
        result = client.upload_images(image_paths)
        
        # Add server information to result
        result["server_id"] = server_id
        result["server_name"] = self.servers[server_id].get("name")
        
        return result
    
    def generate_model_from_images(
        self,
        image_paths: List[str],
        output_format: str = "obj",
        wait_for_completion: bool = True,
        poll_interval: int = 5,
        server_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a 3D model from images using the best available server.
        
        Args:
            image_paths: List of paths to images
            output_format: Format of the output model
            wait_for_completion: Whether to wait for job completion
            poll_interval: Interval in seconds to poll for job status
            server_id: ID of the server to use (uses best server if None)
            
        Returns:
            Dictionary with job status and model information if completed
        """
        # Get server to use
        if server_id is None:
            server_id = self.get_best_server()
            if not server_id:
                return {"status": "error", "message": "No available servers"}
        
        client = self.get_client(server_id)
        if not client:
            return {"status": "error", "message": f"Client for server {server_id} not found"}
        
        # Generate model
        result = client.generate_model_from_images(
            image_paths=image_paths,
            output_format=output_format,
            wait_for_completion=wait_for_completion,
            poll_interval=poll_interval
        )
        
        # Add server information to result
        result["server_id"] = server_id
        result["server_name"] = self.servers[server_id].get("name")
        
        return result
    
    def get_job_status(self, job_id: str, server_id: str) -> Dict[str, Any]:
        """
        Get the status of a job on a specific server.
        
        Args:
            job_id: ID of the job to check
            server_id: ID of the server
            
        Returns:
            Dictionary with job status information
        """
        client = self.get_client(server_id)
        if not client:
            return {"status": "error", "message": f"Client for server {server_id} not found"}
        
        return client.get_job_status(job_id)
    
    def download_model(self, job_id: str, server_id: str, output_format: str = "obj") -> Dict[str, Any]:
        """
        Download a processed 3D model from a specific server.
        
        Args:
            job_id: ID of the job to download
            server_id: ID of the server
            output_format: Format of the model to download
            
        Returns:
            Dictionary with download status and local file path
        """
        client = self.get_client(server_id)
        if not client:
            return {"status": "error", "message": f"Client for server {server_id} not found"}
        
        return client.download_model(job_id, output_format)
    
    def cancel_job(self, job_id: str, server_id: str) -> Dict[str, Any]:
        """
        Cancel a running job on a specific server.
        
        Args:
            job_id: ID of the job to cancel
            server_id: ID of the server
            
        Returns:
            Dictionary with cancellation status
        """
        client = self.get_client(server_id)
        if not client:
            return {"status": "error", "message": f"Client for server {server_id} not found"}
        
        return client.cancel_job(job_id)
    
    def cleanup(self):
        """
        Clean up resources.
        """
        self.stop_health_check()
        self.stop_discovery()


class CUDAMVSServiceListener(ServiceListener):
    """
    Zeroconf service listener for CUDA MVS servers.
    """
    
    def __init__(self, connection_manager: CUDAMVSConnectionManager):
        """
        Initialize the service listener.
        
        Args:
            connection_manager: Connection manager to update with discovered servers
        """
        self.connection_manager = connection_manager
    
    def add_service(self, zc: Zeroconf, type_: str, name: str):
        """
        Called when a service is discovered.
        
        Args:
            zc: Zeroconf instance
            type_: Service type
            name: Service name
        """
        info = zc.get_service_info(type_, name)
        if info:
            try:
                # Extract server information
                server_id = name.split('.')[0]
                server_name = info.properties.get(b'name', b'Unknown').decode('utf-8')
                
                # Get server URL
                addresses = info.parsed_addresses()
                if not addresses:
                    return
                
                server_url = f"http://{addresses[0]}:{info.port}"
                
                # Parse capabilities
                capabilities = {}
                if b'capabilities' in info.properties:
                    try:
                        capabilities = json.loads(info.properties[b'capabilities'].decode('utf-8'))
                    except json.JSONDecodeError:
                        pass
                
                # Create server info
                server_info = {
                    "server_id": server_id,
                    "name": server_name,
                    "url": server_url,
                    "capabilities": capabilities,
                    "status": "unknown",
                    "discovered_at": time.time()
                }
                
                # Add server to connection manager
                self.connection_manager.add_server(server_info)
                
            except Exception as e:
                logger.error(f"Error processing discovered service: {e}")
    
    def remove_service(self, zc: Zeroconf, type_: str, name: str):
        """
        Called when a service is removed.
        
        Args:
            zc: Zeroconf instance
            type_: Service type
            name: Service name
        """
        try:
            server_id = name.split('.')[0]
            self.connection_manager.remove_server(server_id)
        except Exception as e:
            logger.error(f"Error removing service: {e}")
    
    def update_service(self, zc: Zeroconf, type_: str, name: str):
        """
        Called when a service is updated.
        
        Args:
            zc: Zeroconf instance
            type_: Service type
            name: Service name
        """
        self.add_service(zc, type_, name)
