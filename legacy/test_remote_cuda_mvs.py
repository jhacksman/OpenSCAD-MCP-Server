"""
Test script for remote CUDA Multi-View Stereo processing.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.remote.cuda_mvs_client import CUDAMVSClient
from src.remote.connection_manager import CUDAMVSConnectionManager
from src.config import REMOTE_CUDA_MVS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRemoteCUDAMVS(unittest.TestCase):
    """
    Test cases for remote CUDA Multi-View Stereo processing.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test output directories
        self.test_output_dir = "output/test_remote_cuda_mvs"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Mock API key
        self.api_key = "test_api_key"
        
        # Create the client with the mock API key
        self.client = CUDAMVSClient(
            api_key=self.api_key,
            output_dir=self.test_output_dir
        )
        
        # Create the connection manager with the mock API key
        self.connection_manager = CUDAMVSConnectionManager(
            api_key=self.api_key,
            discovery_port=REMOTE_CUDA_MVS["DISCOVERY_PORT"],
            use_lan_discovery=True
        )
    
    @patch('src.remote.cuda_mvs_client.requests.post')
    def test_upload_images(self, mock_post):
        """
        Test uploading images to a remote CUDA MVS server.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "uploaded",
            "message": "Images uploaded successfully"
        }
        mock_post.return_value = mock_response
        
        # Test parameters
        server_url = "http://test-server:8765"
        image_paths = [
            os.path.join(self.test_output_dir, "test_image_1.png"),
            os.path.join(self.test_output_dir, "test_image_2.png")
        ]
        
        # Create test images
        for path in image_paths:
            with open(path, "w") as f:
                f.write("test image data")
        
        # Call the method
        result = self.client.upload_images(server_url, image_paths)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], "test_job_123")
        self.assertEqual(result["status"], "uploaded")
        
        # Verify the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertTrue(server_url in args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")
    
    @patch('src.remote.cuda_mvs_client.requests.post')
    def test_process_job(self, mock_post):
        """
        Test processing a job on a remote CUDA MVS server.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "processing",
            "message": "Job started processing"
        }
        mock_post.return_value = mock_response
        
        # Test parameters
        server_url = "http://test-server:8765"
        job_id = "test_job_123"
        params = {
            "quality": "normal",
            "output_format": "obj"
        }
        
        # Call the method
        result = self.client.process_job(server_url, job_id, params)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], "test_job_123")
        self.assertEqual(result["status"], "processing")
        
        # Verify the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertTrue(server_url in args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")
        self.assertEqual(kwargs["json"]["job_id"], job_id)
        self.assertEqual(kwargs["json"]["params"]["quality"], "normal")
    
    @patch('src.remote.cuda_mvs_client.requests.get')
    def test_get_job_status(self, mock_get):
        """
        Test getting the status of a job on a remote CUDA MVS server.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "completed",
            "progress": 100,
            "message": "Job completed successfully"
        }
        mock_get.return_value = mock_response
        
        # Test parameters
        server_url = "http://test-server:8765"
        job_id = "test_job_123"
        
        # Call the method
        result = self.client.get_job_status(server_url, job_id)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], "test_job_123")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["progress"], 100)
        
        # Verify the API call
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue(server_url in args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")
    
    @patch('src.remote.cuda_mvs_client.requests.get')
    def test_download_model(self, mock_get):
        """
        Test downloading a model from a remote CUDA MVS server.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test model data"
        mock_get.return_value = mock_response
        
        # Test parameters
        server_url = "http://test-server:8765"
        job_id = "test_job_123"
        output_dir = os.path.join(self.test_output_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the method
        result = self.client.download_model(server_url, job_id, output_dir)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("model_path" in result)
        self.assertTrue(os.path.exists(result["model_path"]))
        
        # Verify the API call
        mock_get.assert_called()
        args, kwargs = mock_get.call_args_list[0]
        self.assertTrue(server_url in args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")
    
    @patch('src.remote.connection_manager.zeroconf.Zeroconf')
    def test_discover_servers(self, mock_zeroconf):
        """
        Test discovering CUDA MVS servers on the network.
        """
        # Mock Zeroconf
        mock_zeroconf_instance = MagicMock()
        mock_zeroconf.return_value = mock_zeroconf_instance
        
        # Mock ServiceBrowser
        with patch('src.remote.connection_manager.ServiceBrowser') as mock_browser:
            # Set up the connection manager to discover servers
            self.connection_manager.discover_servers()
            
            # Verify Zeroconf was initialized
            mock_zeroconf.assert_called_once()
            
            # Verify ServiceBrowser was initialized
            mock_browser.assert_called_once()
            args, kwargs = mock_browser.call_args
            self.assertEqual(args[0], mock_zeroconf_instance)
            self.assertEqual(args[1], "_cudamvs._tcp.local.")
    
    @patch('src.remote.connection_manager.CUDAMVSClient')
    def test_upload_images_with_connection_manager(self, mock_client_class):
        """
        Test uploading images using the connection manager.
        """
        # Mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock upload_images method
        mock_client.upload_images.return_value = {
            "job_id": "test_job_123",
            "status": "uploaded",
            "message": "Images uploaded successfully"
        }
        
        # Add a mock server
        self.connection_manager.servers = {
            "test_server": {
                "id": "test_server",
                "name": "Test Server",
                "url": "http://test-server:8765",
                "status": "online"
            }
        }
        
        # Test parameters
        server_id = "test_server"
        image_paths = [
            os.path.join(self.test_output_dir, "test_image_1.png"),
            os.path.join(self.test_output_dir, "test_image_2.png")
        ]
        
        # Create test images
        for path in image_paths:
            with open(path, "w") as f:
                f.write("test image data")
        
        # Call the method
        result = self.connection_manager.upload_images(server_id, image_paths)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["job_id"], "test_job_123")
        self.assertEqual(result["status"], "uploaded")
        
        # Verify the client method was called
        mock_client.upload_images.assert_called_once()
        args, kwargs = mock_client.upload_images.call_args
        self.assertEqual(args[0], "http://test-server:8765")
        self.assertEqual(args[1], image_paths)
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Clean up test output directory
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

if __name__ == "__main__":
    unittest.main()
