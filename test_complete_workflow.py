"""
Test script for the complete workflow from text to 3D model.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.gemini_api import GeminiImageGenerator
from src.workflow.image_approval import ImageApprovalManager
from src.models.cuda_mvs import CUDAMultiViewStereo
from src.workflow.multi_view_to_model_pipeline import MultiViewToModelPipeline
from src.config import MULTI_VIEW_PIPELINE, IMAGE_APPROVAL, REMOTE_CUDA_MVS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCompleteWorkflow(unittest.TestCase):
    """
    Test cases for the complete workflow from text to 3D model.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test output directories
        self.test_output_dir = "output/test_complete_workflow"
        self.test_images_dir = os.path.join(self.test_output_dir, "images")
        self.test_multi_view_dir = os.path.join(self.test_output_dir, "multi_view")
        self.test_approved_dir = os.path.join(self.test_output_dir, "approved")
        self.test_models_dir = os.path.join(self.test_output_dir, "models")
        
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_images_dir, exist_ok=True)
        os.makedirs(self.test_multi_view_dir, exist_ok=True)
        os.makedirs(self.test_approved_dir, exist_ok=True)
        os.makedirs(self.test_models_dir, exist_ok=True)
        
        # Mock API key
        self.api_key = "test_api_key"
        
        # Create the components
        self.image_generator = GeminiImageGenerator(
            api_key=self.api_key,
            output_dir=self.test_images_dir
        )
        
        self.approval_manager = ImageApprovalManager(
            output_dir=self.test_multi_view_dir,
            approved_dir=self.test_approved_dir,
            min_approved_images=3,
            auto_approve=False
        )
        
        self.cuda_mvs = CUDAMultiViewStereo(
            output_dir=self.test_models_dir,
            use_gpu=False
        )
        
        # Create the pipeline
        self.pipeline = MultiViewToModelPipeline(
            image_generator=self.image_generator,
            approval_manager=self.approval_manager,
            model_generator=self.cuda_mvs,
            output_dir=self.test_output_dir,
            config=MULTI_VIEW_PIPELINE
        )
    
    @patch('src.ai.gemini_api.requests.post')
    def test_generate_images_from_text(self, mock_post):
        """
        Test generating images from text description.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Generated image description"
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test parameters
        prompt = "A low-poly rabbit"
        num_views = 4
        
        # Call the method
        results = self.pipeline.generate_images_from_text(prompt, num_views)
        
        # Verify the results
        self.assertEqual(len(results), num_views)
        for i, result in enumerate(results):
            self.assertTrue("view_direction" in result)
            self.assertEqual(result["view_index"], i + 1)
            self.assertTrue("local_path" in result)
            self.assertTrue(os.path.exists(result["local_path"]))
        
        # Verify the API calls
        self.assertEqual(mock_post.call_count, num_views)
    
    def test_approve_images(self):
        """
        Test approving images in the workflow.
        """
        # Create test images
        test_images = []
        for i in range(4):
            image_path = os.path.join(self.test_multi_view_dir, f"test_image_{i}.png")
            with open(image_path, "w") as f:
                f.write(f"test image data {i}")
            test_images.append({
                "id": f"image_{i}",
                "local_path": image_path,
                "view_index": i + 1,
                "view_direction": f"view_{i}"
            })
        
        # Add images to the pipeline
        self.pipeline.add_images_for_approval(test_images)
        
        # Approve some images
        self.pipeline.approve_image("image_0")
        self.pipeline.approve_image("image_1")
        self.pipeline.approve_image("image_2")
        
        # Reject an image
        self.pipeline.reject_image("image_3")
        
        # Get the status
        status = self.pipeline.get_approval_status()
        
        # Verify the status
        self.assertEqual(status["total_images"], 4)
        self.assertEqual(status["pending_count"], 0)
        self.assertEqual(status["approved_count"], 3)
        self.assertEqual(status["rejected_count"], 1)
        self.assertTrue(status["has_minimum_approved"])
    
    @patch('src.models.cuda_mvs.subprocess.run')
    def test_create_model_from_approved_images(self, mock_run):
        """
        Test creating a 3D model from approved images.
        """
        # Mock subprocess.run
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Create test images
        test_images = []
        for i in range(4):
            image_path = os.path.join(self.test_approved_dir, f"image_{i}.png")
            with open(image_path, "w") as f:
                f.write(f"test image data {i}")
            test_images.append({
                "id": f"image_{i}",
                "local_path": image_path,
                "view_index": i + 1,
                "view_direction": f"view_{i}"
            })
        
        # Create a mock model file
        model_path = os.path.join(self.test_models_dir, "test_model.obj")
        with open(model_path, "w") as f:
            f.write("test model data")
        
        # Call the method
        result = self.pipeline.create_model_from_approved_images("test_model")
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("model_path" in result)
        self.assertTrue("model_id" in result)
        self.assertTrue("format" in result)
        self.assertEqual(result["format"], "obj")
    
    @patch('src.ai.gemini_api.requests.post')
    @patch('src.models.cuda_mvs.subprocess.run')
    def test_complete_workflow(self, mock_run, mock_post):
        """
        Test the complete workflow from text to 3D model.
        """
        # Mock Gemini API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Generated image description"
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Mock subprocess.run
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Create a mock model file
        model_path = os.path.join(self.test_models_dir, "test_model.obj")
        with open(model_path, "w") as f:
            f.write("test model data")
        
        # Create a pipeline with auto-approve
        auto_pipeline = MultiViewToModelPipeline(
            image_generator=self.image_generator,
            approval_manager=ImageApprovalManager(
                output_dir=self.test_multi_view_dir,
                approved_dir=self.test_approved_dir,
                min_approved_images=3,
                auto_approve=True
            ),
            model_generator=self.cuda_mvs,
            output_dir=self.test_output_dir,
            config=MULTI_VIEW_PIPELINE
        )
        
        # Test parameters
        prompt = "A low-poly rabbit"
        num_views = 4
        
        # Call the complete workflow
        result = auto_pipeline.complete_workflow(prompt, num_views, "test_model")
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("model_path" in result)
        self.assertTrue("model_id" in result)
        self.assertTrue("format" in result)
        self.assertEqual(result["format"], "obj")
        self.assertTrue("prompt" in result)
        self.assertEqual(result["prompt"], prompt)
        self.assertTrue("num_views" in result)
        self.assertEqual(result["num_views"], num_views)
        self.assertTrue("approved_images" in result)
        self.assertEqual(len(result["approved_images"]), num_views)
    
    @patch('src.remote.cuda_mvs_client.requests.post')
    @patch('src.remote.cuda_mvs_client.requests.get')
    def test_remote_workflow(self, mock_get, mock_post):
        """
        Test the workflow with remote CUDA MVS processing.
        """
        # Mock upload response
        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 200
        mock_upload_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "uploaded",
            "message": "Images uploaded successfully"
        }
        
        # Mock process response
        mock_process_response = MagicMock()
        mock_process_response.status_code = 200
        mock_process_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "processing",
            "message": "Job started processing"
        }
        
        # Mock status response
        mock_status_response = MagicMock()
        mock_status_response.status_code = 200
        mock_status_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "completed",
            "progress": 100,
            "message": "Job completed successfully"
        }
        
        # Mock download response
        mock_download_response = MagicMock()
        mock_download_response.status_code = 200
        mock_download_response.content = b"test model data"
        
        # Set up the mock responses
        mock_post.side_effect = [mock_upload_response, mock_process_response]
        mock_get.side_effect = [mock_status_response, mock_download_response]
        
        # Create test images
        test_images = []
        for i in range(4):
            image_path = os.path.join(self.test_approved_dir, f"image_{i}.png")
            with open(image_path, "w") as f:
                f.write(f"test image data {i}")
            test_images.append({
                "id": f"image_{i}",
                "local_path": image_path,
                "view_index": i + 1,
                "view_direction": f"view_{i}"
            })
        
        # Create a remote CUDA MVS client
        from src.remote.cuda_mvs_client import CUDAMVSClient
        remote_client = CUDAMVSClient(
            api_key=self.api_key,
            output_dir=self.test_models_dir
        )
        
        # Create a pipeline with the remote client
        remote_pipeline = MultiViewToModelPipeline(
            image_generator=self.image_generator,
            approval_manager=self.approval_manager,
            model_generator=remote_client,
            output_dir=self.test_output_dir,
            config=MULTI_VIEW_PIPELINE
        )
        
        # Add the approved images
        remote_pipeline.add_images_for_approval(test_images)
        for image in test_images:
            remote_pipeline.approve_image(image["id"])
        
        # Call the method to create a model using the remote client
        with patch('src.workflow.multi_view_to_model_pipeline.CUDAMVSClient', return_value=remote_client):
            result = remote_pipeline.create_model_from_approved_images("test_model", server_url="http://test-server:8765")
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("model_path" in result)
        self.assertTrue("model_id" in result)
        self.assertTrue("format" in result)
        self.assertEqual(result["format"], "obj")
        self.assertTrue("job_id" in result)
        self.assertEqual(result["job_id"], "test_job_123")
    
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
