"""
Test script for multi-view to model pipeline.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.multi_view_to_model_pipeline import MultiViewToModelPipeline
from src.ai.gemini_api import GeminiImageGenerator
from src.models.cuda_mvs import CUDAMultiViewStereo
from src.workflow.image_approval import ImageApprovalTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMultiViewPipeline(unittest.TestCase):
    """
    Test cases for multi-view to model pipeline.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test directories
        self.test_output_dir = "output/test_pipeline"
        self.test_images_dir = os.path.join(self.test_output_dir, "images")
        self.test_models_dir = os.path.join(self.test_output_dir, "models")
        
        for directory in [self.test_output_dir, self.test_images_dir, self.test_models_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create mock CUDA MVS path
        self.cuda_mvs_path = "mock_cuda_mvs"
        os.makedirs(os.path.join(self.cuda_mvs_path, "build"), exist_ok=True)
        
        # Create mock executable
        with open(os.path.join(self.cuda_mvs_path, "build", "app_patch_match_mvs"), "w") as f:
            f.write("#!/bin/bash\necho 'Mock CUDA MVS'\n")
        os.chmod(os.path.join(self.cuda_mvs_path, "build", "app_patch_match_mvs"), 0o755)
        
        # Create mock components
        self.mock_gemini = MagicMock(spec=GeminiImageGenerator)
        self.mock_cuda_mvs = MagicMock(spec=CUDAMultiViewStereo)
        self.mock_approval = MagicMock(spec=ImageApprovalTool)
        
        # Configure mock responses
        self.configure_mocks()
        
        # Create the pipeline with mock components
        self.pipeline = MultiViewToModelPipeline(
            gemini_generator=self.mock_gemini,
            cuda_mvs=self.mock_cuda_mvs,
            approval_tool=self.mock_approval,
            output_dir=self.test_output_dir
        )
    
    def configure_mocks(self):
        """
        Configure mock responses for components.
        """
        # Mock Gemini image generation
        def mock_generate_image(prompt, **kwargs):
            image_path = os.path.join(self.test_images_dir, f"{prompt[:10].replace(' ', '_')}.png")
            with open(image_path, "w") as f:
                f.write(f"Mock image for {prompt}")
            return {
                "prompt": prompt,
                "local_path": image_path,
                "image_data": b"mock_image_data"
            }
        
        def mock_generate_multiple_views(prompt, num_views, **kwargs):
            results = []
            for i in range(num_views):
                image_path = os.path.join(self.test_images_dir, f"view_{i}.png")
                with open(image_path, "w") as f:
                    f.write(f"Mock image for {prompt} - view {i}")
                results.append({
                    "prompt": f"{prompt} - view {i}",
                    "local_path": image_path,
                    "image_data": b"mock_image_data",
                    "view_direction": f"view {i}",
                    "view_index": i + 1
                })
            return results
        
        self.mock_gemini.generate_image.side_effect = mock_generate_image
        self.mock_gemini.generate_multiple_views.side_effect = mock_generate_multiple_views
        
        # Mock CUDA MVS
        def mock_generate_model(image_paths, **kwargs):
            model_dir = os.path.join(self.test_models_dir, "mock_model")
            os.makedirs(model_dir, exist_ok=True)
            
            point_cloud_file = os.path.join(model_dir, "mock_model.ply")
            with open(point_cloud_file, "w") as f:
                f.write("Mock point cloud")
            
            obj_file = os.path.join(model_dir, "mock_model.obj")
            with open(obj_file, "w") as f:
                f.write("Mock OBJ file")
            
            return {
                "model_id": "mock_model",
                "output_dir": model_dir,
                "point_cloud_file": point_cloud_file,
                "obj_file": obj_file,
                "input_images": image_paths
            }
        
        self.mock_cuda_mvs.generate_model_from_images.side_effect = mock_generate_model
        self.mock_cuda_mvs.convert_ply_to_obj.return_value = os.path.join(self.test_models_dir, "mock_model", "mock_model.obj")
        
        # Mock approval tool
        def mock_present_image(image_path, metadata):
            return {
                "approval_id": os.path.basename(image_path).split('.')[0],
                "image_path": image_path,
                "image_url": f"/images/{os.path.basename(image_path)}",
                "metadata": metadata or {}
            }
        
        def mock_process_approval(approval_id, approved, image_path):
            if approved:
                approved_path = os.path.join(self.test_output_dir, "approved", os.path.basename(image_path))
                os.makedirs(os.path.dirname(approved_path), exist_ok=True)
                with open(approved_path, "w") as f:
                    f.write(f"Approved image {approval_id}")
                
                return {
                    "approval_id": approval_id,
                    "approved": True,
                    "original_path": image_path,
                    "approved_path": approved_path
                }
            else:
                return {
                    "approval_id": approval_id,
                    "approved": False,
                    "original_path": image_path
                }
        
        self.mock_approval.present_image_for_approval.side_effect = mock_present_image
        self.mock_approval.process_approval.side_effect = mock_process_approval
        self.mock_approval.get_approved_images.return_value = [
            os.path.join(self.test_output_dir, "approved", f"view_{i}.png") for i in range(3)
        ]
    
    def test_generate_model_from_text(self):
        """
        Test generating a 3D model from text prompt.
        """
        # Test parameters
        prompt = "A low-poly rabbit"
        num_views = 3
        
        # Mock approvals - approve all images
        def mock_get_approval(approval_request):
            return True
        
        # Call the method
        result = self.pipeline.generate_model_from_text(
            prompt, num_views=num_views, get_approval_callback=mock_get_approval
        )
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("model_id" in result)
        self.assertTrue("obj_file" in result)
        self.assertTrue("point_cloud_file" in result)
        
        # Verify component calls
        self.mock_gemini.generate_multiple_views.assert_called_once_with(
            prompt, num_views=num_views, output_dir=os.path.join(self.test_output_dir, "multi_view")
        )
        
        self.assertEqual(self.mock_approval.present_image_for_approval.call_count, num_views)
        self.assertEqual(self.mock_approval.process_approval.call_count, num_views)
        
        self.mock_cuda_mvs.generate_model_from_images.assert_called_once()
        self.mock_cuda_mvs.convert_ply_to_obj.assert_called_once()
    
    def test_generate_model_from_image(self):
        """
        Test generating a 3D model from a base image.
        """
        # Create a mock base image
        base_image_path = os.path.join(self.test_images_dir, "base_image.png")
        with open(base_image_path, "w") as f:
            f.write("Mock base image")
        
        # Test parameters
        prompt = "A low-poly rabbit based on this image"
        num_views = 3
        
        # Mock approvals - approve all images
        def mock_get_approval(approval_request):
            return True
        
        # Call the method
        result = self.pipeline.generate_model_from_image(
            base_image_path, prompt, num_views=num_views, get_approval_callback=mock_get_approval
        )
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("model_id" in result)
        self.assertTrue("obj_file" in result)
        self.assertTrue("point_cloud_file" in result)
        
        # Verify component calls
        self.mock_gemini.generate_multiple_views.assert_called_once_with(
            prompt, num_views=num_views, base_image_path=base_image_path, 
            output_dir=os.path.join(self.test_output_dir, "multi_view")
        )
        
        self.assertEqual(self.mock_approval.present_image_for_approval.call_count, num_views)
        self.assertEqual(self.mock_approval.process_approval.call_count, num_views)
        
        self.mock_cuda_mvs.generate_model_from_images.assert_called_once()
        self.mock_cuda_mvs.convert_ply_to_obj.assert_called_once()
    
    def test_selective_approval(self):
        """
        Test selective approval of generated images.
        """
        # Test parameters
        prompt = "A low-poly rabbit"
        num_views = 4
        
        # Mock approvals - only approve views 0 and 2
        def mock_get_approval(approval_request):
            view_index = int(approval_request["approval_id"].split('_')[1])
            return view_index % 2 == 0  # Approve even-indexed views
        
        # Call the method
        result = self.pipeline.generate_model_from_text(
            prompt, num_views=num_views, get_approval_callback=mock_get_approval
        )
        
        # Verify the result
        self.assertIsNotNone(result)
        
        # Verify component calls
        self.assertEqual(self.mock_approval.present_image_for_approval.call_count, num_views)
        self.assertEqual(self.mock_approval.process_approval.call_count, num_views)
        
        # Only 2 images should be approved and used for model generation
        approved_images = [call[0][0] for call in self.mock_cuda_mvs.generate_model_from_images.call_args_list]
        if approved_images:
            self.assertEqual(len(approved_images[0]), 2)  # Only 2 images approved
    
    def test_error_handling(self):
        """
        Test error handling in the pipeline.
        """
        # Test parameters
        prompt = "A low-poly rabbit"
        
        # Mock error in Gemini API
        self.mock_gemini.generate_multiple_views.side_effect = Exception("Mock API error")
        
        # Call the method and expect an exception
        with self.assertRaises(Exception):
            self.pipeline.generate_model_from_text(prompt)
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Clean up test output directory
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        
        # Clean up mock CUDA MVS path
        if os.path.exists(self.cuda_mvs_path):
            shutil.rmtree(self.cuda_mvs_path)

if __name__ == "__main__":
    unittest.main()
