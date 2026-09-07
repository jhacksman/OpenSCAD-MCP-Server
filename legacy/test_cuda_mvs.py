"""
Test script for CUDA Multi-View Stereo integration.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cuda_mvs import CUDAMultiViewStereo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCUDAMVS(unittest.TestCase):
    """
    Test cases for CUDA Multi-View Stereo integration.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create a test output directory
        self.test_output_dir = "output/test_cuda_mvs"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create test image directory
        self.test_images_dir = "output/test_cuda_mvs/images"
        os.makedirs(self.test_images_dir, exist_ok=True)
        
        # Create mock CUDA MVS path
        self.cuda_mvs_path = "mock_cuda_mvs"
        os.makedirs(os.path.join(self.cuda_mvs_path, "build"), exist_ok=True)
        
        # Create mock executable
        with open(os.path.join(self.cuda_mvs_path, "build", "app_patch_match_mvs"), "w") as f:
            f.write("#!/bin/bash\necho 'Mock CUDA MVS'\n")
        os.chmod(os.path.join(self.cuda_mvs_path, "build", "app_patch_match_mvs"), 0o755)
        
        # Create test images
        for i in range(3):
            with open(os.path.join(self.test_images_dir, f"view_{i}.png"), "w") as f:
                f.write(f"Mock image {i}")
        
        # Create the CUDA MVS wrapper with the mock path
        with patch('os.path.exists', return_value=True):
            self.cuda_mvs = CUDAMultiViewStereo(
                cuda_mvs_path=self.cuda_mvs_path,
                output_dir=self.test_output_dir
            )
    
    @patch('subprocess.Popen')
    def test_generate_model_from_images(self, mock_popen):
        """
        Test generating a 3D model from multiple images.
        """
        # Mock subprocess
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("Mock stdout", "")
        mock_popen.return_value = mock_process
        
        # Mock file creation
        def mock_exists(path):
            if "point_cloud_file" in str(path):
                # Create the mock point cloud file
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("Mock point cloud")
                return True
            return os.path.exists(path)
        
        # Test parameters
        image_paths = [os.path.join(self.test_images_dir, f"view_{i}.png") for i in range(3)]
        output_name = "test_model"
        
        # Call the method with patched os.path.exists
        with patch('os.path.exists', side_effect=mock_exists):
            result = self.cuda_mvs.generate_model_from_images(image_paths, output_name=output_name)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["model_id"], output_name)
        self.assertTrue("point_cloud_file" in result)
        self.assertTrue("camera_params_file" in result)
        self.assertEqual(len(result["input_images"]), 3)
        
        # Verify the subprocess call
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        self.assertTrue("app_patch_match_mvs" in args[0][0])
    
    def test_generate_camera_params(self):
        """
        Test generating camera parameters from images.
        """
        # Test parameters
        image_paths = [os.path.join(self.test_images_dir, f"view_{i}.png") for i in range(3)]
        model_dir = os.path.join(self.test_output_dir, "camera_params_test")
        os.makedirs(model_dir, exist_ok=True)
        
        # Mock PIL.Image.open
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        
        # Call the method with patched PIL.Image.open
        with patch('PIL.Image.open', return_value=mock_image):
            params_file = self.cuda_mvs._generate_camera_params(image_paths, model_dir)
        
        # Verify the result
        self.assertTrue(os.path.exists(params_file))
        
        # Read the params file
        import json
        with open(params_file, "r") as f:
            params = json.load(f)
        
        # Verify the params
        self.assertEqual(len(params), 3)
        for i, param in enumerate(params):
            self.assertEqual(param["image_id"], i)
            self.assertEqual(param["width"], 800)
            self.assertEqual(param["height"], 600)
            self.assertTrue("camera" in param)
            self.assertEqual(param["camera"]["model"], "PINHOLE")
    
    def test_convert_ply_to_obj(self):
        """
        Test converting PLY point cloud to OBJ mesh.
        """
        # Create a mock PLY file
        ply_file = os.path.join(self.test_output_dir, "test.ply")
        with open(ply_file, "w") as f:
            f.write("Mock PLY file")
        
        # Call the method
        obj_file = self.cuda_mvs.convert_ply_to_obj(ply_file)
        
        # Verify the result
        self.assertTrue(os.path.exists(obj_file))
        self.assertTrue(obj_file.endswith(".obj"))
        
        # Read the OBJ file
        with open(obj_file, "r") as f:
            content = f.read()
        
        # Verify the content
        self.assertTrue("# Converted from test.ply" in content)
        self.assertTrue("v " in content)
        self.assertTrue("f " in content)
    
    def test_error_handling(self):
        """
        Test error handling in the CUDA MVS wrapper.
        """
        # Test parameters
        image_paths = [os.path.join(self.test_images_dir, f"view_{i}.png") for i in range(3)]
        output_name = "error_test"
        
        # Mock subprocess with error
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = ("", "Mock error")
        
        # Call the method with patched subprocess.Popen
        with patch('subprocess.Popen', return_value=mock_process):
            with self.assertRaises(RuntimeError):
                self.cuda_mvs.generate_model_from_images(image_paths, output_name=output_name)
    
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
