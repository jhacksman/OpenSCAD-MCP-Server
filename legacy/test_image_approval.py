"""
Test script for image approval tool.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.image_approval import ImageApprovalTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestImageApproval(unittest.TestCase):
    """
    Test cases for image approval tool.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create a test output directory
        self.test_output_dir = "output/test_approval"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create test images directory
        self.test_images_dir = "output/test_approval/images"
        os.makedirs(self.test_images_dir, exist_ok=True)
        
        # Create test images
        self.test_images = []
        for i in range(3):
            image_path = os.path.join(self.test_images_dir, f"view_{i}.png")
            with open(image_path, "w") as f:
                f.write(f"Mock image {i}")
            self.test_images.append(image_path)
        
        # Create the approval tool
        self.approval_tool = ImageApprovalTool(
            output_dir=os.path.join(self.test_output_dir, "approved")
        )
    
    def test_present_image_for_approval(self):
        """
        Test presenting an image for approval.
        """
        # Test parameters
        image_path = self.test_images[0]
        metadata = {
            "prompt": "A test image",
            "view_direction": "front view",
            "view_index": 1
        }
        
        # Call the method
        result = self.approval_tool.present_image_for_approval(image_path, metadata)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertTrue("approval_id" in result)
        self.assertEqual(result["image_path"], image_path)
        self.assertTrue("image_url" in result)
        self.assertEqual(result["metadata"], metadata)
    
    def test_process_approval_approved(self):
        """
        Test processing an approved image.
        """
        # Test parameters
        image_path = self.test_images[0]
        approval_id = "test_approval_1"
        
        # Call the method
        result = self.approval_tool.process_approval(approval_id, True, image_path)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["approval_id"], approval_id)
        self.assertTrue(result["approved"])
        self.assertEqual(result["original_path"], image_path)
        self.assertTrue("approved_path" in result)
        
        # Verify the file was copied
        self.assertTrue(os.path.exists(result["approved_path"]))
    
    def test_process_approval_denied(self):
        """
        Test processing a denied image.
        """
        # Test parameters
        image_path = self.test_images[1]
        approval_id = "test_approval_2"
        
        # Call the method
        result = self.approval_tool.process_approval(approval_id, False, image_path)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["approval_id"], approval_id)
        self.assertFalse(result["approved"])
        self.assertEqual(result["original_path"], image_path)
        self.assertFalse("approved_path" in result)
    
    def test_get_approved_images(self):
        """
        Test getting approved images.
        """
        # Approve some images
        for i, image_path in enumerate(self.test_images):
            self.approval_tool.process_approval(f"test_approval_{i}", True, image_path)
        
        # Call the method
        approved_images = self.approval_tool.get_approved_images()
        
        # Verify the result
        self.assertEqual(len(approved_images), len(self.test_images))
    
    def test_get_approval_status(self):
        """
        Test getting approval status.
        """
        # Approve an image
        approval_id = "test_approval_status"
        self.approval_tool.process_approval(approval_id, True, self.test_images[0])
        
        # Call the method
        status = self.approval_tool.get_approval_status(approval_id)
        
        # Verify the result
        self.assertIsNotNone(status)
        self.assertEqual(status["approval_id"], approval_id)
        self.assertTrue(status["approved"])
        self.assertTrue("approved_path" in status)
        
        # Test with non-existent approval ID
        status = self.approval_tool.get_approval_status("non_existent")
        self.assertIsNotNone(status)
        self.assertEqual(status["approval_id"], "non_existent")
        self.assertFalse(status["approved"])
    
    def test_batch_process_approvals(self):
        """
        Test batch processing of approvals.
        """
        # Test parameters
        approvals = [
            {
                "approval_id": "batch_1",
                "approved": True,
                "image_path": self.test_images[0]
            },
            {
                "approval_id": "batch_2",
                "approved": False,
                "image_path": self.test_images[1]
            },
            {
                "approval_id": "batch_3",
                "approved": True,
                "image_path": self.test_images[2]
            }
        ]
        
        # Call the method
        results = self.approval_tool.batch_process_approvals(approvals)
        
        # Verify the results
        self.assertEqual(len(results), len(approvals))
        
        # Check approved images
        approved_images = self.approval_tool.get_approved_images()
        self.assertEqual(len(approved_images), 2)  # Two images were approved
    
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
