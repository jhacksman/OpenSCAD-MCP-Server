"""
Test script for the image approval workflow.
"""

import os
import sys
import json
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow.image_approval import ImageApprovalManager
from src.config import IMAGE_APPROVAL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestImageApprovalWorkflow(unittest.TestCase):
    """
    Test cases for the image approval workflow.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test output directories
        self.test_output_dir = "output/test_image_approval"
        self.test_approved_dir = os.path.join(self.test_output_dir, "approved")
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_approved_dir, exist_ok=True)
        
        # Create the approval manager
        self.approval_manager = ImageApprovalManager(
            output_dir=self.test_output_dir,
            approved_dir=self.test_approved_dir,
            min_approved_images=3,
            auto_approve=False
        )
        
        # Create test images
        self.test_images = []
        for i in range(5):
            image_path = os.path.join(self.test_output_dir, f"test_image_{i}.png")
            with open(image_path, "w") as f:
                f.write(f"test image data {i}")
            self.test_images.append({
                "id": f"image_{i}",
                "local_path": image_path,
                "view_index": i,
                "view_direction": f"view_{i}"
            })
    
    def test_add_images(self):
        """
        Test adding images to the approval manager.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Verify images were added
        self.assertEqual(len(self.approval_manager.images), 5)
        self.assertEqual(len(self.approval_manager.pending_images), 5)
        self.assertEqual(len(self.approval_manager.approved_images), 0)
        self.assertEqual(len(self.approval_manager.rejected_images), 0)
    
    def test_approve_image(self):
        """
        Test approving an image.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Approve an image
        result = self.approval_manager.approve_image("image_0")
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["image_id"], "image_0")
        self.assertEqual(result["status"], "approved")
        
        # Verify the image was moved to approved
        self.assertEqual(len(self.approval_manager.pending_images), 4)
        self.assertEqual(len(self.approval_manager.approved_images), 1)
        self.assertEqual(len(self.approval_manager.rejected_images), 0)
        
        # Verify the image was copied to the approved directory
        approved_path = os.path.join(self.test_approved_dir, "image_0.png")
        self.assertTrue(os.path.exists(approved_path))
    
    def test_reject_image(self):
        """
        Test rejecting an image.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Reject an image
        result = self.approval_manager.reject_image("image_1")
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["image_id"], "image_1")
        self.assertEqual(result["status"], "rejected")
        
        # Verify the image was moved to rejected
        self.assertEqual(len(self.approval_manager.pending_images), 4)
        self.assertEqual(len(self.approval_manager.approved_images), 0)
        self.assertEqual(len(self.approval_manager.rejected_images), 1)
    
    def test_get_approval_status(self):
        """
        Test getting the approval status.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Approve some images
        self.approval_manager.approve_image("image_0")
        self.approval_manager.approve_image("image_1")
        self.approval_manager.approve_image("image_2")
        
        # Reject an image
        self.approval_manager.reject_image("image_3")
        
        # Get the status
        status = self.approval_manager.get_status()
        
        # Verify the status
        self.assertEqual(status["total_images"], 5)
        self.assertEqual(status["pending_count"], 1)
        self.assertEqual(status["approved_count"], 3)
        self.assertEqual(status["rejected_count"], 1)
        self.assertTrue(status["has_minimum_approved"])
        self.assertEqual(len(status["approved_images"]), 3)
        self.assertEqual(len(status["pending_images"]), 1)
        self.assertEqual(len(status["rejected_images"]), 1)
    
    def test_get_approved_images(self):
        """
        Test getting approved images.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Approve some images
        self.approval_manager.approve_image("image_0")
        self.approval_manager.approve_image("image_2")
        self.approval_manager.approve_image("image_4")
        
        # Get approved images
        approved = self.approval_manager.get_approved_images()
        
        # Verify approved images
        self.assertEqual(len(approved), 3)
        self.assertEqual(approved[0]["id"], "image_0")
        self.assertEqual(approved[1]["id"], "image_2")
        self.assertEqual(approved[2]["id"], "image_4")
    
    def test_auto_approve(self):
        """
        Test auto-approval mode.
        """
        # Create an auto-approve manager
        auto_manager = ImageApprovalManager(
            output_dir=self.test_output_dir,
            approved_dir=self.test_approved_dir,
            min_approved_images=3,
            auto_approve=True
        )
        
        # Add images
        auto_manager.add_images(self.test_images)
        
        # Verify all images were auto-approved
        self.assertEqual(len(auto_manager.pending_images), 0)
        self.assertEqual(len(auto_manager.approved_images), 5)
        self.assertEqual(len(auto_manager.rejected_images), 0)
    
    def test_has_minimum_approved(self):
        """
        Test checking if minimum approved images are met.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Initially should not have minimum
        self.assertFalse(self.approval_manager.has_minimum_approved())
        
        # Approve two images
        self.approval_manager.approve_image("image_0")
        self.approval_manager.approve_image("image_1")
        
        # Still should not have minimum
        self.assertFalse(self.approval_manager.has_minimum_approved())
        
        # Approve one more image
        self.approval_manager.approve_image("image_2")
        
        # Now should have minimum
        self.assertTrue(self.approval_manager.has_minimum_approved())
    
    def test_save_and_load_state(self):
        """
        Test saving and loading the approval state.
        """
        # Add images
        self.approval_manager.add_images(self.test_images)
        
        # Approve and reject some images
        self.approval_manager.approve_image("image_0")
        self.approval_manager.approve_image("image_2")
        self.approval_manager.reject_image("image_3")
        
        # Save the state
        state_file = os.path.join(self.test_output_dir, "approval_state.json")
        self.approval_manager.save_state(state_file)
        
        # Create a new manager
        new_manager = ImageApprovalManager(
            output_dir=self.test_output_dir,
            approved_dir=self.test_approved_dir,
            min_approved_images=3,
            auto_approve=False
        )
        
        # Load the state
        new_manager.load_state(state_file)
        
        # Verify the state was loaded correctly
        self.assertEqual(len(new_manager.images), 5)
        self.assertEqual(len(new_manager.pending_images), 2)
        self.assertEqual(len(new_manager.approved_images), 2)
        self.assertEqual(len(new_manager.rejected_images), 1)
    
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
