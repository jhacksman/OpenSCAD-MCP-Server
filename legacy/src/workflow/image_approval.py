"""
Image approval tool for MCP clients.
"""

import os
import logging
import shutil
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ImageApprovalTool:
    """
    Tool for image approval/denial in MCP clients.
    """
    
    def __init__(self, output_dir: str = "output/approved_images"):
        """
        Initialize the image approval tool.
        
        Args:
            output_dir: Directory to store approved images
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def present_image_for_approval(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Present an image to the user for approval.
        
        Args:
            image_path: Path to the image
            metadata: Optional metadata about the image
            
        Returns:
            Dictionary with image path and approval request ID
        """
        # For MCP server, we just prepare the response
        # The actual approval is handled by the client
        
        approval_id = os.path.basename(image_path).split('.')[0]
        
        return {
            "approval_id": approval_id,
            "image_path": image_path,
            "image_url": f"/images/{os.path.basename(image_path)}",
            "metadata": metadata or {}
        }
    
    def process_approval(self, approval_id: str, approved: bool, image_path: str) -> Dict[str, Any]:
        """
        Process user's approval or denial of an image.
        
        Args:
            approval_id: ID of the approval request
            approved: Whether the image was approved
            image_path: Path to the image
            
        Returns:
            Dictionary with approval status and image path
        """
        if approved:
            # Copy approved image to output directory
            approved_path = os.path.join(self.output_dir, os.path.basename(image_path))
            os.makedirs(os.path.dirname(approved_path), exist_ok=True)
            shutil.copy2(image_path, approved_path)
            
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
    
    def get_approved_images(self, filter_pattern: Optional[str] = None) -> List[str]:
        """
        Get list of approved images.
        
        Args:
            filter_pattern: Optional pattern to filter image names
            
        Returns:
            List of paths to approved images
        """
        import glob
        
        if filter_pattern:
            pattern = os.path.join(self.output_dir, filter_pattern)
        else:
            pattern = os.path.join(self.output_dir, "*")
        
        return glob.glob(pattern)
    
    def get_approval_status(self, approval_id: str) -> Dict[str, Any]:
        """
        Get the approval status for a specific approval ID.
        
        Args:
            approval_id: ID of the approval request
            
        Returns:
            Dictionary with approval status
        """
        # Check if any approved image matches the approval ID
        approved_images = self.get_approved_images()
        
        for image_path in approved_images:
            if approval_id in os.path.basename(image_path):
                return {
                    "approval_id": approval_id,
                    "approved": True,
                    "approved_path": image_path
                }
        
        return {
            "approval_id": approval_id,
            "approved": False
        }
    
    def batch_process_approvals(self, approvals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple approvals at once.
        
        Args:
            approvals: List of dictionaries with approval_id, approved, and image_path
            
        Returns:
            List of dictionaries with approval results
        """
        results = []
        
        for approval in approvals:
            result = self.process_approval(
                approval_id=approval["approval_id"],
                approved=approval["approved"],
                image_path=approval["image_path"]
            )
            results.append(result)
        
        return results
