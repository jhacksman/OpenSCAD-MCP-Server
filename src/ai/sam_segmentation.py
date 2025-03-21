"""
SAM2 (Segment Anything Model 2) integration for object segmentation.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SAMSegmenter:
    """
    Wrapper for Segment Anything Model 2 (SAM2) for object segmentation.
    """
    
    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None,
                use_gpu: bool = True, output_dir: str = "output/masks"):
        """
        Initialize the SAM segmenter.
        
        Args:
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to model checkpoint
            use_gpu: Whether to use GPU for inference
            output_dir: Directory to store segmentation results
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.use_gpu = use_gpu
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Model will be initialized on first use to avoid loading it unnecessarily
        self.model = None
        self.predictor = None
    
    def _initialize_model(self) -> None:
        """
        Initialize the SAM model.
        
        Note: This requires PyTorch and the segment-anything package to be installed.
        """
        try:
            # Import here to avoid dependency issues if SAM is not installed
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            if not self.checkpoint_path:
                raise ValueError("SAM checkpoint path is required")
            
            # Check if checkpoint exists
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found at {self.checkpoint_path}")
            
            # Determine device
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            
            # Load SAM model
            self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.model.to(device=device)
            self.predictor = SamPredictor(self.model)
            
            logger.info(f"Initialized SAM model ({self.model_type}) on {device}")
        except ImportError as e:
            logger.error(f"Required packages not installed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing SAM model: {str(e)}")
            raise
    
    def segment_image(self, image_path: str, points: Optional[List[Tuple[int, int]]] = None,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Segment objects in an image.
        
        Args:
            image_path: Path to input image
            points: Optional list of (x, y) points to guide segmentation
            output_dir: Optional directory to save segmentation results
            
        Returns:
            Dictionary containing segmentation masks and metadata
        """
        # Initialize model if not already initialized
        if self.model is None:
            self._initialize_model()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image in predictor
            self.predictor.set_image(image)
            
            # Generate masks
            if points:
                # Convert points to numpy arrays
                import numpy as np
                point_coords = np.array(points)
                point_labels = np.ones(len(points))
                
                # Generate masks from points
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
            else:
                # Automatic segmentation (using center point)
                h, w = image.shape[:2]
                center_point = np.array([[w//2, h//2]])
                center_label = np.array([1])
                
                masks, scores, logits = self.predictor.predict(
                    point_coords=center_point,
                    point_labels=center_label,
                    multimask_output=True
                )
            
            # Use provided output directory or default
            output_dir = output_dir or os.path.join(self.output_dir, Path(image_path).stem)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process results
            masked_images = []
            for i, mask in enumerate(masks):
                # Apply mask to image
                masked_image = self._apply_mask_to_image(image, mask)
                
                # Save masked image
                output_path = os.path.join(output_dir, f"mask_{i}.png")
                cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
                
                masked_images.append(output_path)
            
            # Convert numpy arrays to lists for JSON serialization
            result = {
                "image_path": image_path,
                "masked_images": masked_images,
                "scores": scores.tolist(),
                "mask_count": len(masks)
            }
            
            return result
        except Exception as e:
            logger.error(f"Error segmenting image: {str(e)}")
            raise
    
    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to image, keeping only the masked region.
        
        Args:
            image: Input image as numpy array
            mask: Binary mask as numpy array
            
        Returns:
            Masked image as numpy array
        """
        # Create a copy of the image
        masked_image = image.copy()
        
        # Apply mask
        masked_image[~mask] = [0, 0, 0]  # Set background to black
        
        return masked_image
    
    def segment_with_auto_points(self, image_path: str, num_points: int = 5,
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Segment image using automatically generated points.
        
        Args:
            image_path: Path to input image
            num_points: Number of points to generate
            output_dir: Optional directory to save segmentation results
            
        Returns:
            Dictionary containing segmentation masks and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        h, w = image.shape[:2]
        
        # Generate points in a grid pattern
        points = []
        rows = int(np.sqrt(num_points))
        cols = num_points // rows
        
        for i in range(rows):
            for j in range(cols):
                x = int(w * (j + 0.5) / cols)
                y = int(h * (i + 0.5) / rows)
                points.append((x, y))
        
        # Segment with generated points
        return self.segment_image(image_path, points, output_dir)
