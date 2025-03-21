import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SAM2 segmenter and config
from src.ai.sam_segmentation import SAMSegmenter
from src.config import SAM2_CHECKPOINT_PATH, SAM2_MODEL_TYPE, SAM2_USE_GPU, MASKS_DIR

def test_sam2_segmentation(image_path: str, output_dir: Optional[str] = None, use_auto_points: bool = True):
    """
    Test SAM2 segmentation on an image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save segmentation results (default: config.MASKS_DIR)
        use_auto_points: Whether to use automatic point generation
    """
    # Validate image path
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return
    
    # Use default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(MASKS_DIR, Path(image_path).stem)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Testing SAM2 segmentation on image: {image_path}")
    logger.info(f"Model type: {SAM2_MODEL_TYPE}")
    logger.info(f"Checkpoint path: {SAM2_CHECKPOINT_PATH}")
    logger.info(f"Using GPU: {SAM2_USE_GPU}")
    
    try:
        # Initialize SAM2 segmenter
        logger.info("Initializing SAM2 segmenter...")
        sam_segmenter = SAMSegmenter(
            model_type=SAM2_MODEL_TYPE,
            checkpoint_path=SAM2_CHECKPOINT_PATH,
            use_gpu=SAM2_USE_GPU,
            output_dir=output_dir
        )
        
        # Perform segmentation
        if use_auto_points:
            logger.info("Using automatic point generation")
            result = sam_segmenter.segment_with_auto_points(image_path)
        else:
            # Use center point of the image for manual point
            logger.info("Using manual center point")
            import cv2
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            center_point = (w // 2, h // 2)
            result = sam_segmenter.segment_image(image_path, points=[center_point])
        
        # Print results
        logger.info(f"Segmentation completed with {result.get('num_masks', 0)} masks")
        
        if result.get('mask_paths'):
            logger.info(f"Mask paths: {result.get('mask_paths')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in SAM2 segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test SAM2 segmentation")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output-dir", help="Directory to save segmentation results")
    parser.add_argument("--manual-points", action="store_true", help="Use manual center point instead of auto points")
    
    args = parser.parse_args()
    
    # Run test
    test_sam2_segmentation(
        args.image_path,
        args.output_dir,
        not args.manual_points
    )
