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

# Import components
from src.ai.venice_api import VeniceImageGenerator
from src.ai.sam_segmentation import SAMSegmenter
from src.models.threestudio_generator import ThreeStudioGenerator
from src.openscad_wrapper.wrapper import OpenSCADWrapper
from src.workflow.image_to_model_pipeline import ImageToModelPipeline
from src.config import (
    VENICE_API_KEY, IMAGES_DIR, MASKS_DIR, MODELS_DIR, SCAD_DIR,
    SAM2_CHECKPOINT_PATH, SAM2_MODEL_TYPE, SAM2_USE_GPU, THREESTUDIO_PATH
)

def test_pipeline(prompt: str, output_dir: Optional[str] = None, 
                 venice_model: str = "fluently-xl", skip_steps: List[str] = None):
    """
    Test the full image-to-model pipeline.
    
    Args:
        prompt: Text prompt for image generation
        output_dir: Directory to save pipeline results
        venice_model: Venice.ai model to use for image generation
        skip_steps: List of steps to skip ('image', 'segment', 'model3d', 'openscad')
    """
    # Use default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pipeline_test")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize skip_steps if None
    skip_steps = skip_steps or []
    
    logger.info(f"Testing image-to-model pipeline with prompt: {prompt}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Venice model: {venice_model}")
    logger.info(f"Skipping steps: {skip_steps}")
    
    try:
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # Venice.ai image generator
        venice_generator = VeniceImageGenerator(
            api_key=VENICE_API_KEY,
            output_dir=os.path.join(output_dir, "images")
        )
        
        # SAM2 segmenter
        sam_segmenter = SAMSegmenter(
            model_type=SAM2_MODEL_TYPE,
            checkpoint_path=SAM2_CHECKPOINT_PATH,
            use_gpu=SAM2_USE_GPU,
            output_dir=os.path.join(output_dir, "masks")
        )
        
        # ThreeStudio generator
        threestudio_generator = ThreeStudioGenerator(
            threestudio_path=THREESTUDIO_PATH,
            output_dir=os.path.join(output_dir, "models")
        )
        
        # OpenSCAD wrapper
        openscad_wrapper = OpenSCADWrapper(
            output_dir=os.path.join(output_dir, "scad")
        )
        
        # Initialize pipeline
        pipeline = ImageToModelPipeline(
            venice_generator=venice_generator,
            sam_segmenter=sam_segmenter,
            threestudio_generator=threestudio_generator,
            openscad_wrapper=openscad_wrapper,
            output_dir=output_dir
        )
        
        # Run pipeline with custom steps
        if 'image' in skip_steps:
            # Skip image generation, use a test image
            logger.info("Skipping image generation, using test image")
            image_path = os.path.join(IMAGES_DIR, "test_image.png")
            if not os.path.exists(image_path):
                logger.error(f"Test image not found: {image_path}")
                return
            
            # TODO: Implement custom pipeline execution with skipped steps
            logger.info("Custom pipeline execution not implemented yet")
            return
        else:
            # Run full pipeline
            logger.info("Running full pipeline...")
            result = pipeline.generate_model_from_text(
                prompt=prompt,
                venice_params={"model": venice_model},
                sam_params={},
                threestudio_params={}
            )
            
            # Print results
            logger.info("Pipeline completed successfully")
            logger.info(f"Pipeline ID: {result.get('pipeline_id')}")
            logger.info(f"Image path: {result.get('image', {}).get('local_path')}")
            logger.info(f"Mask count: {result.get('segmentation', {}).get('num_masks', 0)}")
            logger.info(f"3D model path: {result.get('model_3d', {}).get('exported_files', [])}")
            logger.info(f"OpenSCAD file: {result.get('openscad', {}).get('scad_file')}")
            
            return result
    
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test image-to-model pipeline")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output-dir", help="Directory to save pipeline results")
    parser.add_argument("--venice-model", default="fluently-xl", help="Venice.ai model to use")
    parser.add_argument("--skip", nargs="+", choices=["image", "segment", "model3d", "openscad"],
                        help="Steps to skip in the pipeline")
    
    args = parser.parse_args()
    
    # Run test
    test_pipeline(
        args.prompt,
        args.output_dir,
        args.venice_model,
        args.skip
    )
