"""
Test script for the Venice.ai to Google Gemini pipeline.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.venice_api import VeniceImageGenerator
from src.ai.gemini_api import GeminiImageGenerator
from src.utils.image_comparison import MultiViewConsistencyEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVeniceToGooglePipeline(unittest.TestCase):
    """
    Test cases for the Venice.ai to Google Gemini pipeline.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test output directories
        self.test_output_dir = "output/test_venice_to_google"
        self.venice_output_dir = os.path.join(self.test_output_dir, "venice")
        self.gemini_output_dir = os.path.join(self.test_output_dir, "gemini")
        self.comparison_output_dir = os.path.join(self.test_output_dir, "comparison")
        
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.venice_output_dir, exist_ok=True)
        os.makedirs(self.gemini_output_dir, exist_ok=True)
        os.makedirs(self.comparison_output_dir, exist_ok=True)
        
        # Get API keys from environment
        self.venice_api_key = os.environ.get("VENICE_API_KEY", "test_venice_key")
        self.gemini_api_key = os.environ.get("google", "test_gemini_key")
        
        # Create the generators
        self.venice_generator = VeniceImageGenerator(
            api_key=self.venice_api_key,
            output_dir=self.venice_output_dir
        )
        
        self.gemini_generator = GeminiImageGenerator(
            api_key=self.gemini_api_key,
            output_dir=self.gemini_output_dir
        )
        
        # Create the consistency evaluator
        self.evaluator = MultiViewConsistencyEvaluator(
            output_dir=self.comparison_output_dir
        )
    
    @patch('src.ai.venice_api.VeniceImageGenerator.generate_image')
    @patch('src.ai.gemini_api.GeminiImageGenerator.generate_multiple_views')
    def test_venice_to_google_pipeline(self, mock_gemini_views, mock_venice_generate):
        """
        Test the pipeline from Venice.ai to Google Gemini.
        
        1. Generate an initial image with Venice.ai
        2. Use that image as input to generate multiple views with Google Gemini
        3. Evaluate consistency between the views
        """
        # Mock Venice.ai image generation
        def mock_venice_side_effect(prompt, model=None, **kwargs):
            # Create a test image
            from PIL import Image, ImageDraw
            image_path = os.path.join(self.venice_output_dir, "venice_source.png")
            
            img = Image.new('RGB', (512, 512), color=(50, 100, 150))
            draw = ImageDraw.Draw(img)
            draw.rectangle((150, 150, 350, 350), fill=(200, 100, 50))
            img.save(image_path)
            
            return {
                "prompt": prompt,
                "model": model or "flux-dev",
                "local_path": image_path
            }
        
        mock_venice_generate.side_effect = mock_venice_side_effect
        
        # Mock Google Gemini multiple views generation
        def mock_gemini_views_side_effect(prompt, num_views=3, source_image=None, **kwargs):
            results = []
            
            # Create test images with variations based on the source image
            from PIL import Image, ImageDraw
            
            # Load source image to extract colors if provided
            base_color = (60, 110, 160)
            fill_color = (210, 110, 60)
            
            if source_image:
                try:
                    src_img = Image.open(source_image)
                    # Extract colors from source image (simplified)
                    base_color = (50, 100, 150)  # Match Venice image
                    fill_color = (200, 100, 50)  # Match Venice image
                except:
                    logger.warning(f"Could not load source image: {source_image}")
            
            for i in range(num_views):
                image_path = os.path.join(self.gemini_output_dir, f"gemini_view_{i}.png")
                
                img = Image.new('RGB', (512, 512), color=base_color)
                draw = ImageDraw.Draw(img)
                
                # Draw a shape with variations based on view angle
                # For front view
                if i == 0:
                    draw.rectangle((150, 150, 350, 350), fill=fill_color)
                # For side view
                elif i == 1:
                    draw.rectangle((200, 150, 400, 350), fill=fill_color)
                # For top view
                else:
                    draw.ellipse((150, 150, 350, 350), fill=fill_color)
                
                img.save(image_path)
                
                results.append({
                    "prompt": prompt,
                    "local_path": image_path,
                    "view_index": i + 1,
                    "view_direction": f"view_{i+1}"
                })
            
            return results
        
        mock_gemini_views.side_effect = mock_gemini_views_side_effect
        
        # Test parameters
        prompt = "A low-poly rabbit with black background"
        
        # Step 1: Generate initial image with Venice.ai
        venice_result = self.venice_generator.generate_image(prompt)
        venice_image_path = venice_result["local_path"]
        
        logger.info(f"Generated Venice.ai image: {venice_image_path}")
        
        # Step 2: Generate multiple views with Google Gemini using the Venice image
        gemini_results = self.gemini_generator.generate_multiple_views(
            prompt, 
            num_views=3,
            source_image=venice_image_path
        )
        
        gemini_image_paths = [result["local_path"] for result in gemini_results]
        
        logger.info(f"Generated Google Gemini views: {gemini_image_paths}")
        
        # Step 3: Evaluate consistency between all images (Venice + Gemini)
        all_images = [venice_image_path] + gemini_image_paths
        consistency_results = self.evaluator.evaluate_view_consistency(all_images)
        
        # Save the evaluation report
        report_path = self.evaluator.save_evaluation_report(
            consistency_results,
            report_name="venice_to_google_pipeline.json"
        )
        
        # Log the results
        logger.info(f"Pipeline consistency score: {consistency_results['overall_consistency_score']}")
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Verify the consistency score is reasonable
        self.assertGreaterEqual(consistency_results['overall_consistency_score'], 0.5)
    
    @patch('src.ai.venice_api.VeniceImageGenerator.generate_image')
    @patch('src.ai.gemini_api.GeminiImageGenerator.generate_multiple_views')
    def test_prompt_engineering_for_pipeline(self, mock_gemini_views, mock_venice_generate):
        """
        Test different prompt engineering techniques for the Venice to Google pipeline.
        """
        # Mock Venice.ai image generation (same for all prompts)
        def mock_venice_side_effect(prompt, model=None, **kwargs):
            # Create a test image
            from PIL import Image, ImageDraw
            
            # Use a different filename for each prompt to avoid overwriting
            prompt_hash = hash(prompt) % 10000
            image_path = os.path.join(self.venice_output_dir, f"venice_source_{prompt_hash}.png")
            
            img = Image.new('RGB', (512, 512), color=(50, 100, 150))
            draw = ImageDraw.Draw(img)
            draw.rectangle((150, 150, 350, 350), fill=(200, 100, 50))
            img.save(image_path)
            
            return {
                "prompt": prompt,
                "model": model or "flux-dev",
                "local_path": image_path
            }
        
        mock_venice_generate.side_effect = mock_venice_side_effect
        
        # Mock Google Gemini multiple views generation
        def mock_gemini_views_side_effect(prompt, num_views=3, source_image=None, **kwargs):
            results = []
            
            # Create test images with variations based on the prompt
            from PIL import Image, ImageDraw
            
            # Base color from source image
            base_color = (50, 100, 150)
            fill_color = (200, 100, 50)
            
            # Adjust consistency based on prompt content
            consistency_factor = 0.5  # Default
            
            if "consistent style" in prompt.lower():
                consistency_factor = 0.8
            if "preserve details" in prompt.lower():
                consistency_factor = 0.9
            if "same object" in prompt.lower():
                consistency_factor = 0.95
            
            # Use a different filename prefix for each prompt
            prompt_hash = hash(prompt) % 10000
            
            for i in range(num_views):
                image_path = os.path.join(self.gemini_output_dir, f"gemini_view_{prompt_hash}_{i}.png")
                
                img = Image.new('RGB', (512, 512), color=base_color)
                draw = ImageDraw.Draw(img)
                
                # Variation factor decreases with higher consistency
                variation = (1.0 - consistency_factor) * 50
                
                # Draw a shape with variations based on view angle and consistency factor
                if i == 0:
                    draw.rectangle((150, 150, 350, 350), fill=fill_color)
                elif i == 1:
                    draw.rectangle((150 + variation, 150, 350 + variation, 350), fill=fill_color)
                else:
                    draw.rectangle((150, 150 + variation, 350, 350 + variation), fill=fill_color)
                
                img.save(image_path)
                
                results.append({
                    "prompt": prompt,
                    "local_path": image_path,
                    "view_index": i + 1,
                    "view_direction": f"view_{i+1}"
                })
            
            return results
        
        mock_gemini_views.side_effect = mock_gemini_views_side_effect
        
        # Test different prompt techniques
        prompt_techniques = [
            "A low-poly rabbit",  # Basic prompt
            "A low-poly rabbit with consistent style and colors",  # Style anchoring
            "A low-poly rabbit, preserve details across all views",  # Detail preservation
            "A low-poly rabbit, same object from different angles",  # Object identity
        ]
        
        all_results = {}
        
        for prompt in prompt_techniques:
            # Step 1: Generate initial image with Venice.ai
            venice_result = self.venice_generator.generate_image(prompt)
            venice_image_path = venice_result["local_path"]
            
            # Step 2: Generate multiple views with Google Gemini
            gemini_results = self.gemini_generator.generate_multiple_views(
                prompt, 
                num_views=3,
                source_image=venice_image_path
            )
            
            gemini_image_paths = [result["local_path"] for result in gemini_results]
            
            # Step 3: Evaluate consistency
            all_images = [venice_image_path] + gemini_image_paths
            consistency_results = self.evaluator.evaluate_view_consistency(all_images)
            
            # Save results for this prompt
            all_results[prompt] = {
                "venice_image": venice_image_path,
                "gemini_images": gemini_image_paths,
                "consistency_score": consistency_results["overall_consistency_score"]
            }
            
            # Log the results
            logger.info(f"Prompt: '{prompt}'")
            logger.info(f"Consistency score: {consistency_results['overall_consistency_score']}")
        
        # Find the best prompt technique
        best_prompt = max(all_results.keys(), key=lambda p: all_results[p]["consistency_score"])
        best_score = all_results[best_prompt]["consistency_score"]
        
        logger.info(f"Best prompt technique: '{best_prompt}' with score {best_score}")
        
        # Save the overall results
        import json
        with open(os.path.join(self.comparison_output_dir, "prompt_techniques.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
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
