"""
Test script for image consistency evaluation.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.image_comparison import ImageComparisonMetrics, MultiViewConsistencyEvaluator
from src.ai.venice_api import VeniceImageGenerator
from src.ai.gemini_api import GeminiImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestImageConsistency(unittest.TestCase):
    """
    Test cases for image consistency evaluation.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test output directories
        self.test_output_dir = "output/test_consistency"
        self.venice_output_dir = os.path.join(self.test_output_dir, "venice")
        self.gemini_output_dir = os.path.join(self.test_output_dir, "gemini")
        self.comparison_output_dir = os.path.join(self.test_output_dir, "comparison")
        
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.venice_output_dir, exist_ok=True)
        os.makedirs(self.gemini_output_dir, exist_ok=True)
        os.makedirs(self.comparison_output_dir, exist_ok=True)
        
        # Create the comparison metrics
        self.metrics = ImageComparisonMetrics()
        
        # Create the consistency evaluator
        self.evaluator = MultiViewConsistencyEvaluator(
            output_dir=self.comparison_output_dir
        )
        
        # Create mock image generators
        self.venice_api_key = "test_venice_key"
        self.gemini_api_key = "test_gemini_key"
        
        self.venice_generator = VeniceImageGenerator(
            api_key=self.venice_api_key,
            output_dir=self.venice_output_dir
        )
        
        self.gemini_generator = GeminiImageGenerator(
            api_key=self.gemini_api_key,
            output_dir=self.gemini_output_dir
        )
        
        # Create test images
        self.create_test_images()
    
    def create_test_images(self):
        """
        Create test images for consistency testing.
        """
        # Create sample images for testing
        from PIL import Image, ImageDraw
        
        # Create Venice.ai test images (similar but with variations)
        self.venice_images = []
        for i in range(3):
            image_path = os.path.join(self.venice_output_dir, f"venice_view_{i}.png")
            img = Image.new('RGB', (512, 512), color=(50, 100, 150))
            draw = ImageDraw.Draw(img)
            
            # Draw a shape that varies slightly by view
            draw.rectangle(
                (100 + i*20, 100 + i*10, 400 - i*20, 400 - i*10), 
                fill=(200, 100, 50)
            )
            
            img.save(image_path)
            self.venice_images.append(image_path)
        
        # Create Google Gemini test images (similar but with variations)
        self.gemini_images = []
        for i in range(3):
            image_path = os.path.join(self.gemini_output_dir, f"gemini_view_{i}.png")
            img = Image.new('RGB', (512, 512), color=(60, 110, 160))
            draw = ImageDraw.Draw(img)
            
            # Draw a shape that varies slightly by view
            draw.rectangle(
                (110 + i*20, 110 + i*10, 410 - i*20, 410 - i*10), 
                fill=(210, 110, 60)
            )
            
            img.save(image_path)
            self.gemini_images.append(image_path)
    
    def test_ssim_metric(self):
        """
        Test the SSIM metric for image comparison.
        """
        # Compare first two Venice images
        ssim_score = self.metrics.compute_ssim(self.venice_images[0], self.venice_images[1])
        
        # SSIM should be between 0 and 1
        self.assertGreaterEqual(ssim_score, 0.0)
        self.assertLessEqual(ssim_score, 1.0)
        
        # Similar images should have high SSIM
        self.assertGreater(ssim_score, 0.7)
        
        # Log the score
        logger.info(f"SSIM score between Venice images 0 and 1: {ssim_score}")
    
    def test_histogram_similarity(self):
        """
        Test the histogram similarity metric.
        """
        # Compare first two Venice images
        hist_score = self.metrics.compute_histogram_similarity(self.venice_images[0], self.venice_images[1])
        
        # Histogram similarity should be between 0 and 1
        self.assertGreaterEqual(hist_score, 0.0)
        self.assertLessEqual(hist_score, 1.0)
        
        # Similar images should have high histogram similarity
        self.assertGreater(hist_score, 0.7)
        
        # Log the score
        logger.info(f"Histogram similarity between Venice images 0 and 1: {hist_score}")
    
    def test_color_palette_similarity(self):
        """
        Test the color palette similarity metric.
        """
        # Compare first two Venice images
        palette_score = self.metrics.compute_color_palette_similarity(self.venice_images[0], self.venice_images[1])
        
        # Color palette similarity should be between 0 and 1
        self.assertGreaterEqual(palette_score, 0.0)
        self.assertLessEqual(palette_score, 1.0)
        
        # Similar images should have high color palette similarity
        self.assertGreater(palette_score, 0.7)
        
        # Log the score
        logger.info(f"Color palette similarity between Venice images 0 and 1: {palette_score}")
    
    def test_all_metrics(self):
        """
        Test computing all metrics at once.
        """
        # Compare first two Venice images
        all_metrics = self.metrics.compute_all_metrics(self.venice_images[0], self.venice_images[1])
        
        # Should have all expected metrics
        self.assertIn("ssim", all_metrics)
        self.assertIn("histogram_similarity", all_metrics)
        self.assertIn("color_palette_similarity", all_metrics)
        self.assertIn("average_similarity", all_metrics)
        
        # Log the metrics
        logger.info(f"All metrics between Venice images 0 and 1: {all_metrics}")
    
    def test_view_consistency_evaluation(self):
        """
        Test evaluating consistency across multiple views.
        """
        # Evaluate Venice images consistency
        venice_results = self.evaluator.evaluate_view_consistency(self.venice_images)
        
        # Should have expected result structure
        self.assertIn("image_count", venice_results)
        self.assertIn("pairwise_metrics", venice_results)
        self.assertIn("average_metrics", venice_results)
        self.assertIn("overall_consistency_score", venice_results)
        
        # Should have correct image count
        self.assertEqual(venice_results["image_count"], len(self.venice_images))
        
        # Should have pairwise comparisons
        expected_pairs = (len(self.venice_images) * (len(self.venice_images) - 1)) // 2
        self.assertEqual(len(venice_results["pairwise_metrics"]), expected_pairs)
        
        # Overall score should be between 0 and 1
        self.assertGreaterEqual(venice_results["overall_consistency_score"], 0.0)
        self.assertLessEqual(venice_results["overall_consistency_score"], 1.0)
        
        # Log the results
        logger.info(f"Venice images consistency score: {venice_results['overall_consistency_score']}")
    
    def test_compare_generation_methods(self):
        """
        Test comparing Venice.ai and Google Gemini generation methods.
        """
        # Compare Venice and Gemini images
        comparison_results = self.evaluator.compare_generation_methods(
            self.venice_images, 
            self.gemini_images
        )
        
        # Should have expected result structure
        self.assertIn("venice_consistency", comparison_results)
        self.assertIn("google_consistency", comparison_results)
        self.assertIn("cross_platform_similarity", comparison_results)
        self.assertIn("average_cross_platform_similarity", comparison_results)
        
        # Should have cross-platform comparisons
        self.assertEqual(len(comparison_results["cross_platform_similarity"]), min(len(self.venice_images), len(self.gemini_images)))
        
        # Save the comparison report
        report_path = self.evaluator.save_evaluation_report(comparison_results)
        self.assertTrue(os.path.exists(report_path))
        
        # Log the results
        logger.info(f"Cross-platform similarity: {comparison_results['average_cross_platform_similarity']}")
    
    @patch('src.ai.venice_api.VeniceImageGenerator.generate_image')
    @patch('src.ai.gemini_api.GeminiImageGenerator.generate_image')
    def test_real_image_generation_consistency(self, mock_gemini_generate, mock_venice_generate):
        """
        Test consistency with mocked image generation.
        """
        # Mock Venice.ai image generation
        def mock_venice_side_effect(prompt, model=None, **kwargs):
            idx = kwargs.get('idx', 0)
            image_path = os.path.join(self.venice_output_dir, f"venice_generated_{idx}.png")
            
            # Create a test image
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color=(50, 100, 150))
            draw = ImageDraw.Draw(img)
            draw.rectangle((100, 100, 400, 400), fill=(200, 100, 50))
            img.save(image_path)
            
            return {
                "prompt": prompt,
                "model": model or "flux-dev",
                "local_path": image_path
            }
        
        mock_venice_generate.side_effect = mock_venice_side_effect
        
        # Mock Google Gemini image generation
        def mock_gemini_side_effect(prompt, model=None, **kwargs):
            idx = kwargs.get('idx', 0)
            image_path = os.path.join(self.gemini_output_dir, f"gemini_generated_{idx}.png")
            
            # Create a test image
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color=(60, 110, 160))
            draw = ImageDraw.Draw(img)
            draw.rectangle((110, 110, 410, 410), fill=(210, 110, 60))
            img.save(image_path)
            
            return {
                "prompt": prompt,
                "model": model or "gemini-2.0-flash-exp-image-generation",
                "local_path": image_path
            }
        
        mock_gemini_generate.side_effect = mock_gemini_side_effect
        
        # Generate test images
        prompt = "A low-poly rabbit with black background"
        
        # Generate Venice images
        venice_results = []
        for i in range(3):
            result = self.venice_generator.generate_image(prompt, idx=i)
            venice_results.append(result["local_path"])
        
        # Generate Gemini images
        gemini_results = []
        for i in range(3):
            result = self.gemini_generator.generate_image(prompt, idx=i)
            gemini_results.append(result["local_path"])
        
        # Evaluate consistency
        comparison_results = self.evaluator.compare_generation_methods(
            venice_results, 
            gemini_results
        )
        
        # Save the comparison report
        report_path = self.evaluator.save_evaluation_report(
            comparison_results, 
            report_name="mock_generation_comparison.json"
        )
        
        # Log the results
        logger.info(f"Mock generation comparison saved to: {report_path}")
    
    def test_venice_to_gemini_pipeline(self):
        """
        Test the Venice.ai to Google Gemini pipeline consistency.
        """
        # Create a mock Venice image
        venice_image_path = os.path.join(self.venice_output_dir, "venice_source.png")
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (512, 512), color=(50, 100, 150))
        draw = ImageDraw.Draw(img)
        draw.rectangle((100, 100, 400, 400), fill=(200, 100, 50))
        img.save(venice_image_path)
        
        # Create mock Gemini images based on the Venice image
        gemini_images = []
        for i in range(3):
            image_path = os.path.join(self.gemini_output_dir, f"gemini_from_venice_{i}.png")
            img = Image.new('RGB', (512, 512), color=(50, 100, 150))  # Same background
            draw = ImageDraw.Draw(img)
            
            # Draw a shape that varies slightly by view but maintains the same color
            draw.rectangle(
                (100 + i*20, 100 + i*10, 400 - i*20, 400 - i*10), 
                fill=(200, 100, 50)  # Same fill color
            )
            
            img.save(image_path)
            gemini_images.append(image_path)
        
        # Evaluate consistency between Venice source and Gemini-generated views
        all_images = [venice_image_path] + gemini_images
        pipeline_results = self.evaluator.evaluate_view_consistency(all_images)
        
        # Save the pipeline evaluation report
        report_path = self.evaluator.save_evaluation_report(
            pipeline_results, 
            report_name="venice_to_gemini_pipeline.json"
        )
        
        # Log the results
        logger.info(f"Venice to Gemini pipeline evaluation saved to: {report_path}")
        logger.info(f"Pipeline consistency score: {pipeline_results['overall_consistency_score']}")
    
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
