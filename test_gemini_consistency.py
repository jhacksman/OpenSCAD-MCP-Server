"""
Test script for Google Gemini API image consistency.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.gemini_api import GeminiImageGenerator
from src.utils.image_comparison import MultiViewConsistencyEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGeminiConsistency(unittest.TestCase):
    """
    Test cases for Google Gemini API image consistency.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test output directories
        self.test_output_dir = "output/test_gemini_consistency"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Get API key from environment
        self.api_key = os.environ.get("google", "test_api_key")
        
        # Create the generator with the API key
        self.gemini_generator = GeminiImageGenerator(
            api_key=self.api_key,
            output_dir=self.test_output_dir
        )
        
        # Create the consistency evaluator
        self.evaluator = MultiViewConsistencyEvaluator(
            output_dir=self.test_output_dir
        )
    
    @patch('src.ai.gemini_api.GeminiImageGenerator.generate_image')
    def test_multi_view_consistency(self, mock_generate):
        """
        Test generating multiple views of the same object and evaluate consistency.
        """
        # Mock image generation
        def mock_generate_side_effect(prompt, model=None, **kwargs):
            view_index = kwargs.get('view_index', 0)
            view_direction = kwargs.get('view_direction', 'front')
            
            # Create a test image with slight variations based on view
            from PIL import Image, ImageDraw
            image_path = os.path.join(self.test_output_dir, f"view_{view_index}_{view_direction}.png")
            
            # Base color varies slightly by view for testing
            base_color = (100 + view_index * 5, 150 - view_index * 3, 200)
            
            img = Image.new('RGB', (512, 512), color=base_color)
            draw = ImageDraw.Draw(img)
            
            # Draw a shape that varies by view direction
            if view_direction == 'front':
                draw.rectangle((150, 150, 350, 350), fill=(200, 100, 50))
            elif view_direction == 'side':
                draw.rectangle((200, 150, 400, 350), fill=(200, 100, 50))
            elif view_direction == 'top':
                draw.ellipse((150, 150, 350, 350), fill=(200, 100, 50))
            else:
                draw.rectangle((150 + view_index * 10, 150, 350 + view_index * 10, 350), fill=(200, 100, 50))
            
            img.save(image_path)
            
            return {
                "prompt": prompt,
                "model": model or "gemini-2.0-flash-exp-image-generation",
                "local_path": image_path,
                "view_index": view_index,
                "view_direction": view_direction
            }
        
        mock_generate.side_effect = mock_generate_side_effect
        
        # Test parameters
        prompt = "A low-poly rabbit with black background"
        view_directions = ['front', 'side', 'top', 'back', 'bottom']
        
        # Generate multiple views
        generated_views = []
        for i, direction in enumerate(view_directions):
            result = self.gemini_generator.generate_image(
                prompt, 
                view_index=i,
                view_direction=direction
            )
            generated_views.append(result["local_path"])
            
            # Log the generated view
            logger.info(f"Generated view {i} ({direction}): {result['local_path']}")
        
        # Evaluate consistency across views
        consistency_results = self.evaluator.evaluate_view_consistency(generated_views)
        
        # Save the evaluation report
        report_path = self.evaluator.save_evaluation_report(
            consistency_results,
            report_name="multi_view_consistency.json"
        )
        
        # Log the results
        logger.info(f"Multi-view consistency score: {consistency_results['overall_consistency_score']}")
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Verify the consistency score is reasonable
        self.assertGreaterEqual(consistency_results['overall_consistency_score'], 0.5)
    
    @patch('src.ai.gemini_api.GeminiImageGenerator.generate_multiple_views')
    def test_prompt_engineering_for_consistency(self, mock_generate_views):
        """
        Test different prompt engineering techniques for improving consistency.
        """
        # Mock multiple views generation
        def mock_generate_views_side_effect(prompt, num_views=3, **kwargs):
            results = []
            
            # Create test images with variations based on prompt
            from PIL import Image, ImageDraw
            
            # Base color varies by prompt complexity
            base_color = (100, 150, 200)
            if "consistent style" in prompt.lower():
                # More consistent base color if style consistency is requested
                base_color = (110, 160, 210)
            
            for i in range(num_views):
                image_path = os.path.join(self.test_output_dir, f"prompt_test_view_{i}.png")
                
                img = Image.new('RGB', (512, 512), color=base_color)
                draw = ImageDraw.Draw(img)
                
                # Draw a shape with variations
                # More consistent shape if detail preservation is requested
                if "preserve details" in prompt.lower():
                    draw.rectangle((150, 150, 350, 350), fill=(200, 100, 50))
                else:
                    draw.rectangle((150 + i * 15, 150, 350 + i * 15, 350), fill=(200, 100, 50))
                
                img.save(image_path)
                
                results.append({
                    "prompt": prompt,
                    "local_path": image_path,
                    "view_index": i + 1,
                    "view_direction": f"view_{i+1}"
                })
            
            return results
        
        mock_generate_views.side_effect = mock_generate_views_side_effect
        
        # Test different prompt techniques
        prompt_techniques = [
            "A low-poly rabbit",  # Basic prompt
            "A low-poly rabbit with consistent style and colors",  # Style anchoring
            "A low-poly rabbit, preserve details across all views",  # Detail preservation
            "A low-poly rabbit, front view, side view, and top view",  # View specification
        ]
        
        all_results = {}
        
        for prompt in prompt_techniques:
            # Generate multiple views with this prompt
            views = self.gemini_generator.generate_multiple_views(prompt, num_views=3)
            
            # Extract image paths
            image_paths = [view["local_path"] for view in views]
            
            # Evaluate consistency
            consistency_results = self.evaluator.evaluate_view_consistency(image_paths)
            
            # Save results for this prompt
            all_results[prompt] = {
                "images": image_paths,
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
        with open(os.path.join(self.test_output_dir, "prompt_techniques.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    @patch('src.ai.gemini_api.GeminiImageGenerator.generate_image')
    def test_3d_object_view_consistency(self, mock_generate):
        """
        Test generating multiple views of a 3D object with specific view angles.
        """
        # Mock image generation
        def mock_generate_side_effect(prompt, model=None, **kwargs):
            view_angle = kwargs.get('view_angle', '0')
            
            # Create a test image with variations based on view angle
            from PIL import Image, ImageDraw
            image_path = os.path.join(self.test_output_dir, f"angle_{view_angle}.png")
            
            # Parse angle to determine image content
            try:
                angle = int(view_angle.replace('°', ''))
            except:
                angle = 0
            
            # Base color is consistent
            base_color = (50, 100, 150)
            
            img = Image.new('RGB', (512, 512), color=base_color)
            draw = ImageDraw.Draw(img)
            
            # Draw a shape that rotates based on the angle
            center_x, center_y = 256, 256
            width, height = 200, 100
            
            # Simple rotation calculation
            import math
            rad = math.radians(angle)
            cos_val = math.cos(rad)
            sin_val = math.sin(rad)
            
            # Calculate corners of a rectangle
            points = [
                (center_x - width/2, center_y - height/2),
                (center_x + width/2, center_y - height/2),
                (center_x + width/2, center_y + height/2),
                (center_x - width/2, center_y + height/2)
            ]
            
            # Rotate points
            rotated_points = []
            for x, y in points:
                x_shifted, y_shifted = x - center_x, y - center_y
                x_rot = x_shifted * cos_val - y_shifted * sin_val
                y_rot = x_shifted * sin_val + y_shifted * cos_val
                rotated_points.append((x_rot + center_x, y_rot + center_y))
            
            # Draw the rotated rectangle
            draw.polygon(rotated_points, fill=(200, 100, 50))
            
            img.save(image_path)
            
            return {
                "prompt": prompt,
                "model": model or "gemini-2.0-flash-exp-image-generation",
                "local_path": image_path,
                "view_angle": view_angle
            }
        
        mock_generate.side_effect = mock_generate_side_effect
        
        # Test parameters
        prompt = "A 3D model of a low-poly rabbit"
        view_angles = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
        
        # Generate views from different angles
        generated_views = []
        for angle in view_angles:
            result = self.gemini_generator.generate_image(
                prompt, 
                view_angle=angle
            )
            generated_views.append(result["local_path"])
            
            # Log the generated view
            logger.info(f"Generated view at angle {angle}: {result['local_path']}")
        
        # Evaluate consistency across views
        consistency_results = self.evaluator.evaluate_view_consistency(generated_views)
        
        # Save the evaluation report
        report_path = self.evaluator.save_evaluation_report(
            consistency_results,
            report_name="3d_view_angle_consistency.json"
        )
        
        # Log the results
        logger.info(f"3D view angle consistency score: {consistency_results['overall_consistency_score']}")
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Verify the consistency score is reasonable
        self.assertGreaterEqual(consistency_results['overall_consistency_score'], 0.5)
    
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
