"""
Test script for Google Gemini API integration.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGeminiAPI(unittest.TestCase):
    """
    Test cases for Google Gemini API integration.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create a test output directory
        self.test_output_dir = "output/test_gemini"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Mock API key
        self.api_key = "test_api_key"
        
        # Create the generator with the mock API key
        self.gemini_generator = GeminiImageGenerator(
            api_key=self.api_key,
            output_dir=self.test_output_dir
        )
    
    @patch('requests.post')
    def test_generate_image(self, mock_post):
        """
        Test generating a single image with Gemini API.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Generated image description"
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test parameters
        prompt = "A low-poly rabbit with black background"
        model = "gemini-2.0-flash-exp-image-generation"
        
        # Call the method
        result = self.gemini_generator.generate_image(prompt, model)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["prompt"], prompt)
        self.assertEqual(result["model"], model)
        self.assertTrue("local_path" in result)
        self.assertTrue(os.path.exists(result["local_path"]))
        
        # Verify the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertTrue("generativelanguage.googleapis.com" in args[0])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], self.api_key)
        self.assertTrue("prompt" in str(kwargs["json"]))
    
    @patch('requests.post')
    def test_generate_multiple_views(self, mock_post):
        """
        Test generating multiple views of an object with Gemini API.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Generated image description"
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test parameters
        prompt = "A low-poly rabbit"
        num_views = 3
        
        # Call the method
        results = self.gemini_generator.generate_multiple_views(prompt, num_views)
        
        # Verify the results
        self.assertEqual(len(results), num_views)
        for i, result in enumerate(results):
            self.assertTrue("view_direction" in result)
            self.assertEqual(result["view_index"], i + 1)
            self.assertTrue("local_path" in result)
            self.assertTrue(os.path.exists(result["local_path"]))
        
        # Verify the API calls
        self.assertEqual(mock_post.call_count, num_views)
    
    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """
        Test error handling in the Gemini API client.
        """
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        # Test parameters
        prompt = "A low-poly rabbit"
        
        # Call the method and expect an exception
        with self.assertRaises(Exception):
            self.gemini_generator.generate_image(prompt)
    
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
