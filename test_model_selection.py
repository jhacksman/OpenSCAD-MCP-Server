import os
import logging
from src.ai.venice_api import VeniceImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venice.ai API key (replace with your own or use environment variable)
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "B9Y68yQgatQw8wmpmnIMYcGip1phCt-43CS0OktZU6")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")

# Test natural language model selection
def test_model_selection():
    """Test the natural language model selection functionality."""
    # Initialize the Venice API client
    venice_generator = VeniceImageGenerator(VENICE_API_KEY, OUTPUT_DIR)
    
    # Test cases - natural language preferences to expected model mappings
    test_cases = [
        ("default", "fluently-xl"),
        ("fastest model please", "fluently-xl"),
        ("I need a high quality image", "flux-dev"),
        ("create an uncensored image", "flux-dev-uncensored"),
        ("make it realistic", "pony-realism"),
        ("I want something artistic", "lustify-sdxl"),
        ("use stable diffusion", "stable-diffusion-3.5"),
        ("invalid model name", "fluently-xl"),  # Should default to fluently-xl
    ]
    
    # Run tests
    for preference, expected_model in test_cases:
        mapped_model = venice_generator.map_model_preference(preference)
        logger.info(f"Preference: '{preference}' -> Model: '{mapped_model}'")
        assert mapped_model == expected_model, f"Expected {expected_model}, got {mapped_model}"
    
    logger.info("All model preference mappings tests passed!")

if __name__ == "__main__":
    logger.info("Starting Venice.ai model selection mapping tests")
    test_model_selection()
