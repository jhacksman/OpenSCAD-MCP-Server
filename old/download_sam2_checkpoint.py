import os
import sys
import logging
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SAM2 checkpoint URLs
CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_vit_h.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_vit_l.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_vit_b.pth"
}

# Checkpoint sizes (approximate, in MB)
CHECKPOINT_SIZES = {
    "vit_h": 2560,  # 2.5 GB
    "vit_l": 1250,  # 1.2 GB
    "vit_b": 380    # 380 MB
}

def download_checkpoint(model_type="vit_b", output_dir="models"):
    """
    Download SAM2 checkpoint.
    
    Args:
        model_type: Model type to download (vit_h, vit_l, vit_b)
        output_dir: Directory to save the checkpoint
    
    Returns:
        Path to the downloaded checkpoint
    """
    if model_type not in CHECKPOINT_URLS:
        raise ValueError(f"Invalid model type: {model_type}. Available types: {list(CHECKPOINT_URLS.keys())}")
    
    url = CHECKPOINT_URLS[model_type]
    output_path = os.path.join(output_dir, f"sam2_{model_type}.pth")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if checkpoint already exists
    if os.path.exists(output_path):
        logger.info(f"Checkpoint already exists at {output_path}")
        return output_path
    
    # Download checkpoint
    logger.info(f"Downloading SAM2 checkpoint ({model_type}) from {url}")
    logger.info(f"Approximate size: {CHECKPOINT_SIZES[model_type]} MB")
    
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Downloading {model_type}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Checkpoint downloaded to {output_path}")
        return output_path
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading checkpoint: {str(e)}")
        # Remove partial download if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        # Remove partial download if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        sys.exit(1)

def main():
    """Main function to parse arguments and download checkpoint."""
    parser = argparse.ArgumentParser(description="Download SAM2 checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=list(CHECKPOINT_URLS.keys()),
                        help="Model type to download (vit_h, vit_l, vit_b). Default: vit_b (smallest)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save the checkpoint")
    
    args = parser.parse_args()
    
    # Print model information
    logger.info(f"Selected model: {args.model_type}")
    logger.info(f"Approximate sizes: vit_h: 2.5 GB, vit_l: 1.2 GB, vit_b: 380 MB")
    
    try:
        checkpoint_path = download_checkpoint(args.model_type, args.output_dir)
        logger.info(f"Checkpoint ready at: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
