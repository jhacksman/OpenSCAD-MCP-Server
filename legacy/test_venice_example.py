import os
import requests
import json

# Venice.ai API configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")  # Set via environment variable
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API endpoint
url = "https://api.venice.ai/api/v1/image/generate"

# Test prompt
prompt = "A coffee mug with geometric patterns"

# Prepare payload
payload = {
    "model": "fluently-xl",  # Try default model instead of flux
    "prompt": prompt,
    "height": 1024,
    "width": 1024,
    "steps": 20,
    "return_binary": False,
    "hide_watermark": False,
    "format": "png",
    "embed_exif_metadata": False
}

# Set up headers
headers = {
    "Authorization": f"Bearer {VENICE_API_KEY}",
    "Content-Type": "application/json"
}

print(f"Sending request to {url} with prompt: '{prompt}'")

# Make API request
try:
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nImage generation result:")
        print(json.dumps(result, indent=2))
        
        # Print image URL if available
        if "images" in result and len(result["images"]) > 0:
            image_url = result["images"][0]
            print(f"\nImage URL: {image_url}")
            
            # Download image
            image_filename = f"{prompt[:20].replace(' ', '_')}_flux.png"
            image_path = os.path.join(OUTPUT_DIR, image_filename)
            
            print(f"Downloading image to {image_path}...")
            img_response = requests.get(image_url, stream=True)
            if img_response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Image saved to {image_path}")
            else:
                print(f"Failed to download image: {img_response.status_code}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {str(e)}")
