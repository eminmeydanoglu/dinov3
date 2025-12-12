from __future__ import annotations

import os
import time
import torch
import sys
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import requests

def setup_image():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        return image
    except Exception as e:
        print(f"Failed to download image: {e}")
        # Create dummy image
        return Image.new('RGB', (224, 224), color='red')

def test_transformers_loading():
    print("\n--- Testing Transformers Loading (Requires HF Login) ---")
    pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    try:
        processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
        print("Successfully loaded model via Transformers!")
        
        image = setup_image()
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.inference_mode():
            outputs = model(**inputs)
        end_time = time.time()
        
        print(f"Inference successful! Shape: {outputs.pooler_output.shape}")
        print(f"Inference time: {end_time - start_time:.4f}s")
        return True
    except Exception as e:
        print(f"Transformers loading failed: {e}")
        return False

def test_local_hub_loading():
    print("\n--- Testing Local Hub Loading ---")
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Repo Dir: {repo_dir}")
    
    # Try loading with default weights (might fail if URL is bad/forbidden)
    # If it fails, we can try to point to a local file if we had one.
    # For now, let's try the standard load.
    try:
        # We need to manually add the repo path to sys.path for hubconf to work if called directly,
        # but torch.hub.load does this. However, source='local' implies we point to the dir.
        # Note: torch.hub.load(source='local') expects the directory to contain hubconf.py
        
        # NOTE: The README example uses:
        # dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT>)
        
        # We will try to use the HF URL as the weights argument if possible, or skip if we need a local file.
        # But wait, the code we saw in backbones.py supports URLs!
        # But the default URL (Facebook DL) was 403. 
        # So we NEED to provide a valid URL or path. 
        # Since we are using HF, maybe we can get the direct URL from HF? 
        # Or just skip this test if we don't have a file.
        # Let's try to lazy-load if possible or just report that we need weights.
        
        print("Attempting to load 'dinov3_vits16' from local source...")
        # For the sake of the test, let's see if we can load the ARCHITECTURE without weights first?
        # The hubconf functions take `pretrained=True` by default.
        # Let's try `pretrained=False` to verify the code structure works.
        
        model = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', pretrained=False)
        print("Successfully loaded architecture (pretrained=False) via Torch Hub!")
        
        # Now try to move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            print("Moved model to CUDA.")
            
            # Dummy inference
            input_tensor = torch.randn(1, 3, 224, 224).cuda()
            with torch.inference_mode():
                output = model(input_tensor)
            print(f"Dummy inference successful! Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"Local Hub loading failed: {e}")
        return False

if __name__ == "__main__":
    success_transformers = test_transformers_loading()
    success_hub = test_local_hub_loading()
    
    if success_transformers or success_hub:
        print("\n--- OVERALL SUCCESS: At least one loading method worked. ---")
    else:
        print("\n--- OVERALL FAILURE: Both loading methods failed. ---")
        sys.exit(1)
