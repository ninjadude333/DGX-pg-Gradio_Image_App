#!/usr/bin/env python3
"""
Download all models for SDXL DGX Image Lab v19
Includes SDXL models + PixArt-\u03a3 + SD3 Medium
"""

import os
from huggingface_hub import snapshot_download

# Models to download
MODELS = {
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "RealVis XL v5.0": "SG161222/RealVisXL_V5.0",
    "CyberRealistic XL 5.8": "John6666/cyberrealistic-xl-v58-sdxl",
    "Animagine XL 4.0": "cagliostrolab/animagine-xl-4.0",
    "Juggernaut XL": "stablediffusionapi/juggernautxl",
    "PixArt Sigma XL 1024": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "SD3 Medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}

def download_model(model_name: str, model_id: str):
    """Download a single model."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}")
    
    # SD3 is gated - requires authentication
    if "stable-diffusion-3" in model_id:
        print("\u26a0\ufe0f  SD3 is a GATED model!")
        print("You must:")
        print("  1. Accept the license at: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers")
        print("  2. Login with: huggingface-cli login")
        print("  3. Use a token with read access")
        print()
    
    try:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        print(f"\u2705 {model_name} downloaded successfully")
    except Exception as e:
        print(f"\u274c Failed to download {model_name}: {e}")
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("\u26a0\ufe0f  This is likely a gated model. Make sure you:")
            print("  - Have accepted the model license on HuggingFace")
            print("  - Are logged in with: huggingface-cli login")

if __name__ == "__main__":
    print("SDXL DGX Image Lab v19 - Model Downloader")
    print(f"Downloading {len(MODELS)} models to HuggingFace cache")
    print("\nNOTE: SD3 Medium requires HuggingFace authentication!")
    print("Run: huggingface-cli login")
    print()
    
    for model_name, model_id in MODELS.items():
        download_model(model_name, model_id)
    
    print("\n" + "="*60)
    print("\u2705 All downloads complete!")
    print("="*60)
