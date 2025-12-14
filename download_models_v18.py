#!/usr/bin/env python3
"""
Download all models for SDXL DGX Image Lab v18
Includes SDXL models + PixArt-Σ
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
}

def download_model(model_name: str, model_id: str):
    """Download a single model."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}")
    
    try:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        print(f"✅ {model_name} downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

if __name__ == "__main__":
    print("SDXL DGX Image Lab v18 - Model Downloader")
    print(f"Downloading {len(MODELS)} models to ~/.cache/huggingface")
    
    for model_name, model_id in MODELS.items():
        download_model(model_name, model_id)
    
    print("\n" + "="*60)
    print("✅ All downloads complete!")
    print("="*60)
