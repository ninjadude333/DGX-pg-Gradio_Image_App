#!/usr/bin/env python3
"""
Download all SDXL DGX Image Lab v16 models into the local HuggingFace cache.

Run this ONCE on a machine with internet access:

    python3 download_models_v16.py

By default it caches into /root/.cache/huggingface (or HF_HOME if set).
You can then mount that directory into your DGX containers and run with HF_HUB_OFFLINE=1.
"""

import os
from huggingface_hub import snapshot_download

# Same IDs as in AVAILABLE_MODELS in gradio_app_multi-v16.py
MODEL_REPOS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "SG161222/RealVisXL_V5.0",
    "John6666/cyberrealistic-xl-v58-sdxl",
    "cagliostrolab/animagine-xl-4.0",
    "stablediffusionapi/juggernautxl",
    "Lykon/dreamshaper-xl-1-0",
    "AiAF/epicrealismXL-vx1Finalkiss_Checkpoint_SDXL",
    "nerijs/pixel-art-xl",
    "Eugeoter/anime_illust_diffusion_xl",
]

HF_CACHE_DIR = os.environ.get("HF_HOME", "/root/.cache/huggingface")


def main():
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    print(f"Using cache dir: {HF_CACHE_DIR}")

    for repo in MODEL_REPOS:
        print(f"\n=== Downloading {repo} ===")
        snapshot_download(
            repo_id=repo,
            cache_dir=HF_CACHE_DIR,
            resume_download=True,
            local_files_only=False,
        )
        print(f"Finished {repo}")

    print("\nAll models downloaded. You can now run with HF_HUB_OFFLINE=1 and mount this cache dir into your DGX containers.")


if __name__ == "__main__":
    main()
