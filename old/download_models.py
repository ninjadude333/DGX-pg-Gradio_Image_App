#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# Optional: be explicit (but HF already defaults to this)
os.environ.setdefault("HF_HUB_CACHE", "/root/.cache/huggingface/hub")

MODELS = {
    # --- Baseline & speed ---
    "sdxl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",

    # --- Photorealistic XL ---
    "realvis-xl-v4": "SG161222/RealVisXL_V4.0",
    "realvis-xl-v5-lightning": "SG161222/RealVisXL_V5.0_Lightning",

    # --- Semi-realistic / artistic XL ---
    "dreamshaper-xl-lightning": "Lykon/dreamshaper-xl-lightning",

    # --- Strong photoreal models ---
    "juggernaut-xl": "glides/juggernautxl",
    "cyberrealistic-xl-v58": "John6666/cyberrealistic-xl-v58-sdxl",

    # --- Extra lightning (very fast SDXL) ---
    "sdxl-lightning-2step": "rupeshs/SDXL-Lightning-2steps",
}

def main():
    print("Using HF cache dir:", os.getenv("HF_HUB_CACHE", "<default>"))
    for name, repo in MODELS.items():
        print(f"\n=== Downloading / warming up: {name} ({repo}) ===")
        snapshot_download(
            repo_id=repo,
            local_files_only=False,   # allow download if not cached
        )
    print("\n✅ All models downloaded into HF cache.")

if __name__ == "__main__":
    main()
