# Docker Command Examples for SDXL DGX Image Lab v17

## Complete Headless Mode Example (All Parameters)

```bash
docker run --name image_gen_full_v17 \
 --gpus all \
 --runtime=nvidia \
 --network host \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
 -e CUDA_VISIBLE_DEVICES=3 \
 -e HF_HUB_OFFLINE=1 \
 -e HEADLESS=true \
 -e PROMPT="a biomechanical alien warrior in an epic battle scene" \
 -e NEGATIVE_PROMPT="cartoon, anime, low quality, blurry" \
 -e MODEL="Frank Frazetta Fantasy" \
 -e STYLE_PROFILE="H.R. Giger Biomechanical" \
 -e SCHEDULER="DPM++ 2M" \
 -e STEPS=35 \
 -e CFG_SCALE=8.5 \
 -e WIDTH=1024 \
 -e HEIGHT=768 \
 -e BATCH_SIZE=6 \
 -e SEED=42 \
 -e RUN_ALL_PROFILES=false \
 -e RUN_ALL_MODELS=false \
 -e INSTANCE_ID=full_example_gpu3 \
 -e GRADIO_PORT=7865 \
 -v /root/.cache/huggingface:/root/.cache/huggingface \
 -v "$(pwd)":/app \
 -w /app \
 gradio_app_generic:dude \
 python3 gradio_app_multi-v17.py
```

## Parameter Explanations

### Docker Runtime Parameters
- `--name image_gen_full_v17` - Container name for easy management
- `--gpus all` - Grant access to all GPUs
- `--runtime=nvidia` - Use NVIDIA container runtime
- `--network host` - Use host networking (recommended for DGX)

### GPU Control
- `NVIDIA_VISIBLE_DEVICES=all` - Make all GPUs visible to container
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` - Required GPU capabilities
- `CUDA_VISIBLE_DEVICES=3` - **Pin to specific GPU** (0,1,2,3,4,5,6,7)

### Hugging Face Cache
- `HF_HUB_OFFLINE=1` - **Force offline mode** (no internet downloads)
- `-v /root/.cache/huggingface:/root/.cache/huggingface` - **Mount model cache**

### Generation Mode
- `HEADLESS=true` - **Enable headless mode** (no UI, auto-exit)
- `HEADLESS=false` or omit - Launch Gradio UI

### Core Generation Parameters
- `PROMPT="..."` - **Main generation prompt** (required for headless)
- `NEGATIVE_PROMPT="..."` - Things to avoid (optional)
- `MODEL="..."` - Model selection:
  - `"SDXL Base 1.0"` - Standard SDXL
  - `"SDXL Turbo"` - Fast generation
  - `"RealVis XL v5.0"` - Photorealistic
  - `"CyberRealistic XL 5.8"` - Cyberpunk/realistic
  - `"Animagine XL 4.0"` - Anime style
  - `"Juggernaut XL"` - Versatile model
  - `"DreamShaper XL"` - Artistic/dreamy
  - `"EpicRealism XL"` - Epic realistic scenes
  - `"Pixel Art XL"` - Pixel art style
  - `"Anime Illust XL"` - Anime illustration

### Style Profiles
- `STYLE_PROFILE="..."` - Artistic style:
  - `"None / Raw"` - No style modifications
  - `"Photoreal"` - Photorealistic enhancement
  - `"Cinematic"` - Movie-like lighting
  - `"Anime / Vibrant"` - Anime style
  - `"Soft Illustration"` - Gentle artistic style
  - `"Black & White"` - Monochrome
  - `"Pencil Sketch"` - Hand-drawn look
  - `"35mm Film"` - Film photography
  - `"Rotoscoping"` - Animation style
  - `"R-Rated"` - Mature themes
  - **New v17 Artist Profiles:**
  - `"Tim Burton Style"` - Gothic whimsical
  - `"Frank Frazetta Fantasy"` - Epic fantasy art
  - `"Ralph Bakshi Animation"` - 1970s rotoscoped
  - `"H.R. Giger Biomechanical"` - Alien horror
  - `"Dark Fantasy / Grimdark"` - Gothic atmosphere

### Technical Parameters
- `SCHEDULER="..."` - Sampling method:
  - `"Default"` - Model's default scheduler
  - `"Euler"` - Euler discrete
  - `"DPM++ 2M"` - DPM++ 2M Karras
  - `"UniPC"` - UniPC scheduler
- `STEPS=35` - Inference steps (4-80, higher = better quality)
- `CFG_SCALE=8.5` - Guidance scale (1.0-20.0, higher = more prompt adherence)

### Image Dimensions
- `WIDTH=1024` - Image width (256-1536, multiple of 8)
- `HEIGHT=768` - Image height (256-1536, multiple of 8)
- **Common presets:**
  - `1024×576` - Widescreen
  - `768×768` - Square
  - `512×512` - Small square
  - `1536×640` - Ultra-wide

### Batch & Seed
- `BATCH_SIZE=6` - Number of images to generate (1-10)
- `SEED=42` - Random seed for reproducibility (-1 for random)

### Multi-Generation Modes
- `RUN_ALL_PROFILES=true` - Generate with ALL style profiles for selected model
- `RUN_ALL_MODELS=true` - Generate with ALL models × ALL profiles (massive job!)
- **Safety:** These are mutually exclusive - only one can be true

### Instance & Logging
- `INSTANCE_ID=full_example_gpu3` - **Unique identifier** for this container
  - Creates separate log file: `jobs_full_example_gpu3.log`
  - Prevents log conflicts in multi-container setups
- `GRADIO_PORT=7865` - UI port (default: 7865)

### Volume Mounts
- `-v "$(pwd)":/app` - Mount current directory as `/app`
- `-w /app` - Set working directory to `/app`

## Quick Examples

### Simple UI Mode
```bash
docker run --name image_gen_ui_v17 \
 --gpus all --runtime=nvidia --network host \
 -e CUDA_VISIBLE_DEVICES=3 \
 -e INSTANCE_ID=ui_gpu3 \
 -v /root/.cache/huggingface:/root/.cache/huggingface \
 -v "$(pwd)":/app -w /app \
 gradio_app_generic:dude \
 python3 gradio_app_multi-v17.py
```

### Quick Headless Generation
```bash
docker run --name quick_gen_v17 \
 --gpus all --runtime=nvidia \
 -e CUDA_VISIBLE_DEVICES=3 \
 -e HEADLESS=true \
 -e PROMPT="a cyberpunk city at night" \
 -e STYLE_PROFILE="Tim Burton Style" \
 -e BATCH_SIZE=4 \
 -v /root/.cache/huggingface:/root/.cache/huggingface \
 -v "$(pwd)":/app -w /app \
 gradio_app_generic:dude \
 python3 gradio_app_multi-v17.py
```

### All Profiles Sweep
```bash
docker run --name all_profiles_v17 \
 --gpus all --runtime=nvidia \
 -e CUDA_VISIBLE_DEVICES=3 \
 -e HEADLESS=true \
 -e PROMPT="a majestic dragon" \
 -e MODEL="CyberRealistic XL 5.8" \
 -e RUN_ALL_PROFILES=true \
 -e BATCH_SIZE=2 \
 -e INSTANCE_ID=dragon_sweep \
 -v /root/.cache/huggingface:/root/.cache/huggingface \
 -v "$(pwd)":/app -w /app \
 gradio_app_generic:dude \
 python3 gradio_app_multi-v17.py
```

## Tips

1. **GPU Selection:** Use `CUDA_VISIBLE_DEVICES=N` to pin to specific GPU
2. **Instance ID:** Always set unique `INSTANCE_ID` for multiple containers
3. **Offline Mode:** Use `HF_HUB_OFFLINE=1` after models are downloaded
4. **Batch Size:** Start small (2-4) to avoid OOM, increase based on VRAM
5. **Seeds:** Use same seed for reproducible results across runs
6. **Profiles:** Try artist profiles for specific aesthetic styles