# Docker Examples v18 🐳

Complete Docker command reference for SDXL DGX Image Lab v18.

---

## 📋 Quick Reference

### Basic UI Mode
```bash
sudo docker run --name image_gen_v18 \
  --gpus all --runtime=nvidia --network host \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e INSTANCE_ID=dgx_gpu3 \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Basic Headless Mode
```bash
sudo docker run --name image_gen_v18_headless \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e HEADLESS=true \
  -e PROMPT="a futuristic cityscape" \
  -e MODEL="SDXL Base 1.0" \
  -e BATCH_SIZE=4 \
  -e INSTANCE_ID=headless_job_1 \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 🆕 v18 New Features

### Using PixArt-Σ Model

**UI Mode with PixArt:**
```bash
sudo docker run --name image_gen_v18_pixart \
  --gpus all --runtime=nvidia --network host \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e INSTANCE_ID=pixart_test \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```
Then select "PixArt Sigma XL 1024" from the model dropdown.

**Headless Mode with PixArt:**
```bash
sudo docker run --name image_gen_v18_pixart_headless \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HEADLESS=true \
  -e PROMPT="a serene mountain landscape at sunset" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=20 \
  -e BATCH_SIZE=2 \
  -e INSTANCE_ID=pixart_job_1 \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Configuring Pipeline Cache

**Larger Cache (3 models):**
```bash
-e PIPE_CACHE_MAX=3
```

**Disable Cache (always reload):**
```bash
-e PIPE_CACHE_MAX=0
```

**Example with cache:**
```bash
sudo docker run --name image_gen_v18_cached \
  --gpus all --runtime=nvidia --network host \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e PIPE_CACHE_MAX=3 \
  -e INSTANCE_ID=cached_instance \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 🎨 Artist/Genre Style Examples

### Tim Burton Style
```bash
sudo docker run --name image_gen_v18_burton \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HEADLESS=true \
  -e PROMPT="a gothic mansion on a hill" \
  -e STYLE_PROFILE="Tim Burton Style" \
  -e MODEL="SDXL Base 1.0" \
  -e STEPS=30 \
  -e BATCH_SIZE=4 \
  -e INSTANCE_ID=burton_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Frank Frazetta Fantasy
```bash
sudo docker run --name image_gen_v18_frazetta \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HEADLESS=true \
  -e PROMPT="a barbarian warrior on a cliff" \
  -e STYLE_PROFILE="Frank Frazetta Fantasy" \
  -e MODEL="Juggernaut XL" \
  -e SCHEDULER="DPM++ 2M" \
  -e STEPS=35 \
  -e BATCH_SIZE=6 \
  -e INSTANCE_ID=frazetta_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### H.R. Giger Biomechanical
```bash
sudo docker run --name image_gen_v18_giger \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HEADLESS=true \
  -e PROMPT="an alien corridor" \
  -e STYLE_PROFILE="H.R. Giger Biomechanical" \
  -e MODEL="RealVis XL v5.0" \
  -e SCHEDULER="Euler" \
  -e STEPS=40 \
  -e BATCH_SIZE=4 \
  -e INSTANCE_ID=giger_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 🚀 Multi-Generation Modes

### Run All Profiles for One Model
```bash
sudo docker run --name image_gen_v18_all_profiles \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e HEADLESS=true \
  -e PROMPT="a cyberpunk street scene" \
  -e MODEL="CyberRealistic XL 5.8" \
  -e RUN_ALL_PROFILES=true \
  -e BATCH_SIZE=2 \
  -e INSTANCE_ID=all_profiles_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Run All Models × All Profiles
```bash
sudo docker run --name image_gen_v18_full_sweep \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HEADLESS=true \
  -e PROMPT="a dragon in flight" \
  -e RUN_ALL_MODELS=true \
  -e BATCH_SIZE=1 \
  -e INSTANCE_ID=full_sweep_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 📐 Resolution Examples

### Square (1024×1024) - PixArt Optimal
```bash
-e WIDTH=1024 \
-e HEIGHT=1024
```

### Widescreen (1536×640)
```bash
-e WIDTH=1536 \
-e HEIGHT=640
```

### Portrait (768×1024)
```bash
-e WIDTH=768 \
-e HEIGHT=1024
```

### Custom Resolution
```bash
-e WIDTH=1280 \
-e HEIGHT=720
```
*Note: Dimensions automatically rounded to nearest multiple of 8*

---

## 🎛️ Parameter Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `HEADLESS` | bool | `false` | Enable headless mode (no UI) |
| `PROMPT` | string | *required* | Main generation prompt |
| `NEGATIVE_PROMPT` | string | `""` | Negative prompt |
| `MODEL` | string | `"SDXL Base 1.0"` | Model to use (see Models section) |
| `SCHEDULER` | string | `"Default"` | Scheduler (Default, Euler, DPM++ 2M, UniPC) |
| `STYLE_PROFILE` | string | `"Photoreal"` | Style profile (see Profiles section) |

### Generation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `STEPS` | int | `26` | 4-80 | Inference steps |
| `CFG_SCALE` | float | `7.5` | 0.0-20.0 | Guidance scale |
| `WIDTH` | int | `1024` | 256-1536 | Image width (divisible by 8) |
| `HEIGHT` | int | `576` | 256-1536 | Image height (divisible by 8) |
| `BATCH_SIZE` | int | `4` | 1-10 | Number of images per generation |
| `SEED` | int | `-1` | -1 or 0-2³² | Random seed (-1 = random) |

### Multi-Generation Flags

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_ALL_PROFILES` | bool | `false` | Generate with all 29 style profiles |
| `RUN_ALL_MODELS` | bool | `false` | Generate with all models × all profiles |

*Note: `RUN_ALL_MODELS` and `RUN_ALL_PROFILES` are mutually exclusive*

### System Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `INSTANCE_ID` | string | `"default"` | Instance identifier for logging |
| `CUDA_VISIBLE_DEVICES` | string | `"0"` | GPU indices (e.g., "0,1,2,3") |
| `HF_HUB_OFFLINE` | bool | `1` | Offline mode for HuggingFace |
| `GRADIO_PORT` | int | `7865` | Gradio UI port |
| `PIPE_CACHE_MAX` | int | `2` | **v18 NEW:** Max cached pipelines |

---

## 🎨 Available Models (v18)

| Model Name | Model ID | Type | Best For |
|------------|----------|------|----------|
| `SDXL Base 1.0` | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL | General purpose |
| `SDXL Turbo` | `stabilityai/sdxl-turbo` | SDXL | Fast inference |
| `RealVis XL v5.0` | `SG161222/RealVisXL_V5.0` | SDXL | Photorealistic |
| `CyberRealistic XL 5.8` | `John6666/cyberrealistic-xl-v58-sdxl` | SDXL | Realistic portraits |
| `Animagine XL 4.0` | `cagliostrolab/animagine-xl-4.0` | SDXL | Anime style |
| `Juggernaut XL` | `stablediffusionapi/juggernautxl` | SDXL | Versatile |
| `PixArt Sigma XL 1024` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | **PixArt** | **High quality, txt2img only** |

---

## 🎨 Available Style Profiles (29 Total)

### Core Profiles (10)
- `None / Raw`
- `Photoreal`
- `Cinematic`
- `Anime / Vibrant`
- `Soft Illustration`
- `Black & White`
- `Pencil Sketch`
- `35mm Film`
- `Rotoscoping`
- `R-Rated`

### Artist/Genre Profiles (5)
- `Tim Burton Style`
- `Frank Frazetta Fantasy`
- `Ralph Bakshi Animation`
- `H.R. Giger Biomechanical`
- `Dark Fantasy / Grimdark`

### Extended Profiles (14)
- `Watercolor`
- `Hyper-Realistic Portrait`
- `ISOTOPIA Sci-Fi Blueprint`
- `Pixar-ish Soft CG`
- `Pixel Art / Isometric Game`
- `Low-Poly 3D / PS1`
- `Product Render / Industrial`
- `Isometric Tech Diagram`
- `Retro Comic / Halftone`
- `Vaporwave / Synthwave`
- `Children's Book Illustration`
- `Ink & Screentone Manga`
- `Analog Horror / VHS`
- `Architectural Visualization`

---

## 🔧 Advanced Examples

### High-Quality PixArt Generation (Canonical)
```bash
sudo docker run --name image_gen_v18_pixart_hq \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HEADLESS=true \
  -e PROMPT="a serene mountain landscape with golden light cascading through clouds" \
  -e NEGATIVE_PROMPT="low quality, distorted" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=30 \
  -e CFG_SCALE=4.5 \
  -e BATCH_SIZE=2 \
  -e SEED=42 \
  -e INSTANCE_ID=pixart_hq \
  -e HF_HUB_OFFLINE=1 \
  -e PIPE_CACHE_MAX=2 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### PixArt Emotional/Abstract Art
```bash
sudo docker run --name image_gen_v18_pixart_abstract \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HEADLESS=true \
  -e PROMPT="swirling colors dancing in cosmic space, ethereal and dreamlike" \
  -e NEGATIVE_PROMPT="low quality, distorted" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=28 \
  -e CFG_SCALE=4.0 \
  -e BATCH_SIZE=2 \
  -e INSTANCE_ID=pixart_abstract \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### PixArt Cinematic Landscape
```bash
sudo docker run --name image_gen_v18_pixart_cinematic \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HEADLESS=true \
  -e PROMPT="wide panoramic view of a mountain range at golden hour, dramatic clouds" \
  -e NEGATIVE_PROMPT="low quality, distorted" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=30 \
  -e CFG_SCALE=4.5 \
  -e BATCH_SIZE=2 \
  -e INSTANCE_ID=pixart_cinematic \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### PixArt Character Portrait
```bash
sudo docker run --name image_gen_v18_pixart_portrait \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e HEADLESS=true \
  -e PROMPT="a wise wizard with flowing robes, serene expression, soft magical glow" \
  -e NEGATIVE_PROMPT="low quality, distorted" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=32 \
  -e CFG_SCALE=4.8 \
  -e BATCH_SIZE=2 \
  -e INSTANCE_ID=pixart_portrait \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Fast Turbo Generation
```bash
sudo docker run --name image_gen_v18_turbo \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HEADLESS=true \
  -e PROMPT="a modern office interior" \
  -e MODEL="SDXL Turbo" \
  -e STEPS=4 \
  -e CFG_SCALE=1.0 \
  -e BATCH_SIZE=8 \
  -e INSTANCE_ID=turbo_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Anime Style with Animagine
```bash
sudo docker run --name image_gen_v18_anime \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HEADLESS=true \
  -e PROMPT="1girl, long hair, school uniform, cherry blossoms, spring" \
  -e NEGATIVE_PROMPT="lowres, bad anatomy, bad hands, text, error" \
  -e MODEL="Animagine XL 4.0" \
  -e STYLE_PROFILE="Anime / Vibrant" \
  -e STEPS=28 \
  -e CFG_SCALE=7.0 \
  -e BATCH_SIZE=4 \
  -e INSTANCE_ID=anime_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 🐛 Troubleshooting

### Model Not Found
```bash
# Ensure model is downloaded in HuggingFace cache
ls -la /root/.cache/huggingface/hub/

# If missing, download first (outside container or with HF_HUB_OFFLINE=0)
```

### Out of Memory (OOM)
```bash
# Reduce batch size
-e BATCH_SIZE=1

# Reduce resolution
-e WIDTH=768 \
-e HEIGHT=768

# Use SDXL Turbo with fewer steps
-e MODEL="SDXL Turbo" \
-e STEPS=4
```

### Cache Issues
```bash
# Disable cache if causing problems
-e PIPE_CACHE_MAX=0

# Or increase cache for more models
-e PIPE_CACHE_MAX=4
```

### PixArt-Specific Issues
```bash
# PixArt requires more VRAM than SDXL Turbo
# Use smaller batch size
-e BATCH_SIZE=2

# Reduce CFG if quality is poor
-e CFG_SCALE=4.5  # Not 7.5+

# Use optimal steps
-e STEPS=30  # Not 50+

# Check prompt style (use verbs/emotion, not keywords)
-e PROMPT="flowing water cascading over rocks"  # Good
# Not: "8k ultra-detailed masterpiece flowing water"  # Bad
```

---

## 📊 Performance Tips

### Optimal Settings by Model

**SDXL Base 1.0:**
- Steps: 25-35
- CFG: 7.0-8.0
- Batch: 4-6

**SDXL Turbo:**
- Steps: 4-8
- CFG: 1.0-2.0
- Batch: 8-10

**PixArt Sigma:**
- Steps: 24-36 (optimal: 30)
- CFG: 3.5-5.5 (optimal: 4.5)
- Batch: 2-4
- Resolution: 1024×1024
- Negative: "low quality, distorted" (minimal)

**RealVis XL / CyberRealistic:**
- Steps: 28-35
- CFG: 6.5-7.5
- Batch: 4-6

**Animagine XL:**
- Steps: 25-30
- CFG: 7.0-8.0
- Batch: 4-6

### Cache Strategy

**2-Model Workflow (Default):**
```bash
-e PIPE_CACHE_MAX=2
```
Switch between SDXL and PixArt frequently.

**3-Model Workflow:**
```bash
-e PIPE_CACHE_MAX=3
```
Rotate between 3 favorite models.

**Single Model (No Cache):**
```bash
-e PIPE_CACHE_MAX=0
```
Always use same model, save VRAM.

### PixArt VRAM Usage

| Batch Size | Resolution | VRAM Usage |
|------------|------------|------------|
| 1 | 1024×1024 | ~12 GB |
| 2 | 1024×1024 | ~16 GB |
| 4 | 1024×1024 | ~24 GB |
| 1 | 1024×576 | ~10 GB |
| 2 | 1024×576 | ~14 GB |

---

## 📝 Notes

- All v17 commands work unchanged in v18
- PixArt-Σ is txt2img only (no img2img in this app)
- Pipeline cache is in-process (not persistent across restarts)
- Dimensions automatically rounded to nearest multiple of 8
- `RUN_ALL_MODELS` and `RUN_ALL_PROFILES` are mutually exclusive
- **PixArt requires different settings than SDXL** (see [PIXART_INTEGRATION_GUIDE_v18.md](PIXART_INTEGRATION_GUIDE_v18.md))

---

## 🔗 Related Documentation

- [README_v18.md](../README_v18.md) - Main documentation
- [CHANGELOG_v18.md](CHANGELOG_v18.md) - v18 changes
- [PIXART_INTEGRATION_GUIDE_v18.md](PIXART_INTEGRATION_GUIDE_v18.md) - **PixArt-Σ complete guide**
- [WorkPlan_v18.md](WorkPlan_v18.md) - Development roadmaped Examples

### High-Quality PixArt Generation
```bash
sudo docker run --name image_gen_v18_pixart_hq \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HEADLESS=true \
  -e PROMPT="a detailed portrait of a wise old wizard with a long white beard, wearing ornate robes, magical atmosphere" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=25 \
  -e CFG_SCALE=4.5 \
  -e BATCH_SIZE=4 \
  -e SEED=42 \
  -e INSTANCE_ID=pixart_hq \
  -e HF_HUB_OFFLINE=1 \
  -e PIPE_CACHE_MAX=2 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Fast Turbo Generation
```bash
sudo docker run --name image_gen_v18_turbo \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HEADLESS=true \
  -e PROMPT="a modern office interior" \
  -e MODEL="SDXL Turbo" \
  -e STEPS=4 \
  -e CFG_SCALE=1.0 \
  -e BATCH_SIZE=8 \
  -e INSTANCE_ID=turbo_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

### Anime Style with Animagine
```bash
sudo docker run --name image_gen_v18_anime \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HEADLESS=true \
  -e PROMPT="1girl, long hair, school uniform, cherry blossoms, spring" \
  -e NEGATIVE_PROMPT="lowres, bad anatomy, bad hands, text, error" \
  -e MODEL="Animagine XL 4.0" \
  -e STYLE_PROFILE="Anime / Vibrant" \
  -e STEPS=28 \
  -e CFG_SCALE=7.0 \
  -e BATCH_SIZE=4 \
  -e INSTANCE_ID=anime_job \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 🐛 Troubleshooting

### Model Not Found
```bash
# Ensure model is downloaded in HuggingFace cache
ls -la /root/.cache/huggingface/hub/

# If missing, download first (outside container or with HF_HUB_OFFLINE=0)
```

### Out of Memory (OOM)
```bash
# Reduce batch size
-e BATCH_SIZE=1

# Reduce resolution
-e WIDTH=768 \
-e HEIGHT=768

# Use SDXL Turbo with fewer steps
-e MODEL="SDXL Turbo" \
-e STEPS=4
```

### Cache Issues
```bash
# Disable cache if causing problems
-e PIPE_CACHE_MAX=0

# Or increase cache for more models
-e PIPE_CACHE_MAX=4
```

### PixArt-Specific Issues
```bash
# PixArt requires more VRAM than SDXL Turbo
# Use smaller batch size
-e BATCH_SIZE=2

# Enable attention slicing (automatic in v18)
# No action needed - enabled by default
```

---

## 📊 Performance Tips

### Optimal Settings by Model

**SDXL Base 1.0:**
- Steps: 25-35
- CFG: 7.0-8.0
- Batch: 4-6

**SDXL Turbo:**
- Steps: 4-8
- CFG: 1.0-2.0
- Batch: 8-10

**PixArt Sigma:**
- Steps: 15-25
- CFG: 4.0-5.0
- Batch: 2-4
- Resolution: 1024×1024

**RealVis XL / CyberRealistic:**
- Steps: 28-35
- CFG: 6.5-7.5
- Batch: 4-6

**Animagine XL:**
- Steps: 25-30
- CFG: 7.0-8.0
- Batch: 4-6

### Cache Strategy

**2-Model Workflow (Default):**
```bash
-e PIPE_CACHE_MAX=2
```
Switch between SDXL and PixArt frequently.

**3-Model Workflow:**
```bash
-e PIPE_CACHE_MAX=3
```
Rotate between 3 favorite models.

**Single Model (No Cache):**
```bash
-e PIPE_CACHE_MAX=0
```
Always use same model, save VRAM.

---

## 📝 Notes

- All v17 commands work unchanged in v18
- PixArt-Σ is txt2img only (no img2img in this app)
- Pipeline cache is in-process (not persistent across restarts)
- Dimensions automatically rounded to nearest multiple of 8
- `RUN_ALL_MODELS` and `RUN_ALL_PROFILES` are mutually exclusive

---

## 🔗 Related Documentation

- [README.md](../README.md) - Main documentation
- [CHANGELOG_v18.md](CHANGELOG_v18.md) - v18 changes
- [WorkPlan.md](WorkPlan.md) - Development roadmap
