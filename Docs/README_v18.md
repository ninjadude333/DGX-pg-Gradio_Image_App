# SDXL DGX Image Lab v18 🚀

Local, offline-friendly SDXL + PixArt image generation lab optimized for single-GPU DGX environment.

This app runs inside a Docker container on an NVIDIA DGX system, uses PyTorch + Diffusers + Gradio, and loads models from a pre-populated Hugging Face cache. It focuses on **stability**, **observability**, and **repeatability**.

---

## ✅ Current Status (v18)

**Core characteristics:**

- ✅ Single-GPU optimized (multi-GPU intentionally removed)
- ✅ Local-only models (no runtime internet access required)
- ✅ Uses Hugging Face cache mounted at: `/root/.cache/huggingface`
- ✅ Gradio-based web UI served from inside a Docker container
- ✅ All generations auto-saved + logged (`jobs_{INSTANCE_ID}.log`)
- ✅ **NEW:** PixArt-Σ model support with model-type aware loading
- ✅ **NEW:** Smart LRU pipeline cache for faster model switching

**Supported models (7 total):**

| Model | Type | Img2Img | Notes |
|-------|------|---------|-------|
| SDXL Base 1.0 | SDXL | ✅ | Baseline model |
| SDXL Turbo | SDXL | ✅ | Fast inference (4-8 steps) |
| RealVis XL v5.0 | SDXL | ✅ | Photorealistic |
| CyberRealistic XL 5.8 | SDXL | ✅ | Realistic portraits |
| Animagine XL 4.0 | SDXL | ✅ | Anime style |
| Juggernaut XL | SDXL | ✅ | Versatile general purpose |
| **PixArt Sigma XL 1024** | **PixArt** | **❌** | **High quality, txt2img only** |

**Available schedulers:**

- Default (whatever the pipeline ships with)
- Euler
- DPM++ 2M
- UniPC

**Main UI modes:**

- **Txt2Img** (all models)
- **Img2Img** (SDXL models only)

---

## ✨ What's New in v18

### 1. PixArt-Σ (PixArt Sigma XL 1024) Model

**New model added:**
- Model ID: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`
- Uses T5 text encoder (instead of CLIP)
- Optimized for 1024×1024 resolution
- Text-to-image only (no img2img in this app)

**Why PixArt?**
- Higher quality outputs than SDXL in many cases
- Better prompt understanding via T5 encoder
- More literal interpretation of prompts
- Excellent for detailed scenes and portraits

**Optimal settings:**
- Resolution: 1024×1024 or 1024×576
- Steps: 15-25 (lower than SDXL)
- CFG Scale: 4.0-5.0
- Batch: 2-4 (uses more VRAM than SDXL)

### 2. Model-Type Aware Loading

**v17 (Before):**
- All models loaded via `AutoPipelineForText2Image`
- PixArt would fail or load incorrectly

**v18 (After):**
- SDXL models → `AutoPipelineForText2Image`
- PixArt-Σ → `PixArtSigmaPipeline`
- Prevents misloading and ensures compatibility

### 3. Smart Pipeline Cache (LRU)

**New caching system:**
- Default cache size: 2 pipelines
- Configurable via `PIPE_CACHE_MAX` env var
- LRU eviction (least recently used)

**Benefits:**
- Switching back to recent model = instant load
- No disk I/O for cached models
- Automatic memory management

**Example workflow:**
1. Load SDXL Base → 120s load time
2. Load PixArt Sigma → 90s load time
3. Switch back to SDXL Base → **instant** (from cache)
4. Switch back to PixArt → **instant** (from cache)
5. Load 3rd model → oldest evicted, new model loaded

**Cache configuration:**
```bash
# Default (2 models)
-e PIPE_CACHE_MAX=2

# Larger cache (3-4 models)
-e PIPE_CACHE_MAX=4

# Disable cache (always reload)
-e PIPE_CACHE_MAX=0
```

### 4. PixArt-Specific Safeguards

**V100 GPU compatibility:**
- FlashAttention disabled automatically
- Triton disabled automatically
- Attention slicing enabled for stability

**Environment variables set automatically:**
```bash
XFORMERS_FORCE_DISABLE_TRITON=1
XFORMERS_DISABLE_FLASH_ATTN=1
DISABLE_FLASH_ATTN=1
```

**Precision defaults:**
- `dtype=torch.float16`
- `enable_attention_slicing()`
- Prevents fp16 RMSNorm crashes

---

## 🎨 Features (Carried from v17)

### Artist/Genre Style Profiles (29 Total)

**Core Profiles (10):**
- None / Raw, Photoreal, Cinematic, Anime / Vibrant, Soft Illustration
- Black & White, Pencil Sketch, 35mm Film, Rotoscoping, R-Rated

**Artist/Genre Profiles (5):**
- **Tim Burton Style** — Gothic, dark whimsical, striped patterns
- **Frank Frazetta Fantasy** — Epic fantasy illustration, muscular heroes
- **Ralph Bakshi Animation** — 1970s rotoscoped animation style
- **H.R. Giger Biomechanical** — Alien, biomechanical, nightmarish beauty
- **Dark Fantasy / Grimdark** — Ominous atmosphere, gothic horror

**Extended Profiles (14):**
- Watercolor, Hyper-Realistic Portrait, ISOTOPIA Sci-Fi Blueprint
- Pixar-ish Soft CG, Pixel Art / Isometric Game, Low-Poly 3D / PS1
- Product Render / Industrial, Isometric Tech Diagram, Retro Comic / Halftone
- Vaporwave / Synthwave, Children's Book Illustration, Ink & Screentone Manga
- Analog Horror / VHS, Architectural Visualization

### Headless Mode

Run generations without UI via environment variables:

```bash
-e HEADLESS=true \
-e PROMPT="a futuristic cityscape" \
-e MODEL="PixArt Sigma XL 1024" \
-e STYLE_PROFILE="Cinematic" \
-e BATCH_SIZE=4
```

See [DOCKER_EXAMPLES_v18.md](Docs/DOCKER_EXAMPLES_v18.md) for complete reference.

### Multi-Generation Modes

**Run all profiles for one model:**
```bash
-e RUN_ALL_PROFILES=true
```
Generates with all 29 style profiles.

**Run all models × all profiles:**
```bash
-e RUN_ALL_MODELS=true
```
Generates with all 7 models × all 29 profiles = 203 combinations.

*Note: These flags are mutually exclusive for safety.*

### Logging & Output

**All generations:**
- Saved under `output_images/run_YYYYMMDD_HHMMSS/`
- Filenames: `YYYYMMDD_HHMMSS_profile_prompt_seedXXXXXX_01.png`
- Logged to `output_images/jobs_{INSTANCE_ID}.log` as JSON

**Log includes:**
- Prompt, styled prompt, negative prompt
- Model, scheduler, steps, guidance scale
- Resolution, batch size, seeds
- Output file paths
- Mode (txt2img / img2img)
- Instance ID

---

## 🚀 Quick Start

### 1. Docker Run (UI Mode)

```bash
sudo docker run --name image_gen_v18 \
  --gpus all --runtime=nvidia --network host \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e INSTANCE_ID=dgx_gpu3 \
  -e HF_HUB_OFFLINE=1 \
  -e PIPE_CACHE_MAX=2 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

Access UI at: `http://<DGX-HOSTNAME>:7865`

### 2. Docker Run (Headless Mode)

```bash
sudo docker run --name image_gen_v18_headless \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HEADLESS=true \
  -e PROMPT="a serene mountain landscape at sunset" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e STEPS=20 \
  -e BATCH_SIZE=4 \
  -e INSTANCE_ID=pixart_job_1 \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 📦 Requirements

### Runtime

- **Python**: 3.10+ (NVIDIA PyTorch container)
- **GPU**: CUDA-capable NVIDIA GPU (DGX class)
- **VRAM**:
  - SDXL: 12-16 GB recommended
  - PixArt: 14-18 GB recommended
  - Higher is better for large batches

### Python Packages

Installed in container:
- `torch` (PyTorch 2.3.x)
- `diffusers` (with PixArtSigmaPipeline support)
- `transformers`
- `accelerate`
- `safetensors`
- `gradio`
- `Pillow`
- `numpy`
- `xformers` (optional, for memory optimization)

### Models

Models **must be pre-downloaded** into HuggingFace cache:
- `/root/.cache/huggingface/hub/...`

Runtime does **not** download models (offline mode).

---

## 🎛️ Environment Variables

### Core Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `HEADLESS` | `false` | Enable headless mode (no UI) |
| `PROMPT` | *required* | Generation prompt |
| `MODEL` | `"SDXL Base 1.0"` | Model to use |
| `STYLE_PROFILE` | `"Photoreal"` | Style profile |
| `INSTANCE_ID` | `"default"` | Instance identifier for logging |
| `PIPE_CACHE_MAX` | `2` | **v18 NEW:** Max cached pipelines |

### Generation Parameters

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `STEPS` | `26` | 4-80 | Inference steps |
| `CFG_SCALE` | `7.5` | 0.0-20.0 | Guidance scale |
| `WIDTH` | `1024` | 256-1536 | Image width (÷8) |
| `HEIGHT` | `576` | 256-1536 | Image height (÷8) |
| `BATCH_SIZE` | `4` | 1-10 | Images per generation |
| `SEED` | `-1` | -1 or 0-2³² | Random seed |

See [DOCKER_EXAMPLES_v18.md](Docs/DOCKER_EXAMPLES_v18.md) for complete reference.

---

## 🔧 Performance Tips

### Model-Specific Settings

**SDXL Base / Juggernaut:**
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

## 🐛 Troubleshooting

### PixArt Out of Memory

**Reduce batch size:**
```bash
-e BATCH_SIZE=2
```

**Use smaller resolution:**
```bash
-e WIDTH=768 -e HEIGHT=768
```

### Cache Issues

**Disable cache:**
```bash
-e PIPE_CACHE_MAX=0
```

**Increase cache:**
```bash
-e PIPE_CACHE_MAX=4
```

### Model Not Found

Ensure model is downloaded in HuggingFace cache:
```bash
ls -la /root/.cache/huggingface/hub/
```

---

## 📚 Documentation

- [CHANGELOG_v18.md](Docs/CHANGELOG_v18.md) - v18 changes
- [DOCKER_EXAMPLES_v18.md](Docs/DOCKER_EXAMPLES_v18.md) - Docker usage examples
- [WorkPlan_v18.md](Docs/WorkPlan_v18.md) - Development roadmap
- [CHANGELOG_v17.md](Docs/CHANGELOG_v17.md) - v17 changes

---

## 🔮 Roadmap

**v19 (Planned):**
- GPU detection and idle scheduling
- Multi-GPU orchestration scripts

**v20 (Planned):**
- Automated prompt generator
- Overnight continuous runner

**v21+ (Future):**
- Favorites-based analytics
- Optional 3D exploration track

---

## 📝 License & Notes

This app depends on:
- NVIDIA PyTorch container license
- Hugging Face model licenses for each checkpoint
- PixArt-α team for PixArt-Σ model

Before sharing or commercializing:
- Check each model's license and TOS
- Ensure compliance with NVIDIA / HF terms

---

**Happy generating! ✨**
