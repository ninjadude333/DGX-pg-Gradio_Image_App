# SDXL DGX Image Lab - Master Changelog

All notable changes across all versions.

---

## v21 (Current) - Quick Wins & UX Improvements

**Release Date:** 2024-12-16

### New Features
- **Model name in output folders:**
  - Single model: `run_TIMESTAMP_modelslug/`
  - Multiple models: `run_TIMESTAMP_multi_models/`
- **Select All / Deselect All buttons** for models and profiles
- **3 New profiles:**
  - Sexy / Adult (sensual/erotic, 35 steps)
  - Porn / Explicit (hardcore/explicit, 40 steps)
  - LucasArts Point & Click (1990s adventure game, 28 steps)
- **Model info tooltips** showing type/VRAM/speed for each model
- **Estimated time display** - Calculates generation time based on selected models/steps/batch
- **Progress tracking** - Real-time progress bar during generation

### Technical
- Added `MODEL_INFO` dictionary with metadata (type, VRAM, speed, time_per_step)
- Added `estimate_time()` function for time calculations
- Added `format_time()` for human-readable time display
- Dynamic UI updates for model info and time estimates
- Integrated `gr.Progress()` for generation tracking
- **VAE slicing enabled for all models** - Prevents OOM during decode phase

### Bug Fixes
- Fixed OOM during VAE decode with large batch sizes (e.g., batch=6)

### Total Profiles: 32

---

## v20 - Checkbox Matrix Selection

**Release Date:** 2024-12-15

### New Features
- **Checkbox matrix for model selection** - Select multiple models simultaneously
- **Checkbox matrix for profile selection** - Select multiple profiles simultaneously
- **Extended widescreen resolutions:**
  - 16:9: 768×432, 1024×576, 1280×720
  - 21:9: 1024×440, 1280×544, 1536×656
  - 32:9: 1280×392, 1536×472
  - 2.35:1: 1024×432
  - Square: 512×512, 768×768, 1024×1024

### Removed
- "Run ALL models" checkbox (replaced by model checkboxes)
- "Run ALL profiles" checkbox (replaced by profile checkboxes)

### Technical
- Multi-GPU support for PixArt (device_map="balanced")
- One-at-a-time generation for SD3/PixArt multi-GPU to prevent corruption
- Fixed blank image issue with multi-GPU SD3

---

## v19 - SD3 Medium & Robust Saving

**Release Date:** 2024-12-14

### New Features
- **SD3 Medium model** (`stabilityai/stable-diffusion-3-medium-diffusers`)
  - Multi-GPU support via device_map="balanced"
  - Float32 required for stability
  - Text-to-image only
- **Robust image saving** with `_save_image_any()` helper
- **Improved directory creation** with `parents=True`

### Technical Changes
- SD3 uses `StableDiffusion3Pipeline`
- CPU generators for SD3 multi-GPU to avoid illegal memory access
- Auto-resolution reduction to 768×768 for PixArt/SD3 on single GPU
- Aggressive memory optimizations (attention slicing, VAE slicing, VAE tiling)

### Bug Fixes
- Fixed empty run folders
- Fixed img2img compatibility check for SD3/PixArt
- Fixed premature UI success message

### Known Limitations
- SD3 requires multi-GPU or >24GB VRAM (float32)
- PixArt requires float32 (APEX compatibility)
- Both models limited to 768×768 on single 16GB GPU

---

## v18 - PixArt Sigma Integration

**Release Date:** 2024-12-13

### New Features
- **PixArt Sigma XL 1024** model (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)
- **Model-type aware loading** (SDXL → AutoPipeline, PixArt → PixArtSigmaPipeline)
- **LRU pipeline cache** (default: 2 models, configurable via `PIPE_CACHE_MAX`)
- **V100 compatibility safeguards** (FlashAttention/Triton disabled)

### Technical Changes
- PixArt uses T5 encoder (AutoTokenizer instead of T5Tokenizer)
- Float32 required for PixArt (APEX compatibility)
- Attention slicing enabled for PixArt
- Explicit GC + empty_cache when switching models

### Documentation
- Created PIXART_INTEGRATION_GUIDE with optimal settings
- PixArt CFG: 3.5-5.5 (never exceed 6.0)
- PixArt Steps: 24-36
- PixArt prompts: Emotion-driven, verbs, motion (not keyword spam)

---

## v17 - Artist Profiles & Headless Mode

**Release Date:** 2024-12-12

### New Features
- **5 New Artist/Genre Profiles:**
  - Tim Burton Style
  - Frank Frazetta Fantasy
  - Ralph Bakshi Animation
  - H.R. Giger Biomechanical
  - Dark Fantasy / Grimdark
- **14 Extended v16 Profiles** (total: 29 profiles)
- **Headless mode** via environment variables
- **Mutually exclusive checkboxes** (Run ALL profiles OR Run ALL models)
- **Per-instance logging** with `INSTANCE_ID` environment variable

### Technical Changes
- Dimension validation (divisible by 8)
- Abort functionality
- Fallback model loading (fp16 variant → no variant)
- Lazy-loaded Img2Img pipeline
- Warning suppression for offline mode

### Bug Fixes
- Fixed seed mutation bug
- Fixed empty image validation
- Fixed img2img save bug

---

## v16 - Expanded Models & Profiles

### New Models
- DreamShaper XL (`Lykon/dreamshaper-xl-1-0`)
- EpicRealism XL (`AiAF/epicrealismXL-vx1Finalkiss_Checkpoint_SDXL`)
- Pixel Art XL (`nerijs/pixel-art-xl`)
- Anime Illust Diffusion XL (`Eugeoter/anime_illust_diffusion_xl`)

### New Profiles
- Pixel Art / Isometric Game
- Low-Poly 3D / PS1
- Product Render / Industrial
- Isometric Tech Diagram
- Retro Comic / Halftone
- Vaporwave / Synthwave
- Children's Book Illustration
- Ink & Screentone Manga
- Analog Horror / VHS
- Architectural Visualization

### Tools
- `download_models-v16.py` script for batch model downloads

---

## v15 - Smart OOM Handling

### Features
- **Auto-resolution downgrade** on OOM (768×768 → 576×576 → 512×512)
- **Verbose Img2Img logging**
- **Deadlock-free Img2Img pipeline loading**
- **Offline mode robustness** (`local_files_only=True` fallback)

---

## v14 - All Models × All Profiles Grid

### Features
- **All-models × all-profiles mode** - Single job iterates over every model and profile
- **Organized output** - Per-model directories in run folder
- **Comprehensive logging** - Each combination logged with metadata

---

## v13 - Multi-Profile Batch Loop

### Features
- **do_all_profiles** - Iterate over all profiles for single model
- **Dedicated multi-profile run folders**
- **Profile sweep logging**

---

## v12 - UI Improvements

### Features
- **Negative prompt visualization** - Show effective negative prompt
- **VRAM warnings** - Highlight heavy models
- **New profiles:**
  - Watercolor
  - Hyper-Realistic Portrait
  - ISOTOPIA Sci-Fi Blueprint
  - Dark Fantasy / Grimdark
  - Pixar-ish Soft CG

---

## v11 - Single-GPU Stable Baseline

### Architecture
- **Single-GPU only** (removed multi-GPU for stability)
- **6 Working SDXL models:**
  - SDXL Base 1.0
  - SDXL Turbo
  - RealVis XL v5.0
  - CyberRealistic XL 5.8
  - Animagine XL 4.0
  - Juggernaut XL

### Features
- **4 Schedulers:** Default, Euler, DPM++ 2M, UniPC
- **9 Style presets**
- **Txt2Img + lazy-loaded Img2Img**
- **Auto-save with JSON logging**
- **Aspect ratio presets**

### Bug Fixes
- Multi-GPU instability resolved
- Juggernaut model shards fixed
- Gradio return-type error fixed
- Negative prompt update bug fixed
- Generate button state fixed

---

## v1-v10 - Early Experiments

- Initial Gradio SDXL app setup
- Early SDXL Base and Turbo support
- Dockerization and DGX GPU access
- Multi-GPU experiments (later simplified)

---

## Model Support Summary

### Current Models (v20)
| Model | Type | Img2Img | Multi-GPU | Notes |
|-------|------|---------|-----------|-------|
| SDXL Base 1.0 | SDXL | ✅ | ❌ | Baseline |
| SDXL Turbo | SDXL | ✅ | ❌ | Fast (4-8 steps) |
| RealVis XL v5.0 | SDXL | ✅ | ❌ | Photorealistic |
| CyberRealistic XL 5.8 | SDXL | ✅ | ❌ | Portraits |
| Animagine XL 4.0 | SDXL | ✅ | ❌ | Anime |
| Juggernaut XL | SDXL | ✅ | ❌ | General purpose |
| PixArt Sigma XL 1024 | PixArt | ❌ | ✅ | High quality, float32 |
| SD3 Medium | SD3 | ❌ | ✅ | Latest, float32 |

### Style Profiles (32 total in v21)
- Core: None/Raw, Photoreal, Cinematic, Anime, Soft Illustration, B&W, Pencil Sketch, 35mm Film, Rotoscoping, R-Rated
- Artist: Tim Burton, Frank Frazetta, Ralph Bakshi, H.R. Giger, Dark Fantasy
- Extended: Watercolor, Hyper-Realistic Portrait, ISOTOPIA Sci-Fi, Pixar-ish, Pixel Art, Low-Poly 3D, Product Render, Isometric Tech, Retro Comic, Vaporwave, Children's Book, Ink & Screentone, Analog Horror, Architectural Viz
- Adult: Sexy/Adult, Porn/Explicit
- Retro Gaming: LucasArts Point & Click

---

## Technical Evolution

### GPU Strategy
- **v1-v10:** Multi-GPU experiments
- **v11-v17:** Single-GPU only (stability)
- **v18-v19:** Single-GPU for SDXL, multi-GPU for PixArt/SD3
- **v20:** Multi-GPU device_map for PixArt and SD3

### Memory Management
- **v11-v17:** Basic VRAM management
- **v18:** Attention slicing for PixArt
- **v19:** VAE slicing + tiling for SD3/PixArt
- **v20:** device_map="balanced" for multi-GPU distribution

### Dtype Strategy
- **v11-v17:** float16 for all models
- **v18-v20:** float32 for PixArt/SD3 (APEX compatibility)

---

## Known Issues & Limitations

### Current (v20)
- PixArt/SD3 require float32 (2x VRAM vs float16)
- PixArt/SD3 no img2img support
- SD3 is gated model (requires HF authentication)
- Proxy/reverse proxy may break image display in gallery
- Multi-GPU SD3/PixArt generates one image at a time (slower batching)

---

## Breaking Changes

### v20
- Removed `do_all_models` and `do_all_profiles` parameters
- Changed `generate_images()` signature to accept lists

### v19
- Added SD3 model type (requires new dependencies)

### v18
- Added PixArt model type (requires sentencepiece)

### v17
- Changed logging to per-instance files

---

**Current Version:** v21  
**Status:** ✅ Production Ready  
**Last Updated:** 2024-12-16
