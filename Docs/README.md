# SDXL DGX Image Lab Documentation

**Current Version:** v21  
**Last Updated:** 2024-12-16

---

## 📚 Documentation Index

### Core Documentation
- **[CHANGELOG_MASTER.md](CHANGELOG_MASTER.md)** - Complete version history (v1-v20)
- **[ROADMAP.md](ROADMAP.md)** - Future plans and next steps
- **[QUICK_REFERENCE_v19.md](QUICK_REFERENCE_v19.md)** - Quick lookup guide for settings

### Docker & Deployment
- **[DOCKER_EXAMPLES_v17.md](DOCKER_EXAMPLES_v17.md)** - Docker run examples (still valid for v20)
- **[DOCKER_EXAMPLES_v18.md](DOCKER_EXAMPLES_v18.md)** - Docker examples with PixArt

### Model-Specific Guides
- **[PIXART_INTEGRATION_GUIDE_v18.md](PIXART_INTEGRATION_GUIDE_v18.md)** - Complete PixArt Sigma guide
- **[README_v18.md](README_v18.md)** - v18 README (PixArt introduction)

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA DGX or GPU server with CUDA
- Docker with NVIDIA runtime
- Pre-downloaded models in HuggingFace cache

### Run v21
```bash
sudo docker run --name image_gen_v21 \
  --gpus all \
  --network host \
  --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e HF_HUB_OFFLINE=1 \
  -e INSTANCE_ID=dgx_gpu01 \
  -e GRADIO_PORT=7860 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app \
  -w /app \
  gradio_app_generic:dude \
  bash -lc "pip install sentencepiece && python3 gradio_app_multi-v21.py"
```

Access: `http://<hostname>:7860`

---

## 📖 What's New in v21

### Quick Wins & UX Improvements
- **Model name in output folders** - Single model: `run_TIMESTAMP_modelslug/`, Multiple: `run_TIMESTAMP_multi_models/`
- **Select All / Deselect All buttons** - Quick selection for models and profiles
- **3 New profiles** - Sexy/Adult, Porn/Explicit, LucasArts Point & Click (32 total)
- **Model info tooltips** - Shows type/VRAM/speed for each model
- **Estimated time display** - Calculates generation time before running
- **Progress tracking** - Real-time progress bar during generation

---

## 🎨 Supported Models (8 total)

| Model | Type | Best For | Img2Img | Multi-GPU |
|-------|------|----------|---------|-----------|
| SDXL Base 1.0 | SDXL | General purpose | ✅ | ❌ |
| SDXL Turbo | SDXL | Fast generation | ✅ | ❌ |
| RealVis XL v5.0 | SDXL | Photorealism | ✅ | ❌ |
| CyberRealistic XL 5.8 | SDXL | Portraits | ✅ | ❌ |
| Animagine XL 4.0 | SDXL | Anime | ✅ | ❌ |
| Juggernaut XL | SDXL | Versatile | ✅ | ❌ |
| PixArt Sigma XL 1024 | PixArt | High quality | ❌ | ✅ |
| SD3 Medium | SD3 | Latest tech | ❌ | ✅ |

---

## 🎭 Style Profiles (32 total)

### Core Styles (10)
None/Raw, Photoreal, Cinematic, Anime/Vibrant, Soft Illustration, Black & White, Pencil Sketch, 35mm Film, Rotoscoping, R-Rated

### Artist Styles (5)
Tim Burton, Frank Frazetta Fantasy, Ralph Bakshi Animation, H.R. Giger Biomechanical, Dark Fantasy/Grimdark

### Extended Styles (14)
Watercolor, Hyper-Realistic Portrait, ISOTOPIA Sci-Fi Blueprint, Pixar-ish Soft CG, Pixel Art/Isometric Game, Low-Poly 3D/PS1, Product Render/Industrial, Isometric Tech Diagram, Retro Comic/Halftone, Vaporwave/Synthwave, Children's Book Illustration, Ink & Screentone Manga, Analog Horror/VHS, Architectural Visualization

### Adult Styles (2)
Sexy/Adult, Porn/Explicit

### Retro Gaming (1)
LucasArts Point & Click

---

## ⚙️ Optimal Settings by Model

### SDXL Models
- **Resolution:** 512×512 to 1536×1536
- **Steps:** 20-40
- **CFG:** 7.0-8.0
- **Batch:** 1-10
- **Scheduler:** Any

### PixArt Sigma
- **Resolution:** 768×432 or 1024×1024 (with 24GB GPU)
- **Steps:** 24-36
- **CFG:** 3.5-5.5 (never exceed 6.0)
- **Batch:** 1-2
- **Scheduler:** Default (DPM-Solver)
- **Prompts:** Emotion-driven, verbs, motion

### SD3 Medium
- **Resolution:** 768×432 (single GPU) or 1024×1024 (multi-GPU)
- **Steps:** 28
- **CFG:** 7.0
- **Batch:** 1-4 (multi-GPU)
- **Scheduler:** Default (FlowMatchEulerDiscreteScheduler)

---

## 🐛 Troubleshooting

### "CUDA out of memory"
- **SDXL:** Reduce batch size or resolution
- **PixArt/SD3:** Use 2+ GPUs or reduce to 768×432
- **All:** Close other GPU processes

### "No CUDA-capable device detected"
- Use `--gpus all` instead of `--gpus "device=X"`
- Check `CUDA_VISIBLE_DEVICES` is set correctly

### Images display as broken links in UI
- **Cause:** Reverse proxy configuration
- **Fix:** Access Gradio directly or configure proxy `/file/` paths

### SD3/PixArt images are corrupted
- **Cause:** Multi-GPU with batched generation
- **Fix:** Already fixed in v20 (one-at-a-time generation)

### "expected scalar type Float but found Half"
- **Cause:** PixArt/SD3 dtype mismatch
- **Fix:** Already fixed in v19+ (uses float32)

---

## 📊 VRAM Requirements

### Single GPU
| Model | Resolution | Batch | VRAM |
|-------|------------|-------|------|
| SDXL | 1024×576 | 4 | ~12 GB |
| SDXL | 1024×1024 | 2 | ~10 GB |
| PixArt | 768×432 | 1 | ~12 GB |
| SD3 | 768×432 | 1 | ~16 GB |

### Multi-GPU (2-3 GPUs)
| Model | Resolution | Batch | VRAM/GPU |
|-------|------------|-------|----------|
| PixArt | 1024×1024 | 1 | ~8-10 GB |
| SD3 | 1024×1024 | 1 | ~10-12 GB |
| SD3 | 768×432 | 4 | ~10-12 GB |

---

## 🔗 Related Files

### Application Files
- `gradio_app_multi-v21.py` - Current version
- `gradio_app_multi-v20.py` - Previous stable
- `download_models_v19.py` - Model download script

### Output
- `output_images/` - Generated images
- `output_images/jobs_<instance>.log` - Generation logs

---

## 📝 Environment Variables

### Required
- `CUDA_VISIBLE_DEVICES` - GPU selection (e.g., "0,1,2")
- `HF_HOME` - HuggingFace cache path

### Optional
- `HF_HUB_OFFLINE` - Offline mode (0 or 1)
- `INSTANCE_ID` - Instance identifier for logs
- `GRADIO_PORT` - Web UI port (default: 7865)
- `PIPE_CACHE_MAX` - Pipeline cache size (default: 2)

---

## 🎯 Next Steps

See **[ROADMAP.md](ROADMAP.md)** for:
- v22: Parallel multi-model execution
- v23: Smart batch adjustment
- v24+: LoRA, ControlNet, Video generation

---

## 📞 Support

- **Issues:** Check troubleshooting section above
- **Feature Requests:** See ROADMAP.md
- **Documentation:** This folder

---

**Version:** v21  
**Status:** ✅ Production Ready  
**Maintainer:** DGX Lab Team
