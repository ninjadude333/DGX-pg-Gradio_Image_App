# Quick Reference - v19.1

## 🎯 Model Quick Guide

### SDXL Models (6 models)
- **Dtype:** float16
- **VRAM:** 8-12 GB per batch
- **Resolution:** Up to 1536×1536
- **Batch:** 1-10
- **Img2Img:** ✅ Yes
- **Scheduler:** Any (Default, Euler, DPM++ 2M, UniPC)

### PixArt Sigma XL 1024
- **Dtype:** float32 (required)
- **VRAM:** 12-22 GB
- **Resolution:** 768×432 (safe) or 1024×1024 (24GB GPU)
- **Batch:** 1-2
- **Img2Img:** ❌ No
- **Scheduler:** Default (DPM-Solver) recommended
- **CFG:** 3.5-5.5 (never >6.0)
- **Steps:** 24-36
- **Prompt Style:** Emotion-driven, verbs, motion
- **Negative:** Minimal only

### SD3 Medium
- **Dtype:** float32 (required)
- **VRAM:** 16-28 GB (single GPU) or 8-14 GB/GPU (multi-GPU)
- **Resolution:** 768×432 (single GPU) or 1024×1024 (multi-GPU)
- **Batch:** 1 (single GPU) or 2-4 (multi-GPU)
- **Img2Img:** ❌ No
- **Scheduler:** Default (FlowMatchEulerDiscreteScheduler) recommended
- **Multi-GPU:** Auto-enabled with device_map="balanced"
- **Gated:** Requires HF authentication

---

## 🎨 Recommended Resolutions

### 16:9 Aspect Ratio
- **768×432** - Safe for all models
- **1024×576** - SDXL only
- **1536×864** - SDXL only (high VRAM)

### Square
- **768×768** - Safe for all models
- **1024×1024** - SDXL, PixArt (24GB), SD3 (multi-GPU)

### Portrait (9:16)
- **432×768** - Safe for all models
- **576×1024** - SDXL only

---

## 🚀 Quick Start Commands

### PixArt (Single GPU)
```bash
# UI Mode
sudo docker run --name pixart_gen \
  --gpus '"device=0"' --network host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e GRADIO_PORT=7864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  bash -lc "pip install sentencepiece && python3 gradio_app_multi-v19.py"
```

### SD3 (Multi-GPU)
```bash
# UI Mode with 3 GPUs
sudo docker run --name sd3_gen \
  --gpus '"device=0,1,2"' --network host --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -e HF_HUB_OFFLINE=0 \
  -e GRADIO_PORT=7864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  bash -lc "pip install sentencepiece && python3 gradio_app_multi-v19.py"
```

### SDXL (Any GPU)
```bash
# Standard SDXL generation
sudo docker run --name sdxl_gen \
  --gpus '"device=0"' --network host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e GRADIO_PORT=7864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v19.py
```

---

## ⚙️ Optimal Settings by Use Case

### Photoreal Portrait (SDXL)
- Model: CyberRealistic XL 5.8 or RealVis XL v5.0
- Profile: Photoreal or Hyper-Realistic Portrait
- Resolution: 768×1024 (portrait)
- Steps: 30-35
- CFG: 7.0-8.0
- Scheduler: DPM++ 2M

### Anime/Illustration (SDXL)
- Model: Animagine XL 4.0
- Profile: Anime / Vibrant
- Resolution: 1024×1024
- Steps: 28-32
- CFG: 7.0
- Scheduler: Euler

### Artistic/Conceptual (PixArt)
- Model: PixArt Sigma XL 1024
- Profile: None / Raw (PixArt doesn't need style profiles)
- Resolution: 768×432 or 1024×1024
- Steps: 28-32
- CFG: 4.0-5.0
- Scheduler: Default
- Prompt: "A serene woman gazing at sunset, peaceful emotion, soft movement"

### High Quality General (SD3)
- Model: SD3 Medium
- Profile: Photoreal or Cinematic
- Resolution: 768×432 (single GPU) or 1024×1024 (multi-GPU)
- Steps: 28
- CFG: 7.0
- Scheduler: Default

### Fast Generation (SDXL Turbo)
- Model: SDXL Turbo
- Profile: Any
- Resolution: 768×768
- Steps: 4-8
- CFG: 1.0-2.0
- Scheduler: Default

---

## 🔍 Troubleshooting

### "expected scalar type Float but found Half"
- **Cause:** PixArt or SD3 loaded in float16
- **Fix:** Already fixed in v19.1 (uses float32)

### "CUDA out of memory"
- **PixArt/SD3:** Reduce resolution to 768×432, batch to 1
- **SD3:** Use multi-GPU mode
- **SDXL:** Reduce batch size or resolution

### SD3 won't load
- **Check:** HF_HUB_OFFLINE=0 (SD3 needs online check)
- **Check:** HuggingFace authentication
- **Check:** Multiple GPUs visible

### PixArt generates low quality
- **Check:** CFG not >6.0
- **Check:** Using emotion/verb-based prompts
- **Check:** Minimal negative prompt

### Images not saving
- **Check:** output_images/ directory exists
- **Check:** Disk space available
- **Check:** [SAVE] logs for errors

---

## 📊 VRAM Quick Reference

| Model | Resolution | Batch | VRAM | GPU |
|-------|------------|-------|------|-----|
| SDXL | 1024×576 | 4 | ~12 GB | 1× 16GB |
| SDXL | 1024×1024 | 2 | ~10 GB | 1× 16GB |
| PixArt | 768×432 | 1 | ~12 GB | 1× 16GB |
| PixArt | 1024×1024 | 1 | ~22 GB | 1× 24GB |
| SD3 | 768×432 | 1 | ~16 GB | 1× 16GB |
| SD3 | 1024×1024 | 1 | ~12 GB/GPU | 3× 16GB |

---

## 🎨 Style Profile Recommendations

### Best for SDXL:
- Photoreal
- Cinematic
- Anime / Vibrant
- Tim Burton Style
- Frank Frazetta Fantasy
- H.R. Giger Biomechanical

### Best for PixArt:
- None / Raw (PixArt responds better to natural language)

### Best for SD3:
- Photoreal
- Cinematic
- Soft Illustration

---

**Quick Tip:** Start with Default scheduler and adjust only if needed. Most models work best with their built-in schedulers.
