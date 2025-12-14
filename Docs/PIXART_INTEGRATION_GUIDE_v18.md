# PixArt-Σ Integration Guide v18 🎨

Complete reference for PixArt-Σ (PixArt Sigma XL 1024) integration, presets, and best practices.

---

## 📋 Quick Reference

**Model:** PixArt-Σ (PixArt Sigma XL 1024)  
**HuggingFace ID:** `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`  
**Architecture:** Diffusion Transformer (DiT) with T5 text encoder  
**Task:** Text-to-image ONLY (no img2img, no inpainting)

---

## 🔑 Critical Differences from SDXL

| Aspect | SDXL | PixArt-Σ |
|--------|------|----------|
| Architecture | UNet | Diffusion Transformer (DiT) |
| Text Encoder | CLIP | T5 |
| Optimal CFG | 7.0-8.0 | 3.5-5.5 |
| Optimal Steps | 25-35 | 24-36 |
| Resolution | Flexible | Square-first (1024×1024) |
| Prompt Style | Keyword-heavy | Verb/emotion-driven |
| Negative Prompts | Long lists | Minimal |
| Multi-GPU | Beneficial | Not beneficial per image |

---

## 🚀 Pipeline Loading Rules

### ✅ Correct Loading

```python
from diffusers import PixArtSigmaPipeline

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=torch.float16,
    use_safetensors=True,
    clean_caption=False,
).to("cuda")

# ALWAYS enable attention slicing
pipe.enable_attention_slicing()
```

### ❌ Incorrect Loading

```python
# DO NOT USE AutoPipeline for PixArt
pipe = AutoPipelineForText2Image.from_pretrained(...)  # WRONG

# DO NOT load without attention slicing
# pipe.enable_attention_slicing()  # REQUIRED
```

---

## 🛡️ V100 GPU Compatibility

### Required Environment Variables

**Set before loading PixArt:**

```bash
export XFORMERS_FORCE_DISABLE_TRITON=1
export XFORMERS_DISABLE_FLASH_ATTN=1
export DISABLE_FLASH_ATTN=1
```

**In Docker:**

```bash
-e XFORMERS_FORCE_DISABLE_TRITON=1 \
-e XFORMERS_DISABLE_FLASH_ATTN=1 \
-e DISABLE_FLASH_ATTN=1
```

### Why These Are Required

- **FlashAttention:** Causes crashes on V100 with PixArt
- **Triton:** Incompatible with PixArt's attention mechanism
- **Apex:** Must not be present (removed at container level)

### Stability Defaults

```python
dtype=torch.float16  # Required
pipe.enable_attention_slicing()  # Required
```

Prevents fp16 RMSNorm crashes and reduces VRAM spikes.

---

## 🎛️ Generation Presets

### CFG Scale (Guidance Scale)

**PixArt is VERY sensitive to CFG. High CFG degrades quality.**

| Use Case | CFG Range | Recommended |
|----------|-----------|-------------|
| Emotional/Abstract | 3.8-4.2 | 4.0 |
| Balanced/Default | 4.0-5.0 | 4.5 |
| Structured Scenes | 5.0-5.5 | 5.2 |

**⚠️ NEVER exceed CFG 6.0 with PixArt**

### Step Count

**PixArt converges faster than SDXL. Too many steps reduce clarity.**

| Quality Level | Steps | Use Case |
|---------------|-------|----------|
| Preview | 24 | Quick tests |
| Quality | 28-30 | Standard generation |
| Max Quality | 32-36 | Final outputs |

**⚠️ More than 40 steps degrades quality**

### Resolution Strategy

**PixArt performs best at square resolutions.**

#### Primary Generation Size

```bash
WIDTH=1024
HEIGHT=1024
```

#### Cinematic Output (Best Practice)

**Option 1: Generate square, crop afterward**
```bash
# Generate
WIDTH=1024 HEIGHT=1024

# Post-process: crop to 1024×576
```

**Option 2: Generate wide directly**
```bash
WIDTH=1024 HEIGHT=576

# Add strong horizontal/cinematic prompt hints
# Example: "wide cinematic shot, panoramic view, horizontal composition"
```

#### Supported Resolutions

| Resolution | Quality | Notes |
|------------|---------|-------|
| 1024×1024 | ⭐⭐⭐⭐⭐ | Optimal |
| 1024×576 | ⭐⭐⭐⭐ | Good with cinematic prompts |
| 768×768 | ⭐⭐⭐ | Lower quality |
| 512×512 | ⭐⭐ | Preview only |

---

## 📝 Prompt Engineering

### ✅ What Works Best

**PixArt responds to:**
- **Verbs** (flowing, dancing, emerging, cascading)
- **Motion** (swirling, drifting, rising, falling)
- **Emotion** (serene, ominous, joyful, melancholic)
- **Color** (golden, azure, crimson, emerald)
- **Spatial dynamics** (foreground, background, depth, layers)

**Example good prompts:**
```
"a serene mountain landscape with golden light cascading through clouds"
"a dancer gracefully spinning in flowing blue fabric"
"an ominous castle emerging from swirling mist"
```

### ❌ What Works Poorly

**PixArt responds poorly to:**
- SDXL-style keyword spam
- Long noun lists
- Camera jargon ("8k, ultra-detailed, masterpiece")
- Technical terms ("photorealistic, highly detailed, sharp focus")

**Example bad prompts:**
```
"8k, ultra-detailed, masterpiece, best quality, highly detailed, sharp focus"
"professional photography, studio lighting, bokeh, HDR"
```

### Prompt Bias Rule

**When using PixArt:**
- Prefer **verbs + emotion** over **adjectives + nouns**
- Focus on **what's happening** not **what it looks like**
- Describe **feeling and flow** not **technical specs**

---

## 🚫 Negative Prompt Rules

### PixArt Needs Minimal Negatives

**Long SDXL negative lists REDUCE quality with PixArt.**

#### ✅ Recommended Negative Prompt

```
low quality, distorted, broken composition
```

#### ❌ Avoid SDXL-Style Negatives

```
# DO NOT USE with PixArt:
extra fingers, bad anatomy, bad hands, text, error, missing fingers, 
extra digit, fewer digits, cropped, worst quality, low quality, 
normal quality, jpeg artifacts, signature, watermark, username, blurry
```

**Why?** PixArt's T5 encoder interprets negatives differently than CLIP. Excessive negatives confuse the model.

---

## 📊 Canonical PixArt Preset

### Default Configuration

```json
{
  "model": "PixArt-Σ",
  "model_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
  "cfg": 4.5,
  "steps": 30,
  "resolution": "1024x1024",
  "post_process": "crop_to_1024x576",
  "prompt_bias": "motion_emotion",
  "negative_prompt": "low quality, distorted",
  "attention_slicing": true,
  "dtype": "float16"
}
```

### Docker Command (Canonical)

```bash
sudo docker run --name pixart_canonical \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HEADLESS=true \
  -e PROMPT="a serene mountain landscape with golden light cascading through clouds" \
  -e NEGATIVE_PROMPT="low quality, distorted" \
  -e MODEL="PixArt Sigma XL 1024" \
  -e STEPS=30 \
  -e CFG_SCALE=4.5 \
  -e WIDTH=1024 \
  -e HEIGHT=1024 \
  -e BATCH_SIZE=2 \
  -e INSTANCE_ID=pixart_canonical \
  -e HF_HUB_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v18.py
```

---

## 🎯 Use Case Presets

### Emotional/Abstract Art

```bash
-e CFG_SCALE=4.0
-e STEPS=28
-e PROMPT="swirling colors dancing in ethereal space, dreamlike atmosphere"
```

### Structured Scenes

```bash
-e CFG_SCALE=5.2
-e STEPS=32
-e PROMPT="a grand cathedral with light streaming through stained glass windows"
```

### Cinematic Landscapes

```bash
-e CFG_SCALE=4.5
-e STEPS=30
-e WIDTH=1024
-e HEIGHT=1024
-e PROMPT="wide cinematic shot of a mountain range at golden hour, panoramic view"
```

### Character Portraits

```bash
-e CFG_SCALE=4.8
-e STEPS=30
-e WIDTH=1024
-e HEIGHT=1024
-e PROMPT="a wise elder with flowing robes, serene expression, soft lighting"
```

---

## 💾 Pipeline Caching Rules

### Smart Loading Strategy

**PixArt must be loaded lazily and cached separately from SDXL.**

```python
# LRU cache size
_PIPE_CACHE_MAX = 2  # Default

# Cache key
cache_key = (model_key, scheduler_name)

# On eviction
del pipeline
torch.cuda.empty_cache()
gc.collect()
```

### Cache Configuration

**2-Model Workflow (SDXL + PixArt):**
```bash
-e PIPE_CACHE_MAX=2  # Default
```

**3-Model Workflow:**
```bash
-e PIPE_CACHE_MAX=3
```

**No Cache (Always Reload):**
```bash
-e PIPE_CACHE_MAX=0
```

### VRAM Sharing Rule

**⚠️ PixArt should NOT share VRAM with SDXL pipelines simultaneously.**

Cache ensures only one pipeline is active at a time.

---

## 🖥️ Multi-GPU Expectations

### Single Image Generation

**PixArt does NOT benefit from multi-GPU for a single image.**

Use 1 GPU per worker:
```bash
-e CUDA_VISIBLE_DEVICES=0  # Single GPU
```

### Concurrent Jobs

**Multi-GPU usage is for:**
- Concurrent jobs (multiple containers)
- Batch splitting (multiple images)

**Example: 4 concurrent PixArt jobs**
```bash
# Container 1
-e CUDA_VISIBLE_DEVICES=0

# Container 2
-e CUDA_VISIBLE_DEVICES=1

# Container 3
-e CUDA_VISIBLE_DEVICES=2

# Container 4
-e CUDA_VISIBLE_DEVICES=3
```

---

## 🔧 Performance Optimization

### VRAM Usage

| Batch Size | Resolution | VRAM Usage |
|------------|------------|------------|
| 1 | 1024×1024 | ~12 GB |
| 2 | 1024×1024 | ~16 GB |
| 4 | 1024×1024 | ~24 GB |
| 1 | 1024×576 | ~10 GB |
| 2 | 1024×576 | ~14 GB |

### Recommended Batch Sizes

| GPU VRAM | Batch Size (1024×1024) | Batch Size (1024×576) |
|----------|------------------------|------------------------|
| 16 GB | 2 | 3 |
| 24 GB | 4 | 6 |
| 32 GB | 6 | 8 |
| 40 GB | 8 | 10 |

### Speed Optimization

**Faster generation:**
```bash
-e STEPS=24  # Minimum quality
-e BATCH_SIZE=1
```

**Quality generation:**
```bash
-e STEPS=30  # Balanced
-e BATCH_SIZE=2
```

**Maximum quality:**
```bash
-e STEPS=36  # Maximum
-e BATCH_SIZE=1
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Reduce batch size:**
```bash
-e BATCH_SIZE=1
```

**Use smaller resolution:**
```bash
-e WIDTH=768 -e HEIGHT=768
```

**Ensure attention slicing is enabled:**
```python
pipe.enable_attention_slicing()  # Automatic in v18
```

### Poor Quality Output

**Check CFG scale:**
```bash
# Too high?
-e CFG_SCALE=4.5  # Reduce from 7.5+
```

**Check step count:**
```bash
# Too many steps?
-e STEPS=30  # Reduce from 50+
```

**Check prompt style:**
```bash
# Remove SDXL keywords
PROMPT="flowing water cascading over rocks"  # Good
# Not: "8k, ultra-detailed, masterpiece, flowing water"  # Bad
```

### Slow Loading

**Enable pipeline cache:**
```bash
-e PIPE_CACHE_MAX=2  # Cache PixArt + 1 SDXL model
```

**Check environment variables:**
```bash
# Ensure these are set
-e XFORMERS_FORCE_DISABLE_TRITON=1
-e XFORMERS_DISABLE_FLASH_ATTN=1
```

---

## 📊 Comparison: SDXL vs PixArt

### When to Use SDXL

- Need img2img or inpainting
- Keyword-heavy prompts
- Flexible aspect ratios
- LoRA support needed
- Lower VRAM available

### When to Use PixArt

- Maximum quality txt2img
- Emotion/motion-driven scenes
- Square or near-square outputs
- T5 prompt understanding
- Faster convergence (fewer steps)

---

## 📚 Example Workflows

### Workflow 1: Cinematic Landscape

```bash
# Generate square
-e MODEL="PixArt Sigma XL 1024"
-e PROMPT="wide panoramic view of a mountain range at golden hour, dramatic clouds"
-e CFG_SCALE=4.5
-e STEPS=30
-e WIDTH=1024
-e HEIGHT=1024

# Post-process: crop to 1024×576
```

### Workflow 2: Character Portrait

```bash
-e MODEL="PixArt Sigma XL 1024"
-e PROMPT="a wise wizard with flowing robes, serene expression, soft magical glow"
-e NEGATIVE_PROMPT="low quality, distorted"
-e CFG_SCALE=4.8
-e STEPS=32
-e WIDTH=1024
-e HEIGHT=1024
```

### Workflow 3: Abstract Art

```bash
-e MODEL="PixArt Sigma XL 1024"
-e PROMPT="swirling colors dancing in cosmic space, ethereal and dreamlike"
-e CFG_SCALE=4.0
-e STEPS=28
-e WIDTH=1024
-e HEIGHT=1024
```

---

## 🔗 Related Documentation

- [CHANGELOG_v18.md](CHANGELOG_v18.md) - v18 changes
- [DOCKER_EXAMPLES_v18.md](DOCKER_EXAMPLES_v18.md) - Docker usage
- [README_v18.md](../README_v18.md) - Main documentation

---

## 📝 Summary Rules

**PixArt is NOT SDXL.**

Treat it as a new-generation DiT model with:
- ✅ Lower CFG (3.5-5.5)
- ✅ Fewer steps (24-36)
- ✅ Square-first resolution (1024×1024)
- ✅ Emotion-driven prompts (verbs + motion)
- ✅ Minimal negatives ("low quality, distorted")
- ✅ Strict attention constraints (slicing required)
- ✅ V100 compatibility safeguards (no FlashAttention/Triton)

**Follow these rules for maximum quality and stability.**
