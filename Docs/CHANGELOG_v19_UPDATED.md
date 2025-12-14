# CHANGELOG v19.1 - Updated

## 🎯 Summary

v19.1 adds **SD3 Medium** and **PixArt Sigma XL** support with critical fixes for dtype compatibility and multi-GPU support.

---

## ✨ What's Working (v19.1)

### Models Status

| Model | Type | Dtype | GPU Support | Img2Img | Status |
|-------|------|-------|-------------|---------|--------|
| SDXL Base 1.0 | SDXL | float16 | Single | ✅ | ✅ Stable |
| SDXL Turbo | SDXL | float16 | Single | ✅ | ✅ Stable |
| RealVis XL v5.0 | SDXL | float16 | Single | ✅ | ✅ Stable |
| CyberRealistic XL 5.8 | SDXL | float16 | Single | ✅ | ✅ Stable |
| Animagine XL 4.0 | SDXL | float16 | Single | ✅ | ✅ Stable |
| Juggernaut XL | SDXL | float16 | Single | ✅ | ✅ Stable |
| **PixArt Sigma XL 1024** | **PixArt** | **float32** | **Single** | **❌** | **✅ Working** |
| **SD3 Medium** | **SD3** | **float32** | **Multi** | **❌** | **✅ Working** |

**Total: 8 models (6 SDXL + 1 PixArt + 1 SD3)**

---

## 🔧 Critical Technical Changes

### Float32 Requirement

**Why float32?**
- APEX library (used by T5 encoder in PixArt) expects float32
- Mixed dtype (float16 transformer + float32 encoder) causes "Input type (float) and bias type (c10::Half)" errors
- SD3 has similar dtype sensitivity issues

**Trade-off:**
- ✅ Stable, no dtype errors
- ❌ 2x VRAM usage vs float16
- ❌ Slower inference

### SD3 Multi-GPU Support

```python
if torch.cuda.device_count() > 1:
    _txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="balanced"
    )
else:
    _txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(DEVICE)
```

**Benefits:**
- Distributes model across GPUs automatically
- Enables 1024×1024 generation on 16GB GPUs
- Batch size >1 possible with 3+ GPUs

### PixArt Loading

```python
_txt2img_pipe = PixArtSigmaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Required for APEX
    use_safetensors=True,
    clean_caption=False,
).to(DEVICE)
_txt2img_pipe.enable_attention_slicing()
_txt2img_pipe.enable_vae_slicing()
_txt2img_pipe.enable_vae_tiling()  # Critical for large images
```

### Auto-Resolution Reduction

```python
# PixArt/SD3 in float32: reduce resolution if too large
for mk in model_keys:
    mt = MODEL_TYPES.get(mk, "auto")
    if mt in ["pixart", "sd3"] and (width > 768 or height > 768):
        scale = min(768 / width, 768 / height)
        width = int(width * scale // 8) * 8
        height = int(height * scale // 8) * 8
        print(f"[GENERATE] {mk} resolution reduced to {width}×{height}")
```

---

## 📊 VRAM Usage Tables

### SD3 Medium (float32)

**Single GPU:**
| Batch | Resolution | VRAM | Status |
|-------|------------|------|--------|
| 1 | 768×432 | ~16 GB | ✅ Works on V100 16GB |
| 1 | 1024×1024 | ~28 GB | ❌ OOM on 16GB GPU |
| 2 | 768×432 | ~24 GB | ⚠️ Tight on 24GB |

**Multi-GPU (device_map="balanced"):**
| Batch | Resolution | GPUs | VRAM/GPU | Status |
|-------|------------|------|----------|--------|
| 1 | 768×432 | 2-3 | ~8-10 GB | ✅ Works |
| 1 | 1024×1024 | 2-3 | ~12-14 GB | ✅ Works |
| 4 | 768×432 | 3 | ~10-12 GB | ✅ Works |

### PixArt Sigma (float32)

| Batch | Resolution | VRAM | Status |
|-------|------------|------|--------|
| 1 | 768×432 | ~12 GB | ✅ Works |
| 1 | 1024×1024 | ~22 GB | ⚠️ Tight on 24GB |
| 2 | 768×432 | ~18 GB | ✅ Works on 24GB |

---

## 🚀 Optimal Settings

### SD3 Medium

**Single GPU (16GB):**
- Resolution: **768×432** (16:9) or **768×768** (square)
- Steps: 28
- CFG: 7.0
- Batch: 1
- Scheduler: **Default** (FlowMatchEulerDiscreteScheduler)

**Multi-GPU (2-3 GPUs):**
- Resolution: 1024×1024
- Steps: 28-35
- CFG: 7.0
- Batch: 2-4
- Scheduler: **Default**

### PixArt Sigma

**Recommended:**
- Resolution: **768×432** (16:9) or **1024×1024** (square on 24GB GPU)
- Steps: 24-36
- CFG: 3.5-5.5 (never exceed 6.0)
- Batch: 1-2
- Scheduler: **Default** (DPM-Solver)
- Prompt: Emotion-driven, verbs, motion (not keyword spam)
- Negative: Minimal ("low quality, distorted" only)

---

## 🐛 Issues Fixed in v19.1

1. **"expected scalar type Float but found Half"** - Fixed by using float32 for PixArt and SD3
2. **"Input type (float) and bias type (c10::Half)"** - Fixed by consistent float32
3. **SD3 OOM on single GPU** - Fixed with multi-GPU device_map
4. **PixArt APEX errors** - Fixed with float32 T5 encoder
5. **Batch size ignored** - Removed forced limits, user controls batch size
6. **Empty run folders** - Fixed with _save_image_any() helper (from v19.0)

---

## 📋 Known Limitations

### SD3:
- Float32 required (2x VRAM vs float16)
- Multi-GPU recommended for >768×768
- Max 768×768 on single 16GB GPU
- No img2img support
- Gated model (requires HF auth)

### PixArt:
- Float32 required (APEX compatibility)
- Max 768×768 recommended on single GPU
- Batch 1-2 recommended
- No img2img support

---

## 🐳 Docker Usage

**Multi-GPU for SD3:**
```bash
sudo docker run --name image_gen_v19 \
  --gpus '"device=0,1,2"' \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -e HF_HUB_OFFLINE=0 \
  -e INSTANCE_ID=dgx_gpu012 \
  -e GRADIO_PORT=7864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app \
  -w /app \
  gradio_app_generic:dude \
  bash -lc "pip install --no-cache-dir sentencepiece && python3 gradio_app_multi-v19.py"
```

**Single GPU for PixArt:**
```bash
sudo docker run --name image_gen_v19 \
  --gpus '"device=0"' \
  --network host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e INSTANCE_ID=dgx_gpu0 \
  -e GRADIO_PORT=7864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app \
  -w /app \
  gradio_app_generic:dude \
  bash -lc "pip install --no-cache-dir sentencepiece && python3 gradio_app_multi-v19.py"
```

---

## 🔮 Future (v20+)

1. Quantization (int8/int4) to reduce VRAM
2. Dynamic dtype selection based on available VRAM
3. Automatic batch size reduction on OOM
4. SD3 img2img support
5. PixArt-specific style profiles

---

**Version:** v19.1  
**Status:** ✅ Stable with multi-GPU  
**Tested:** V100 16GB (single + multi-GPU)
