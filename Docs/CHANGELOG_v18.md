# CHANGELOG v18

## v18 Release Notes

**Release Date:** 2025-01-XX

### 🎯 Summary

v18 adds **PixArt-Σ (PixArt Sigma XL 1024)** as a new model option with model-type aware loading and smart pipeline caching for faster model switching.

---

## ✨ New Features

### 1. PixArt-Σ Model Support

**New Model Added:**
- **PixArt Sigma XL 1024** (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)
  - Text-to-image only (no img2img support in this app)
  - Uses T5 text encoder instead of CLIP
  - Optimized for 1024×1024 resolution
  - Compatible with V100 GPUs via attention slicing

**Model-Type Aware Loading:**
- SDXL models → `AutoPipelineForText2Image`
- PixArt-Σ → `PixArtSigmaPipeline`
- Prevents misloading PixArt as SDXL UNet

### 2. Smart Pipeline Cache (LRU)

**Cache Behavior:**
- Default cache size: 2 pipelines (configurable via `PIPE_CACHE_MAX` env var)
- Switching back to recently used model = instant load from cache
- Oldest pipeline evicted when cache is full
- Automatic cleanup: `del pipeline` + `torch.cuda.empty_cache()` + `gc.collect()`

**Benefits:**
- Faster UX when switching between 2-3 models
- Prevents GPU memory fragmentation
- Keeps PixArt and SDXL from coexisting unintentionally

### 3. PixArt-Specific Runtime Safeguards

**Automatic Environment Setup:**
```python
XFORMERS_FORCE_DISABLE_TRITON=1
XFORMERS_DISABLE_FLASH_ATTN=1
DISABLE_FLASH_ATTN=1
```

**Purpose:**
- Ensures V100 GPU compatibility
- Prevents xformers from attempting flash attention/triton paths
- No effect on SDXL models

**Stability Defaults:**
- `dtype=torch.float16`
- `enable_attention_slicing()` enabled
- Prevents fp16 RMSNorm crashes

---

## 🔧 Technical Changes

### Model Loading Logic

**v17 (Before):**
```python
_txt2img_pipe = AutoPipelineForText2Image.from_pretrained(model_id, ...)
```

**v18 (After):**
```python
if model_type == "pixart":
    _txt2img_pipe = PixArtSigmaPipeline.from_pretrained(model_id, ...)
else:
    _txt2img_pipe = AutoPipelineForText2Image.from_pretrained(model_id, ...)
```

### Cache Implementation

**Cache Structure:**
```python
_pipe_cache: Dict[Tuple[str, str], Any] = {}  # (model_key, scheduler) → pipeline
_pipe_cache_order: List[Tuple[str, str]] = []  # LRU order
```

**Cache Hit:**
- Reuses existing pipeline from cache
- Updates LRU order
- Message: `✅ Model {model_key} loaded from cache with {scheduler} scheduler`

**Cache Miss:**
- Loads model from disk
- Adds to cache
- Evicts oldest if cache full

### Img2Img Compatibility

**PixArt Limitation:**
```python
if MODEL_TYPES.get(_current_model_key) == "pixart":
    print("[LOAD] Img2Img is not supported for PixArt Sigma in this app")
    return False
```

---

## 📊 Supported Models (v18)

| Model | Type | Img2Img | Notes |
|-------|------|---------|-------|
| SDXL Base 1.0 | SDXL | ✅ | Baseline model |
| SDXL Turbo | SDXL | ✅ | Fast inference |
| RealVis XL v5.0 | SDXL | ✅ | Photorealistic |
| CyberRealistic XL 5.8 | SDXL | ✅ | Realistic portraits |
| Animagine XL 4.0 | SDXL | ✅ | Anime style |
| Juggernaut XL | SDXL | ✅ | General purpose |
| **PixArt Sigma XL 1024** | **PixArt** | **❌** | **New in v18** |

---

## 🚫 What Was NOT Changed

✅ **Preserved from v17:**
- All 29 style profiles (10 core + 5 artist + 14 extended)
- Headless mode via environment variables
- Mutually exclusive checkboxes (all profiles / all models)
- Per-instance logging with `INSTANCE_ID`
- Dimension validation (divisible by 8)
- Abort functionality
- Seed handling
- Output directory structure
- Job logging format

✅ **No Breaking Changes:**
- All v17 Docker commands work in v18
- All v17 environment variables supported
- All v17 style profiles unchanged
- All v17 SDXL models work identically

---

## 🐛 Bug Fixes

### Fixed in v18:
- **Seed mutation bug** (from v17 hotfix): `seed_base` no longer mutates across profiles
- **Empty image validation** (from v17 hotfix): Catches empty pipeline results before save
- **Img2img save bug** (from v17 hotfix): Images now save correctly in img2img mode

---

## 🔄 Migration Guide (v17 → v18)

### Docker Command Changes

**No changes required!** All v17 commands work in v18.

**Optional: Enable larger cache**
```bash
-e PIPE_CACHE_MAX=3  # Cache up to 3 models (default: 2)
```

### Using PixArt-Σ

**UI Mode:**
```bash
# Select "PixArt Sigma XL 1024" from model dropdown
# Use Text to Image mode only
# Recommended resolution: 1024×1024
```

**Headless Mode:**
```bash
-e MODEL="PixArt Sigma XL 1024" \
-e WIDTH=1024 \
-e HEIGHT=1024 \
-e STEPS=20
```

### Performance Tips

**Fast Model Switching:**
- Keep `PIPE_CACHE_MAX=2` (default) for 2-model workflows
- Increase to 3-4 if you frequently switch between more models
- Higher cache = more VRAM usage

**PixArt Optimization:**
- Use 1024×1024 or 1024×576 for best results
- Lower steps (15-25) often sufficient
- Attention slicing enabled by default for VRAM efficiency

---

## 📝 Known Limitations

### PixArt-Σ Constraints:
- **No img2img support** in this app (txt2img only)
- **No inpainting support**
- **T5 encoder** may be slower to load than CLIP
- **Prompt style** differs from SDXL (more literal interpretation)

### Cache Limitations:
- Cache is **in-process only** (not persistent across restarts)
- Cache eviction is **LRU** (least recently used)
- Cache size is **per-container** (not shared across instances)

---

## 🔮 Future Considerations (v19+)

Potential improvements for future versions:

1. **PixArt img2img support** via custom pipeline
2. **Persistent cache** to disk for faster restarts
3. **Model preloading** on startup for frequently used models
4. **Cache statistics** in UI (hit rate, evictions)
5. **Multi-model batching** (generate with multiple models in one request)
6. **PixArt-specific profiles** optimized for T5 encoder

---

## 📚 Related Documentation

- [README.md](../README.md) - Main documentation
- [DOCKER_EXAMPLES_v18.md](DOCKER_EXAMPLES_v18.md) - Docker usage examples
- [WorkPlan.md](WorkPlan.md) - Development roadmap
- [CHANGELOG_v17.md](CHANGELOG_v17.md) - Previous version changes

---

## 🙏 Acknowledgments

- **PixArt-α Team** for the PixArt-Σ model
- **Hugging Face Diffusers** for pipeline abstractions
- **NVIDIA** for PyTorch container and DGX infrastructure

---

**Version:** v18  
**Status:** ✅ Stable  
**Recommended For:** Production use with PixArt-Σ support
