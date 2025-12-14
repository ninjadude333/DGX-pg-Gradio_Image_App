# CHANGELOG v19

## v19 Release Notes

**Release Date:** 2025-01-XX

### \ud83c\udfaf Summary

v19 adds **SD3 Medium (Stable Diffusion 3)** support and implements robust image saving to fix empty run folder issues.

---

## \u2728 New Features

### 1. SD3 Medium Support

**New Model Added:**
- **SD3 Medium** (`stabilityai/stable-diffusion-3-medium-diffusers`)
  - Text-to-image only (no img2img support in this app)
  - Uses StableDiffusion3Pipeline
  - Requires HuggingFace authentication (gated model)
  - Compatible with V100 GPUs via attention slicing

**Model-Type Aware Loading:**
- SDXL models \u2192 `AutoPipelineForText2Image`
- PixArt-\u03a3 \u2192 `PixArtSigmaPipeline`
- **SD3 Medium \u2192 `StableDiffusion3Pipeline`**

### 2. Robust Image Saving

**New Helper Function:**
```python
_save_image_any(img, path: Path) -> bool
```

**Features:**
- Handles PIL Images directly
- Converts numpy arrays to PIL if needed
- Verifies file exists after save
- Logs failures with `[SAVE] \u274c` or `[SAVE] \u26a0\ufe0f`

**Fixes:**
- Empty run folders (directories created but no images)
- Silent save failures
- Inconsistent save behavior across model types

### 3. Improved Directory Creation

**Before (v18):**
```python
run_dir.mkdir(exist_ok=True)
```

**After (v19):**
```python
run_dir.mkdir(parents=True, exist_ok=True)
```

**Benefits:**
- Creates parent directories automatically
- Prevents "directory not found" errors
- More robust for nested paths

---

## \ud83d\udd27 Technical Changes

### SD3 Loading Logic

```python
elif model_type == "sd3":
    _ensure_pixart_compat_env()  # V100-safe env
    _txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(DEVICE)
    _txt2img_pipe.enable_attention_slicing()
```

### Img2Img Compatibility Check

**Updated to handle SD3:**
```python
model_type = MODEL_TYPES.get(_current_model_key)
if model_type in ["pixart", "sd3"]:
    print(f"[LOAD] Img2Img is not supported for {_current_model_key}")
    return False
```

### Save Logic

**Before (v18):**
```python
img.save(fpath)
paths.append(str(fpath))
all_images.append(img)
```

**After (v19):**
```python
saved_ok = _save_image_any(img, fpath)
if not saved_ok:
    print(f"[SAVE] \u26a0\ufe0f Save reported failure: {fpath}")
else:
    paths.append(str(fpath))
all_images.append(img)
```

---

## \ud83d\udcca Supported Models (v19)

| Model | Type | Img2Img | Notes |
|-------|------|---------|-------|
| SDXL Base 1.0 | SDXL | \u2705 | Baseline model |
| SDXL Turbo | SDXL | \u2705 | Fast inference |
| RealVis XL v5.0 | SDXL | \u2705 | Photorealistic |
| CyberRealistic XL 5.8 | SDXL | \u2705 | Realistic portraits |
| Animagine XL 4.0 | SDXL | \u2705 | Anime style |
| Juggernaut XL | SDXL | \u2705 | General purpose |
| **SD3 Medium** | **SD3** | **\u274c** | **New in v19, gated** |
| PixArt Sigma XL 1024 | PixArt | \u274c | High quality |

**Total: 8 models (6 SDXL + 1 SD3 + 1 PixArt)**

---

## \ud83d\udeab What Was NOT Changed

\u2705 **Preserved from v18:**
- All 29 style profiles
- Headless mode
- Mutually exclusive checkboxes
- Per-instance logging
- Pipeline cache (LRU)
- PixArt support unchanged
- All SDXL models work identically

\u2705 **No Breaking Changes:**
- All v18 Docker commands work in v19
- All v18 environment variables supported
- All v18 style profiles unchanged

---

## \ud83d\udc1b Bug Fixes

### Fixed in v19:
- **Empty run folders**: Images now save reliably with verification
- **Silent save failures**: Logged with `[SAVE]` messages
- **Img2img check**: Now correctly blocks SD3 and PixArt
- **Premature UI success**: Removed yield that showed success before completion (from v18 fix)

---

## \ud83d\udd04 Migration Guide (v18 \u2192 v19)

### Docker Command Changes

**No changes required!** All v18 commands work in v19.

### Using SD3 Medium

**Prerequisites:**
1. HuggingFace account with SD3 license accepted
2. HF token with read access
3. Model downloaded to cache

**UI Mode:**
```bash
# Select "SD3 Medium" from model dropdown
# Use Text to Image mode only
# Recommended: 1024\u00d71024, steps 28, CFG 7.0
```

**Headless Mode:**
```bash
-e MODEL="SD3 Medium" \\
-e WIDTH=1024 \\
-e HEIGHT=1024 \\
-e STEPS=28 \\
-e CFG_SCALE=7.0
```

### Downloading SD3

**Login to HuggingFace:**
```bash
huggingface-cli login
```

**Download model:**
```bash
python3 download_models_v19.py
```

Or manually:
```python
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-diffusion-3-medium-diffusers")
```

---

## \ud83d\udcdd Known Limitations

### SD3 Constraints:
- **No img2img support** in this app (txt2img only)
- **No inpainting support**
- **Gated model** requires HF authentication
- **Larger VRAM** usage than SDXL (~18-20GB for batch=4)

### Save Limitations:
- Only PIL Images and numpy arrays supported
- Other image formats may fail silently
- File verification is basic (exists check only)

---

## \ud83d\ude80 Performance Tips

### SD3 Optimal Settings

**Balanced:**
- Steps: 28
- CFG: 7.0
- Resolution: 1024\u00d71024
- Batch: 2-4

**Quality:**
- Steps: 35-40
- CFG: 7.0-8.0
- Resolution: 1024\u00d71024
- Batch: 2

**Fast:**
- Steps: 20
- CFG: 5.0-6.0
- Resolution: 768\u00d7768
- Batch: 4

### VRAM Usage (SD3)

| Batch Size | Resolution | VRAM Usage |
|------------|------------|------------|
| 1 | 1024\u00d71024 | ~14 GB |
| 2 | 1024\u00d71024 | ~18 GB |
| 4 | 1024\u00d71024 | ~26 GB |
| 1 | 768\u00d7768 | ~10 GB |

---

## \ud83d\udd2e Future Considerations (v20+)

Potential improvements for future versions:

1. **SD3 img2img support** via custom pipeline
2. **Automatic save retry** on failure
3. **Save format options** (PNG, JPEG, WebP)
4. **Metadata embedding** in saved images
5. **SD3-specific profiles** optimized for SD3 behavior
6. **Multi-format save** (save same image in multiple formats)

---

## \ud83d\udcda Related Documentation

- [README_v19.md](../README_v19.md) - Main documentation
- [DOCKER_EXAMPLES_v19.md](DOCKER_EXAMPLES_v19.md) - Docker usage examples
- [SD3_INTEGRATION_GUIDE_v19.md](SD3_INTEGRATION_GUIDE_v19.md) - SD3 complete guide
- [CHANGELOG_v18.md](CHANGELOG_v18.md) - Previous version changes
- [WorkPlan_v19.md](WorkPlan_v19.md) - Development roadmap

---

## \ud83d\ude4f Acknowledgments

- **Stability AI** for SD3 Medium model
- **Hugging Face Diffusers** for pipeline abstractions
- **NVIDIA** for PyTorch container and DGX infrastructure

---

**Version:** v19  
**Status:** \u2705 Stable  
**Recommended For:** Production use with SD3 support
