# SDXL DGX Image Lab - Roadmap & Next Steps

**Current Version:** v20  
**Last Updated:** 2024-12-15

---

## 🎯 Immediate Next Steps (v21)

### 1. Parallel Multi-Model Execution ⭐ HIGH PRIORITY
**Goal:** Run multiple models simultaneously on different GPUs

**Benefits:**
- 3-4x faster when running multiple models
- Example: 3 models × 2 min each = 6 min → 2 min total

**Implementation:**
- Pre-load selected models to different GPUs
- Thread pool for parallel generation
- Synchronize results at end

**Complexity:** Medium  
**Impact:** High (major speed improvement)

---

### 2. Smart Batch Size Auto-Adjustment
**Goal:** Automatically reduce batch size on OOM instead of failing

**Current:** User must manually reduce batch size after OOM  
**Proposed:** Catch OOM → retry with batch_size // 2 → log adjustment

**Implementation:**
- Wrap generation in try/except for OOM
- Recursive retry with halved batch
- Log: "Reduced batch from 4 to 2 due to VRAM"

**Complexity:** Low  
**Impact:** Medium (better UX)

---

### 3. Resolution Presets by Model Type
**Goal:** Show recommended resolutions per model

**Current:** All resolutions available for all models  
**Proposed:** 
- SDXL: Show all resolutions
- PixArt/SD3: Highlight 768×432, 768×768 (safe)
- Dim/disable unsafe resolutions

**Implementation:**
- Add `recommended_resolutions` to MODEL_TYPES
- Update UI to show warnings/recommendations

**Complexity:** Low  
**Impact:** Low (UX improvement)

**Note:** This will be superseded by #15 (Dynamic Settings Calculator) which calculates optimal resolutions at runtime

---

### 4. Favorites System
**Goal:** Mark generated images as favorites for later review

**Features:**
- Star/heart button on gallery images
- Save favorites list to JSON
- "Show only favorites" filter
- Export favorites to separate folder

**Implementation:**
- Add metadata to job log with favorite flag
- UI button to toggle favorite
- Filter gallery by favorites

**Complexity:** Medium  
**Impact:** Medium (workflow improvement)

---

## 🔮 Future Enhancements (v22+)

### 5. Prompt Templates & Library
**Goal:** Save and reuse common prompts

**Features:**
- Save prompt + negative + settings as template
- Template library UI
- Quick-load templates
- Share templates as JSON

**Complexity:** Low  
**Impact:** Medium

---

### 6. Image Comparison View
**Goal:** Side-by-side comparison of same prompt across models/profiles

**Features:**
- Grid view of same prompt × different models
- Slider comparison (A/B)
- Highlight differences
- Export comparison sheet

**Complexity:** Medium  
**Impact:** Medium

---

### 7. Automated Prompt Generation
**Goal:** Generate variations of base prompt automatically

**Features:**
- Input: "a cat"
- Output: "a cat sitting", "a cat sleeping", "a cat playing", etc.
- Use LLM or template-based expansion
- Batch generate all variations

**Complexity:** High  
**Impact:** High (creative exploration)

---

### 8. LoRA Support
**Goal:** Add LoRA model support for fine-tuning

**Features:**
- Load LoRA weights on top of base models
- LoRA strength slider
- Multiple LoRAs simultaneously
- LoRA library management

**Complexity:** High  
**Impact:** High (massive creative expansion)

---

### 9. ControlNet Integration
**Goal:** Add ControlNet for guided generation

**Features:**
- Pose control
- Depth control
- Canny edge control
- Scribble control

**Complexity:** Very High  
**Impact:** Very High (professional use case)

---

### 10. Video Generation (AnimateDiff)
**Goal:** Generate short video clips

**Features:**
- AnimateDiff integration
- Frame interpolation
- Motion control
- Export as MP4/GIF

**Complexity:** Very High  
**Impact:** Very High (new capability)

---

### 11. Inpainting/Outpainting
**Goal:** Edit specific regions of images

**Features:**
- Mask editor
- Inpaint selected regions
- Outpaint to extend canvas
- Multiple iterations

**Complexity:** High  
**Impact:** High (editing workflow)

---

### 12. Upscaling Integration
**Goal:** Upscale generated images

**Features:**
- Real-ESRGAN integration
- 2x, 4x upscaling
- Batch upscale
- Face enhancement

**Complexity:** Medium  
**Impact:** Medium (quality improvement)

---

### 13. Negative Prompt Library
**Goal:** Curated negative prompts for common issues

**Features:**
- Pre-built negative prompts
- Category-based (anatomy, artifacts, style)
- One-click apply
- Custom additions

**Complexity:** Low  
**Impact:** Low (convenience)

---

### 14. Scheduler Recommendations
**Goal:** Auto-suggest best scheduler per model/style

**Features:**
- Model-specific scheduler hints
- Style-specific scheduler hints
- Auto-select optimal scheduler
- A/B test schedulers

**Complexity:** Low  
**Impact:** Low (UX improvement)

---

### 15. Dynamic Settings Calculator & VRAM Monitor ⭐ ENHANCED
**Goal:** Calculate optimal settings based on available VRAM and model constraints

**Features:**
- **Runtime VRAM detection** - Query `torch.cuda.mem_get_info()` at startup
- **Per-model optimal settings:**
  - Max batch size based on available VRAM
  - Max safe resolution
  - Recommended steps range
  - Recommended CFG range
- **Live VRAM monitor:**
  - Real-time VRAM usage graph
  - Predict VRAM for current settings
  - Warning before OOM
- **UI hints:**
  - Show "Safe" / "Risky" / "Will OOM" indicators
  - Auto-suggest optimal settings
  - Dim/disable unsafe combinations

**Implementation:**
```python
def calculate_optimal_settings(model_type, available_vram_gb):
    base_vram = {"sdxl": 8, "pixart": 12, "sd3": 16}[model_type]
    max_batch = int(available_vram_gb / base_vram)
    max_pixels = (available_vram_gb / base_vram) * (1024 * 1024)
    max_res = int((max_pixels ** 0.5) // 8) * 8
    return {"max_batch": max_batch, "max_resolution": max_res, ...}
```

**Complexity:** Medium  
**Impact:** High (prevents OOM, optimizes performance)

---

### 16. Batch Job Queue
**Goal:** Queue multiple jobs for sequential execution

**Features:**
- Add jobs to queue
- View queue status
- Pause/resume queue
- Priority ordering

**Complexity:** Medium  
**Impact:** Medium (workflow)

---

### 17. Model Download Manager
**Goal:** Download models from UI

**Features:**
- Browse HuggingFace models
- One-click download
- Progress tracking
- Verify checksums

**Complexity:** Medium  
**Impact:** Low (convenience)

---

### 18. Export to Training Dataset
**Goal:** Export generated images as training data

**Features:**
- Auto-caption images
- Organize by prompt/style
- Export metadata
- DreamBooth-ready format

**Complexity:** Medium  
**Impact:** Medium (training workflow)

---

### 19. API Mode
**Goal:** REST API for programmatic access

**Features:**
- POST /generate endpoint
- GET /status endpoint
- Webhook callbacks
- API key authentication

**Complexity:** Medium  
**Impact:** High (automation)

---

### 20. Multi-User Support
**Goal:** Multiple users with separate workspaces

**Features:**
- User authentication
- Per-user output folders
- Per-user favorites
- Usage quotas

**Complexity:** High  
**Impact:** High (multi-tenant)

---

## 🛠️ Technical Debt & Improvements

### Code Quality
- [ ] Add type hints throughout
- [ ] Add unit tests for core functions
- [ ] Refactor generate_images() (too long)
- [ ] Extract UI building to separate module
- [ ] Add docstrings to all functions

### Performance
- [ ] Profile generation pipeline
- [ ] Optimize image saving (async?)
- [ ] Cache model configs
- [ ] Reduce memory allocations

### Documentation
- [ ] API documentation
- [ ] Architecture diagram
- [ ] Deployment guide
- [ ] Troubleshooting guide

### Infrastructure
- [ ] Docker Compose setup
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline
- [ ] Automated testing

---

## 📊 Priority Matrix

| Feature | Complexity | Impact | Priority |
|---------|-----------|--------|----------|
| Parallel Multi-Model | Medium | High | ⭐⭐⭐⭐⭐ |
| Smart Batch Adjustment | Low | Medium | ⭐⭐⭐⭐ |
| Favorites System | Medium | Medium | ⭐⭐⭐⭐ |
| LoRA Support | High | High | ⭐⭐⭐⭐ |
| Automated Prompts | High | High | ⭐⭐⭐ |
| ControlNet | Very High | Very High | ⭐⭐⭐ |
| Video Generation | Very High | Very High | ⭐⭐⭐ |
| Inpainting | High | High | ⭐⭐⭐ |
| API Mode | Medium | High | ⭐⭐⭐ |
| Upscaling | Medium | Medium | ⭐⭐ |

---

## 🎯 Recommended Development Order

### Phase 1: Performance & UX (v21-v22)
1. Parallel Multi-Model Execution
2. Smart Batch Adjustment
3. Favorites System
4. Resolution Presets

### Phase 2: Creative Tools (v23-v24)
5. Prompt Templates
6. Image Comparison
7. Automated Prompts
8. Negative Prompt Library

### Phase 3: Advanced Features (v25-v26)
9. LoRA Support
10. Upscaling Integration
11. VRAM Monitor
12. Batch Job Queue

### Phase 4: Professional Features (v27+)
13. ControlNet Integration
14. Inpainting/Outpainting
15. Video Generation
16. API Mode

### Phase 5: Enterprise (v30+)
17. Multi-User Support
18. Model Download Manager
19. Export to Training Dataset
20. Full CI/CD

---

## 💡 Community Requests

Track user-requested features here:
- [ ] Custom aspect ratios (user input)
- [ ] Seed browser (explore seed variations)
- [ ] Style mixing (blend multiple profiles)
- [ ] Prompt weighting syntax
- [ ] Regional prompting
- [ ] Batch rename tool
- [ ] Image metadata viewer
- [ ] Dark mode UI

---

## 🚫 Out of Scope

Features explicitly NOT planned:
- ❌ Training models from scratch (use external tools)
- ❌ 3D model generation (different domain)
- ❌ Audio generation (different domain)
- ❌ Mobile app (web-only)
- ❌ Blockchain/NFT integration

---

**Next Review:** After v21 release  
**Feedback:** Open GitHub issues or contact maintainer
