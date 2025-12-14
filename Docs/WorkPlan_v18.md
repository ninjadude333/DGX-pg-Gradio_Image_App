# SDXL DGX Image Lab – Work Plan v18+
Roadmap: v18 → v21+

---

## Version History

### v17 – Artist/Genre Profiles + Headless Mode ✅ COMPLETED
- ✅ Artist/Genre profiles (Tim Burton, Frank Frazetta, Ralph Bakshi, H.R. Giger)
- ✅ Headless mode via environment variables
- ✅ Mutually exclusive checkboxes for safety
- ✅ Per-instance log files with INSTANCE_ID

### v18 – PixArt-Σ Model & Smart Pipeline Cache ✅ COMPLETED
- ✅ PixArt Sigma XL 1024 model added
- ✅ Model-type aware loading (SDXL vs PixArt pipelines)
- ✅ Smart LRU pipeline cache (default: 2 models)
- ✅ PixArt-specific runtime safeguards for V100
- ✅ Attention slicing for stability

---

## v19 – Multi-GPU Awareness (Planned)

### Goals
- Make multi-GPU usage intelligent without multi-GPU pipelines
- Ensure jobs only run on completely idle GPUs
- Provide foundation for external orchestrator script

### v19.1 – GPU Introspection
- [ ] Implement `gpu_status.py` helper module
- [ ] Use `pynvml` or `nvidia-smi` parsing
- [ ] Functions: `list_gpus()`, `find_idle_gpus()`
- [ ] CLI utility: `gpu_pick.py`

### v19.2 – Non-sharing Policy
- [ ] Document external orchestrator approach
- [ ] Create `run_dgx_job.sh` sample script
- [ ] Wait for idle GPU → launch container → repeat

---

## v20 – Automated Prompt Generator (Planned)

### Goals
- Keep DGX busy overnight without manual intervention
- Auto-generate prompts based on themes/genres
- Chain jobs continuously

### v20.1 – Prompt Generator Module
- [ ] Create `prompt_generator.py`
- [ ] Template system for prompts
- [ ] YAML/JSON configuration for themes
- [ ] Random seed-based generation

### v20.2 – Continuous Runner
- [ ] Implement `auto_night_runner.py`
- [ ] Loop: generate prompt → run job → repeat
- [ ] Time limit or N jobs limit
- [ ] Combine with v19 GPU detection

---

## v21+ – Analytics & Future Tracks (Planned)

### v21.1 – Favorite-based Analytics
- [ ] Implement `analyze_favorites.py`
- [ ] Parse `jobs.log` for favorite images
- [ ] Aggregate statistics by model/profile/settings
- [ ] Generate "best configs" insights

### v21.2 – Optional 3D Track
- [ ] Separate project: 2D → 3D pipelines
- [ ] NeRF-based approaches
- [ ] Printable mesh generation (STL/OBJ)

---

## Summary Roadmap

| Version | Status | Key Features |
|---------|--------|--------------|
| v17 | ✅ Complete | Artist profiles, headless mode, safety UX |
| v18 | ✅ Complete | PixArt-Σ, pipeline cache, V100 compat |
| v19 | 📋 Planned | GPU detection, idle scheduling |
| v20 | 📋 Planned | Prompt generator, overnight runner |
| v21+ | 📋 Future | Analytics, 3D exploration |

---

## Current Models (v18)

| Model | Type | Img2Img |
|-------|------|---------|
| SDXL Base 1.0 | SDXL | ✅ |
| SDXL Turbo | SDXL | ✅ |
| RealVis XL v5.0 | SDXL | ✅ |
| CyberRealistic XL 5.8 | SDXL | ✅ |
| Animagine XL 4.0 | SDXL | ✅ |
| Juggernaut XL | SDXL | ✅ |
| PixArt Sigma XL 1024 | PixArt | ❌ |

---

## Current Style Profiles (29 Total)

**Core (10):** None/Raw, Photoreal, Cinematic, Anime/Vibrant, Soft Illustration, Black & White, Pencil Sketch, 35mm Film, Rotoscoping, R-Rated

**Artist/Genre (5):** Tim Burton Style, Frank Frazetta Fantasy, Ralph Bakshi Animation, H.R. Giger Biomechanical, Dark Fantasy/Grimdark

**Extended (14):** Watercolor, Hyper-Realistic Portrait, ISOTOPIA Sci-Fi Blueprint, Pixar-ish Soft CG, Pixel Art/Isometric Game, Low-Poly 3D/PS1, Product Render/Industrial, Isometric Tech Diagram, Retro Comic/Halftone, Vaporwave/Synthwave, Children's Book Illustration, Ink & Screentone Manga, Analog Horror/VHS, Architectural Visualization
