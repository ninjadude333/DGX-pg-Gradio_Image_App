# Changelog
All notable changes to **SDXL DGX Image Lab** will be documented in this file.

This log reflects the versions and features as reconstructed from current context (v11 onward). Earlier versions are summarized briefly.

Format:  
`## [version] – (notes)`  

Dates are omitted or approximate; they can be filled in from git history later.

---

## [v16] – Expanded Models & Profiles (Current Baseline)

**Added**
- New SDXL-based models:
  - **DreamShaper XL** – `Lykon/dreamshaper-xl-1-0`
  - **EpicRealism XL** – `AiAF/epicrealismXL-vx1Finalkiss_Checkpoint_SDXL`
  - **Pixel Art XL** – `nerijs/pixel-art-xl`
  - **Anime Illust Diffusion XL** – `Eugeoter/anime_illust_diffusion_xl`
- New style profiles:
  - **Pixel Art / Isometric Game**  
    - Focus on 16-bit isometric pixel art, crisp pixels, limited palette.
  - **Low-Poly 3D / PS1**  
    - Retro 3D look with visible polygons and flat shading.
  - **Product Render / Industrial**  
    - Clean studio product renders with softbox lighting and simple backgrounds.
  - **Isometric Tech Diagram**  
    - Minimal, line-based isometric diagrams for technical visualization.
  - **Retro Comic / Halftone**  
    - Vintage comic book halftone and Ben-Day dot aesthetics.
  - **Vaporwave / Synthwave**  
    - Neon grids, sunsets, chrome elements, retro color palettes.
  - **Children’s Book Illustration**  
    - Soft, friendly storybook style with expressive characters.
  - **Ink & Screentone Manga**  
    - Black and white manga style with screentones and line art.
  - **Analog Horror / VHS**  
    - VHS scanlines, film grain, desaturated eerie horror vibe.
  - **Architectural Visualization**  
    - Realistic archviz-style renders with proper materials and lighting.

**Changed / Improved**
- Extended the **STYLE_PROFILES** dictionary to unify behavior across Txt2Img and Img2Img for all profiles.
- Ensured compatibility of new models with the existing:
  - Txt2Img / Img2Img pipelines.
  - All-profiles and all-models × profiles batch modes.
- Kept full backward compatibility with v15 behavior, including OOM logic and logging.

**Added Tools**
- `download_models-v16.py` script:
  - Downloads all v16 models into a local HuggingFace cache.
  - Supports specifying `HF_HOME` to avoid needing root permissions.
  - Intended to be run once on an internet-capable machine and reused in offline DGX containers.

---

## [v15.x] – Smart OOM Handling & Img2Img Stability

> Includes v15, v15.1, and v15.2 refinements.

**v15 – Auto Resolution Downgrade for OOM**
- Added **smart auto-downgrade** logic:
  - On `torch.cuda.OutOfMemoryError`, generation is retried at smaller resolutions:
    - Example: 768×768 → ~576×576 → ~512×512.
  - If all attempts fail, job logs an explicit OOM error.
- Improved logging around OOM handling:
  - Messages like:
    - `[GEN][OOM] Attempt 1/3 at 768x768 failed. Trying lower resolution...`
- Preserved batch-size behavior:
  - Batch size remains as requested; only resolution is adjusted.

**v15.1 – Verbose Img2Img Logging**
- Added more detailed console logs for Img2Img:
  - On request:
    - `[IMG2IMG] Request received: model=..., style=..., size=..., batch=..., strength=..., init_image_shape=...`
  - During generation attempts:
    - `[GEN][IMG2IMG] Attempt X/Y at WxH, batch=B, strength=S`
- Clarified whether Img2Img pipelines are newly loaded or reused.

**v15.2 – Deadlock-free Img2Img Pipeline Loading & Offline Robustness**
- Refactored Img2Img pipeline loading to avoid deadlocks and long hangs:
  - Introduced `_load_img2img_pipeline()` with careful lock usage.
  - Avoided calling heavy operations under a global lock.
- Explicitly handled **offline mode**:
  - First try `local_files_only=True` if `HF_HUB_OFFLINE=1`.
  - Fallback to `local_files_only=False` if offline flag is not set or local-only fails.
- Unified logging:
  - Clear messages for:
    - Txt2Img loading: `[LOAD] Loading txt2img pipeline for model=...`
    - Img2Img loading: `[LOAD][IMG2IMG] ...`
    - Reuse: `[LOAD][IMG2IMG] Reusing already loaded img2img pipeline ...`
- Result:
  - Img2Img no longer “hangs forever” when loading pipelines.
  - Clear logs indicate exactly when models are being loaded and how long it takes.

---

## [v14] – All Models × All Profiles Grid Runs

**Added**
- **All-models × all-profiles mode**:
  - A single job can iterate over:
    - Every registered model (SDXL Base, Turbo, RealVis, CyberRealistic, Animagine, Juggernaut, etc.)
    - Every style profile in `STYLE_PROFILES`.
  - Each combination produces a batch of images.
- Output organization:
  - Created a run directory named:
    - `output_images/all_models_profiles_<timestamp>/`
  - Inside, per-model directories:
    - `model_sdxl-base-10/`
    - `model_sdxl-turbo/`
    - `model_realvis-xl-v50/`
    - etc.
- Logging:
  - Each combination logged to `jobs.log` with:
    - `multi_profile` flag.
    - Model and profile identifiers.
    - Seed list and file paths.

**Improved**
- Multi-profile loop logic to work robustly over many combinations and long runs.
- Summary text reporting:
  - Includes total counts like:
    - `6 models × 15 profiles × batch 8 = N images`

---

## [v13] – Multi-profile Batch Loop (Single Model)

**Added**
- **do_all_profiles** feature:
  - For a single selected model, iterate over all style profiles automatically.
- Output:
  - Dedicated multi-profile run folder:
    - `output_images/multi_profiles_<timestamp>/`
- Logging:
  - Each profile run logs:
    - Effective prompt (with suffix).
    - Effective negative prompt (with negative suffix).
    - Seeds, file paths, and errors if any.

**Behavior**
- Enabled “profile sweeps” for exploring style differences with one model and one base prompt.
- Helped identify which profiles work best per model and subject.

---

## [v12] – UI Improvements & Profiles Expansion

**Added**
- UI improvements:
  - (Early versions) Tooltips and/or clearer labels near:
    - Models
    - Samplers/schedulers
  - Plan to show:
    - Model type (realistic / anime / fast / heavy).
    - Scheduler notes (e.g., “good for photoreal, good for animation”).
- Negative prompt visualization:
  - Auto-constructed *effective* negative prompt (base + profile suffix).
  - Intent: keep the user aware of what’s actually being used.
- Highlighted warnings:
  - VRAM warnings, heavy-model hints, or low-VRAM suggestions.

**New profiles (relative to v11):**
- **Watercolor**
- **Hyper-Realistic Portrait**
- **ISOTOPIA Sci-Fi Blueprint**
- **Dark Fantasy / Grimdark**
- **Pixar-ish Soft CG**

**Improved**
- Img2Img lazy-loading paths:
  - Refined so Img2Img only loads when needed and is reused.
- Model load persistence (conceptualized):
  - Idea to detect already loaded model when a second browser tab connects and pre-populate UI state.

---

## [v11] – Single-GPU Stable Baseline & Fixes

This version is the **foundation** for all subsequent work.

**Architecture**
- Switched to **single-GPU mode only**:
  - Removed multi-GPU pipeline logic to avoid:
    - OOM issues.
    - Fragmentation and instability.
- Models confirmed downloaded and working:
  - `stabilityai/stable-diffusion-xl-base-1.0`
  - `stabilityai/sdxl-turbo`
  - `SG161222/RealVisXL_V5.0`
  - `John6666/cyberrealistic-xl-v58-sdxl`
  - `cagliostrolab/animagine-xl-4.0`
  - `stablediffusionapi/juggernautxl` (fixed from prior shard issue).

**Schedulers**
- Default
- Euler
- DPM++ 2M
- UniPC

**Features**
- Txt2Img + lazy-loaded Img2Img:
  - Img2Img pipeline is only loaded when needed.
- Style presets:
  - Photoreal
  - Cinematic
  - Anime / Vibrant
  - Soft Illustration
  - R-Rated
  - Pencil Sketch
  - Black & White
  - 35mm Film
  - Rotoscoping
- Aspect ratio presets:
  - Including 16:9 default and low-res options.
- Automatic negative-prompt injection based on selected style.
- Auto-save:
  - All batch images saved with:
    - Timestamp
    - Slugged prompt and style
    - Seed
- JSON jobs log:
  - `output_images/jobs.log` with per-run metadata.
- Batch size:
  - Fully working, no longer clamped to a fixed small value (like 2).
- Model load status:
  - Shown with timing, with green “loaded” indicators.
- Gallery:
  - Switched to PIL images to avoid Gradio “dict vs Image” issues.

**Fixed**
- Multi-GPU instability:
  - OOM and fragmentation issues resolved by simplifying to single GPU.
- Juggernaut model shards:
  - Replaced earlier broken config with `stablediffusionapi/juggernautxl`.
- Gradio return-type error:
  - Fixed by returning correct data types to `Gallery`.
- Negative prompt update bug:
  - Previously not updating correctly when changing style; fixed.
- “Model loaded but button disabled” bug:
  - Ensured generate button state is consistent with model-loading state.

---

## [v1 – v10] – Early Experiments (Summary Only)

> These versions are not fully reconstructed here, but generally included:
> - Initial Gradio SDXL app setup.
> - Early support for SDXL Base, SDXL Turbo, and a small set of profiles.
> - Iterations on dockerization and DGX GPU access.
> - Experiments with multi-GPU pipelines and subsequent decision to simplify.

---

## Future Versions (Planned)

For detailed plans, see `WORK_PLAN.md`.

Planned highlights:
- **v17** – New artist profiles (Tim Burton, Frank Frazetta, Ralph Bakshi, H.R. Giger), headless mode, mutually exclusive batch options, per-instance logs.
- **v18** – GPU detection and idle-only scheduling support.
- **v19** – Automated prompt generation and continuous overnight job runner.
- **v20+** – Favorites-based analytics and optional 2D→3D exploration.

---
