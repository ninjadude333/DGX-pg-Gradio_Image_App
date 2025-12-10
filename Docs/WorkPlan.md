# SDXL DGX Image Lab – Work Plan
Roadmap: v17 → v20+

This document describes the planned evolution of the **SDXL DGX Image Lab** application, focusing on stability, observability, and creative power on a DGX system with multiple GPUs and offline HuggingFace caching.

The plan is intentionally incremental so each version can be tested and trusted before moving on.

---

## 0. Current State (v17 Completed)

**v17 Status: ✅ COMPLETED**
- ✅ Artist/Genre profiles added (Tim Burton, Frank Frazetta, Ralph Bakshi, H.R. Giger)
- ✅ Headless mode implemented with environment variable control
- ✅ Mutually exclusive checkboxes for safety
- ✅ Per-instance log files with INSTANCE_ID support
- ✅ Single-GPU optimization with enhanced stability

## Previous State (v16 Baseline)

**Core characteristics:**
- Runs in Docker on a DGX, single-GPU per container (no true multi-GPU pipeline).
- Uses local HuggingFace cache:
  - Mounted at `/root/.cache/huggingface` inside the container.
  - Supports offline mode using `HF_HUB_OFFLINE=1`.
- Models (v16 set):
  - `stabilityai/stable-diffusion-xl-base-1.0` (SDXL Base 1.0)
  - `stabilityai/sdxl-turbo` (SDXL Turbo)
  - `SG161222/RealVisXL_V5.0` (RealVis XL V5.0)
  - `John6666/cyberrealistic-xl-v58-sdxl` (CyberRealistic XL v5.8)
  - `cagliostrolab/animagine-xl-4.0` (Animagine XL 4.0)
  - `stablediffusionapi/juggernautxl` (Juggernaut XL)
  - `Lykon/dreamshaper-xl-1-0` (DreamShaper XL)
  - `AiAF/epicrealismXL-vx1Finalkiss_Checkpoint_SDXL` (EpicRealism XL)
  - `nerijs/pixel-art-xl` (Pixel Art XL)
  - `Eugeoter/anime_illust_diffusion_xl` (Anime Illust Diffusion XL)
- Schedulers:
  - Default (as provided by the pipeline)
  - Euler
  - DPM++ 2M
  - UniPC

**Features:**
- Txt2Img and lazy-loaded Img2Img:
  - Img2Img pipeline loading is:
    - Lazy (only when first needed).
    - Reused (no reload for same model + scheduler).
    - Protected against deadlocks and long “black hole” waits.
- Style profiles (v16 set):
  - Core: Photoreal, Cinematic, Anime / Vibrant, Soft Illustration,
    R-Rated, Pencil Sketch, Black & White, 35mm Film, Rotoscoping.
  - Extended: Watercolor, Hyper-Realistic Portrait, ISOTOPIA Sci-Fi Blueprint,
    Dark Fantasy / Grimdark, Pixar-ish Soft CG.
  - v16 additions:
    - Pixel Art / Isometric Game
    - Low-Poly 3D / PS1
    - Product Render / Industrial
    - Isometric Tech Diagram
    - Retro Comic / Halftone
    - Vaporwave / Synthwave
    - Children’s Book Illustration
    - Ink & Screentone Manga
    - Analog Horror / VHS
    - Architectural Visualization
- Aspect ratio + resolution presets (selectable width/height sliders).
- Batch size works correctly (no hard-coded clamp).
- Automatic negative-prompt injection for style profiles.
- Multi-profile mode:
  - `do_all_profiles`: loop through all profiles for a selected model.
- All-models × all-profiles mode:
  - `do_all_models`: loop through all models and all profiles.
  - Creates a timestamped run directory, and a subdirectory per model.
- Auto OOM handling:
  - On CUDA OOM, the app automatically retries with smaller resolutions:
    - e.g. 768×768 → 576×576 → 512×512.
- Logging & outputs:
  - All images auto-saved with:
    - Timestamp
    - Profile slug
    - Prompt slug
    - Seed
    - Model-specific folders for all-models runs.
  - A global JSONL `output_images/jobs.log` with:
    - timestamp, mode (txt2img/img2img), multi_profile flags
    - profile style, model, model_id, scheduler, steps, CFG
    - width, height, batch size, seed_base, seed list
    - list of file paths
    - run directory
    - error if any.
  - Verbose terminal logging:
    - Model load timings
    - Img2Img pipeline load status
    - OOM attempts with fallback resolutions.
- v16 support utility:
  - `download_models-v16.py` to prefetch all model weights into a local cache.
- Dynamic port:
  - Server port is controlled by `GRADIO_PORT` env var (defaults to 7860).

This is our **stable baseline** for future enhancements.

---

## 1. Version v17 – Artist Profiles, Headless Mode & Safety UX ✅ COMPLETED

### Goals
- Add high-value creative tools (artist/genre-inspired profiles).
- Add **headless mode** for running jobs without UI.
- Make “run all profiles” vs “run all models+profiles” UX safe and unambiguous.
- Lay groundwork for better logging and metadata analysis.

### v17.1 – Artist / Genre Profiles

**New style profiles to add:**

1. **Tim Burton**
   - Prompt suffix ideas:
     - `gothic fairytale, elongated silhouettes, pale faces, dark whimsical atmosphere, quirky characters, high contrast`
   - Negative suffix ideas:
     - `bright saturated colors, realistic proportions, sci-fi tech, modern clean design`
   - Intended usage:
     - Character-focused scenes, gothic cityscapes, surreal environments.

2. **Frank Frazetta**
   - Prompt suffix:
     - `heroic fantasy, muscular warriors, dramatic poses, dynamic lighting, painterly brushwork, moody atmosphere`
   - Negative suffix:
     - `flat composition, modern sci-fi, minimal detail, cartoon style`
   - Intended usage:
     - Barbarian / fantasy scenes, epic monsters, heavy metal cover vibes.

3. **Ralph Bakshi**
   - Prompt suffix:
     - `hand-drawn animation, gritty urban fantasy, exaggerated characters, 70s aesthetic, strong outlines`
   - Negative suffix:
     - `hyperrealistic, sterile digital render, polished CGI`
   - Intended usage:
     - Urban fantasy, surreal animation shots, stylized storytelling.

4. **H.R. Giger**
   - Prompt suffix:
     - `biomechanical surrealism, organic machinery, alien architecture, dark monochrome tones, intricate details`
   - Negative suffix:
     - `bright cheerful colors, simple shapes, low detail, cartoon`
   - Intended usage:
     - Sci-fi horror, alien structures, biomechanical beings.

**Tasks:**
- [x] Add these profiles to `STYLE_PROFILES` with:
  - `prompt_suffix`
  - `negative_suffix`
  - Optional `default_steps` and `default_scheduler`.
- [x] Verify:
  - Profiles appear in the dropdown.
  - They work in Txt2Img and Img2Img.
  - They combine well with existing models (esp. EpicRealism, DreamShaper, CyberRealistic, Juggernaut).

---

### v17.2 – Headless Mode (No UI)

**Objective:**  
Allow a container to be launched so that it **runs a generation job and exits**, without starting the Gradio UI.

**Requirements:**
- Trigger via environment variables **or** CLI args, for example:
  - `HEADLESS=1`
  - `PROMPT="..."`  
  - `NEGATIVE_PROMPT="..."` (optional)
  - `MODEL_KEY="SDXL Base 1.0"` or HF id
  - `STYLE_PROFILE="Photoreal"` (or `None (raw)`)
  - `MODE="txt2img"` or `"img2img"`
  - `WIDTH`, `HEIGHT`
  - `STEPS`, `GUIDANCE`
  - `BATCH_SIZE`
  - `SEED`
  - For Img2Img:
    - `INIT_IMAGE_PATH=/app/input/some_image.png`
    - `IMG2IMG_STRENGTH`
  - New sweeps:
    - `RUN_ALL_PROFILES=true|false`
    - `RUN_ALL_MODELS=true|false` (all-models × all-profiles run)
- Behavior:
  - If `HEADLESS=1` or any “headless input” variable is present:
    - Do **not** launch Gradio UI.
    - Instead:
      - Parse parameters.
      - Call the same internal `generate_images(...)` function.
      - Save outputs to disk + jobs.log as usual.
      - Print a clear summary to stdout (including a list of generated files).
      - Exit.

**Tasks:**
- [x] Add a CLI/env parsing layer in `if __name__ == "__main__":`
  - Detect `HEADLESS` or presence of `PROMPT` env.
  - If headless:
    - Build arguments for `generate_images()`.
  - Else:
    - Launch Gradio UI normally.
- [x] Ensure:
  - All-models × all-profiles sweeps can be triggered with:
    - `RUN_ALL_MODELS=true`
    - Optionally `RUN_ALL_PROFILES=true` but **see exclusivity rules below**.
- [x] Return non-zero exit code on fatal errors (optional but nice for automation).

---

### v17.3 – Mutually Exclusive Checkboxes (Safety)

**Problem:**
- In the UI, both:
  - `Run ALL profiles for this model`
  - `Run ALL models × ALL profiles`
- Can be checked at the same time, which can cause confusing behavior.

**Desired behavior:**
- Only **one** of these can be active at once:
  - If user checks “All models × All profiles”:
    - Automatically uncheck “All profiles”.
  - If user checks “All profiles”:
    - Automatically uncheck “All models × All profiles”.

**Tasks:**
- [x] Add simple UI logic (via Gradio `change` callbacks):
  - When `do_all_models` is set to True:
    - Force `do_all_profiles` = False.
  - When `do_all_profiles` is set to True:
    - Force `do_all_models` = False.
- [x] Reflect these invariants in headless mode:
  - If both `RUN_ALL_MODELS` and `RUN_ALL_PROFILES` are set true, define a priority:
    - e.g. `RUN_ALL_MODELS` wins → run full grid.
    - Or refuse and exit with error (safer).

---

### v17.4 – Per-instance Log Files & Metadata Considerations

You requested **separate logs per container instance**, to avoid different containers writing to the same `jobs.log`.

**Requirements:**
- Each container instance should log to its **own** log file.
- Use an identifier such as:
  - `CONTAINER_NAME` env var (passed at `docker run`).
  - Or `HOSTNAME` inside the container.
  - Or explicit `INSTANCE_ID`.

**Design:**
- Introduce a pattern:
  - `JOBS_LOG_BASENAME = "jobs.log"` by default.
  - If `INSTANCE_ID` env is set:
    - Use `jobs_<instance_id>.log`.
  - Else, optionally fall back to hostname:
    - `jobs_<hostname>.log`.
- You still keep **file-per-image** outputs under `output_images/` exactly as today.

**Tasks:**
- [x] Add an initialization function:
  - Reads `INSTANCE_ID` env or hostname.
  - Constructs `JOBS_LOG_PATH`.
- [x] Ensure all existing logging calls use this dynamic path.

---

## 2. Version v18 – GPU Detection & Scheduling

### Goals
- Make multi-GPU usage more intelligent **without** multi-GPU pipelines.
- Ensure jobs only run on **completely idle** GPUs.
- Provide a foundation for an external “orchestrator” script if needed.

### v18.1 – GPU Introspection

**Requirements:**
- Detect available GPUs and their usage:
  - Tools:
    - `pynvml` (preferred) or
    - Fallback to `nvidia-smi` parsing.
- For each GPU:
  - Total and used memory.
  - Active processes / utilization.
- Define “idle GPU” as:
  - No processes OR
  - Very low utilization and memory usage (configurable threshold).

**Tasks:**
- [ ] Implement a small `gpu_status.py` helper module:
  - Functions:
    - `list_gpus() -> List[GpuInfo]`
    - `find_idle_gpus(threshold_mem_mb, threshold_util_pct) -> List[int]`
- [ ] Provide a CLI/utility script:
  - `/app/gpu_pick.py` that prints an idle GPU id or exits non-zero if none.

### v18.2 – Non-sharing Policy

**Desired behavior:**
- You **do not** want to share GPUs between jobs.
- A job should only start if:
  - There is a GPU with zero (or near-zero) usage.

**Approach options:**

1. **External orchestrator (recommended):**
   - A “runner” script outside the container:
     - Checks idle GPU using `gpu_pick.py`.
     - Sets `CUDA_VISIBLE_DEVICES` for the container.
     - Launches a new container bound to that GPU.
   - The app inside just uses `cuda:0` (since CUDA_VISIBLE_DEVICES maps).

2. **Internal queue (single-process, multi-job):**
   - App itself:
     - Monitors GPUs.
     - Only starts generation when its assigned GPU looks idle.
   - More complex in practice, especially with Gradio UI.

Given you already run **multiple containers**, option (1) is more natural.

**Tasks:**
- [ ] Document recommendation in `WORK_PLAN.md` and/or `README.md`:
  - Provide sample bash script for:
    - Checking for idle GPU.
    - Launching a container pinned to that GPU.
- [ ] Optionally create a `run_dgx_job.sh`:
  - Loops:
    - Wait for idle GPU.
    - Start container with the headless job.
    - Repeat.

---

## 3. Version v19 – Automated Prompt Generator & Continuous Job Runner

### Goals
- Keep the DGX **busy all night** without manual intervention.
- Automatically generate prompts based on themes/genres.
- Chain jobs so each new job starts when the previous finishes.

### v19.1 – Prompt Generator Module

**Requirements:**
- Input:
  - A configuration describing:
    - Genres/themes (e.g. fantasy city, sci-fi corridor, moody portrait).
    - Style hints (e.g. use certain profiles or combinations).
    - Optional constraints (NSFW filters, etc.).
- Output:
  - A structured prompt string.
  - Optionally also decide:
    - Model key (or allowed set).
    - Profile.
    - Seed variation pattern.

**Tasks:**
- [ ] Create a `prompt_generator.py`:
  - Contains rules/functions to:
    - Generate prompts from:
      - A base “topic list”.
      - A random seed.
      - Maybe a simple template system:
        - e.g. `"A {adjective} {subject} in a {environment}, {style_tags}"`.
- [ ] Allow configuration via:
  - YAML or JSON file with:
    - Lists of subjects, environments, moods, etc.

### v19.2 – Continuous Runner

**Requirements:**
- A loop that:
  - Picks or generates a prompt.
  - Launches a headless job (using v17 headless interface).
  - Waits for completion.
  - Repeats until:
    - N jobs complete, or
    - A time limit is reached, or
    - Stopped manually.

**Design:**
- Implement a Python script (e.g. `auto_night_runner.py`):
  - Option A: Runs inside a container:
    - Uses internal functions directly.
  - Option B: Orchestrates **multiple containers** externally:
    - Uses `docker run` with headless parameters.
    - Combines with GPU detection from v18.

**Tasks:**
- [ ] Implement:
  - `run_one_job(config)`:
    - Decides:
      - Model, profile, prompt, seed.
    - Calls headless mode (e.g. via subprocess or function call).
  - `main_loop()`:
    - For `i in range(N)` or until time > limit:
      - `run_one_job()`
      - Sleep if necessary.
- [ ] Add simple logging for:
  - Which prompts were generated.
  - Which jobs correspond to which prompts.

---

## 4. Version v20+ – Analytics, Preferences & (Optional) 3D Exploration

### v20.1 – Favorite-based Analytics

**You requested:**
- Ability to:
  - Cherry-pick favorite images (from large batches).
  - Feed them to a script that:
    - Reads the metadata.
    - Figures out:
      - Which models/profiles/steps/resolutions/seeds are most common among favorites.
      - Which combinations are “your best configurations”.

**Current infrastructure:**
- `jobs.log` already stores:
  - All run metadata and file paths.
- This is enough to build a **metadata index**.

**Plan:**
- [ ] Implement `analyze_favorites.py`:
  - Input:
    - A list of favorite image paths:
      - From:
        - A text file, or
        - A directory with symlinks to favorites, or
        - A CSV.
  - Behavior:
    - For each favorite:
      - Find its metadata in `jobs.log` (matching `paths`).
      - Aggregate statistics by:
        - Model, style profile, resolution, steps, CFG, scheduler.
    - Output:
      - Summary tables:
        - Top models, profiles.
        - Typical resolution & CFG.
      - Possibly JSON report.

**Optional extension:**
- [ ] Add an “insights” mode:
  - Suggests:
    - “Favorite bundle” presets:
      - e.g. “You seem to love: CyberRealistic XL + Dark Fantasy / Grimdark + 768×768 + DPM++ 2M”.

---

### v20.2 – Optional Future Track: 2D → 3D

Not part of the main SDXL lab roadmap, but as a **separate project**:

Ideas:
- Explore image → 3D pipelines like:
  - NeRF-based approaches (Instant-NGP / Instant NeRF).
  - Text/image → 3D tools (e.g. DreamFusion-style methods, Shap-E-like models).
- Goal:
  - Produce printable meshes (STL/OBJ) from generated images or sketches.

This is intentionally separate so it doesn’t destabilize the core SDXL DGX Image Lab.

---

## 5. Summary

**Short version of the roadmap:**

- **v17 – Artist/Genre Profiles + Headless Mode + Safer UI**
  - Add Tim Burton, Frank Frazetta, Ralph Bakshi, H.R. Giger profiles.
  - Implement headless generation via env/CLI.
  - Make “all profiles” vs “all models+profiles” mutually exclusive.
  - Support per-instance log files.

- **v18 – Multi-GPU Awareness (Non-sharing GPUs)**
  - GPU status helper (`pynvml`/`nvidia-smi`).
  - Scripts to schedule jobs only on idle GPUs.

- **v19 – Automated Prompt Generator & Overnight Runner**
  - Prompt generator module for genres/themes.
  - Continuous loop that uses headless mode to keep DGX busy.

- **v20+ – Analytics & Optional 3D Track**
  - Favorites-based analysis & “best configs” insights.
  - Optional separate project for 2D → 3D pipelines.

This file is your high-level design + task list for the coming versions.
