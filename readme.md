# SDXL DGX Image Lab v17 🚀

Local, offline-friendly SDXL image generation lab optimized for a single-GPU DGX environment.

This app runs inside a Docker container on an NVIDIA DGX system, uses PyTorch + Diffusers + Gradio, and loads SDXL models from a pre-populated Hugging Face cache. It focuses on **stability**, **observability**, and **repeatability** rather than maximum cleverness.

---

## ✅ Current Status (v17)

**Core characteristics:**

- ✅ Single-GPU only (multi-GPU intentionally removed)
- ✅ Local-only models (no runtime internet access required)
- ✅ Uses Hugging Face cache mounted at:  
  `/root/.cache/huggingface`
- ✅ Gradio-based web UI served from inside a Docker container
- ✅ All generations auto-saved + logged (`jobs.log`)

**Supported SDXL models (already downloaded & tested):**

- `stabilityai/stable-diffusion-xl-base-1.0`
- `stabilityai/sdxl-turbo`
- `SG161222/RealVisXL_V5.0`
- `John6666/cyberrealistic-xl-v58-sdxl`
- `cagliostrolab/animagine-xl-4.0`
- `stablediffusionapi/juggernautxl`

**Available schedulers:**

- Default (whatever the pipeline ships with)
- Euler
- DPM++ 2M
- UniPC

**Main UI modes:**

- **Txt2Img / Img2Img**
- **Animate Steps (Light Mode)** — pseudo-motion via prompt variations

---

## ✨ Features in v17

### 1. Artist/Genre Style Profiles

Each profile automatically updates:

- Prompt suffix
- Negative prompt suffix
- Optional scheduler
- Optional default steps

Existing styles (from earlier versions):

- Photoreal
- Cinematic
- Anime / Vibrant
- Soft Illustration
- R-Rated
- Pencil Sketch
- Black & White
- 35mm Film
- Rotoscoping
- None / Raw

**New v17 Artist/Genre Profiles:**
- **Tim Burton Style** — Gothic, dark whimsical, striped patterns
- **Frank Frazetta Fantasy** — Epic fantasy illustration, muscular heroes  
- **Ralph Bakshi Animation** — 1970s rotoscoped animation style
- **H.R. Giger Biomechanical** — Alien, biomechanical, nightmarish beauty
- **Dark Fantasy / Grimdark** — Ominous atmosphere, gothic horror

These are applied as suffixes to the user’s prompt and negative prompt and may override the sampler/steps for that style.

---

### 2. Negative prompt auto-augmentation (visible in UI)

When a style preset is selected:

- A style-specific **negative suffix** is automatically appended.
- The **negative prompt textbox is updated** with the effective negative prompt used for generation.
- The user always sees the real negative prompt sent to the model.

This removes the “hidden negative prompt” problem and makes the behavior transparent and reproducible.

---

### 3. VRAM pre-check (heuristic)

Optional **“Enable VRAM pre-check (heuristic)”** checkbox:

- Uses `torch.cuda.mem_get_info()` to read free GPU memory.
- Estimates VRAM usage based on:
  - Resolution (width × height)
  - Batch size
  - Number of steps
- If the request looks risky, the UI:
  - Shows an orange ⚠ warning
  - Suggests a “safer” approximate resolution

> Note: This is **heuristic**, not a guarantee, but it helps avoid obvious OOM traps on smaller GPUs.

---

### 4. Experimental threaded batch generation

New checkbox: **“Threaded Batch (experimental)”**

- Only affects **Txt2Img**.
- If enabled and `batch_size > 1`:
  - Uses a `ThreadPoolExecutor` to generate each image with its own seed and `torch.Generator`.
  - Still uses **one GPU**, not multi-GPU.
- Semantics:
  - Each image uses a different seed: `base_seed + i`.
  - The goal is to explore concurrency; it may or may not be faster depending on workload.

> This is explicitly experimental. For fully deterministic, “golden” runs, it’s recommended to leave threaded batch **off** and use the normal batched call.

---

### 5. Img2Img lazy loading with status

- Img2Img uses `AutoPipelineForImage2Image` and is **lazy-loaded** only when:
  - The user clicks **“Preload Img2Img (lazy-load)”**, or
  - They run Img2Img for the first time with an init image.
- Status is shown as:
  - `Img2Img pipeline: not loaded yet.`
  - Then updated after loading.

This avoids paying the Img2Img load cost if you only ever run Txt2Img.

---

### 6. Model load persistence across tabs

Problem: opening a new browser tab used to show “no model loaded” even when the backend already had a model in memory.

Solution in v12:

- On app startup, the backend checks if a model is already loaded.
- UI is auto-synchronized:
  - Model dropdown set to the loaded model
  - Status shows:
    > Re-using already loaded model `<model>` with scheduler `<scheduler>`…

Now multiple tabs share the same backend model state correctly.

---

### 7. Animate Steps (Light Mode)

A dedicated tab for generating **short frame sequences** that *look* like motion while still using normal SDXL models (no specialized video model):

- Default frames: 8 (`frame_001` … `frame_008`)
- Each frame:
  - Uses the base prompt + style
  - Applies small prompt variations with motion-oriented phrases
  - Uses a different seed (`base_seed + frame_index`)

Output filenames:

- `YYYYMMDD-HHMMSS_slug_frame_001.png`
- `YYYYMMDD-HHMMSS_slug_frame_002.png`
- …

This is useful for:

- Storyboarding
- Quick animatic-like sequences
- Exploring temporal variations of a scene

Everything is logged in `output_images/jobs.log` just like static generations.

---

### 8. Logging & output layout

All generations:

- Are saved under `output_images/` as `.png`
- Use filenames including timestamp, slugified prompt, seed, and index:
  - `20250101-120000_cityscape_seed123456_01.png`
- Append a line to `output_images/jobs.log` with detailed JSON:
  - Prompt, styled prompt, negatives
  - Model, scheduler, steps, guidance
  - Resolution, batch, seeds
  - Output file paths
  - Mode (`txt2img`, `img2img`, `animate_steps`)

This makes it easy to:

- Reproduce a given image
- Audit what settings produced a given output
- Post-process logs with external tools

---

## 🧩 Architecture

High-level structure:

- Single-file Python app (e.g. `gradio_app_multi-v12-beta.py`)
- Core components:
  - **Pipelines**:
    - `_txt2img_pipe`: `AutoPipelineForText2Image`
    - `_img2img_pipe`: `AutoPipelineForImage2Image`
  - **State**:
    - `_CURRENT_MODEL_KEY`, `_CURRENT_MODEL_ID`, `_CURRENT_SCHEDULER`
  - **Model loading**:
    - Strictly **single-GPU** (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`)
    - Uses shared Hugging Face cache for all models
  - **Schedulers**:
    - Swapped dynamically per model using `from_config`

Concurrency:

- Global lock used only for model loading (`_state_lock`)
- Optional threaded batch for Txt2Img
- No multi-GPU distribution, no complex CUDA device juggling

---

## 📦 Requirements

### Runtime

- **Python**: 3.10+ (as provided by the NVIDIA PyTorch container)
- **GPU**: CUDA-capable NVIDIA GPU (DGX class)
- **VRAM**:
  - 16 GB recommended for SDXL comfort
  - Higher is better for large resolutions / big batches

### Python packages

Installed in the container image (or required if running bare-metal):

- `torch` (PyTorch 2.3.x in the DGX image)
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `gradio`
- `Pillow`
- `numpy`
- `xformers` (optional, used if available for attention memory optimization)

### Models

Models **must be pre-downloaded** into the Hugging Face cache, e.g.:

- `/root/.cache/huggingface/hub/...`

Runtime does **not** reach out to the internet for downloads.  
If a model isn’t present in the cache, loading will fail.

---

## 🚀 Quick Start

### 1. Directory layout

Repo structure (minimal):

```text
.
├── gradio_app_multi-v12-beta.py
├── README.md
└── output_images/            # created automatically if missing
```

### 2. Docker run (DGX)

Example (based on the current working command):

sudo docker rm -f image_gen_ref_v12_beta 2>/dev/null || true

sudo docker run --name image_gen_ref_v12_beta \
  --gpus all \
  --runtime=nvidia \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HF_HUB_OFFLINE=0 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app \
  -w /app \
  gradio_app_generic:dude \
  bash -c "python3 gradio_app_multi-v12-beta.py"


Notes:

--network host means -p 7868:7868 is effectively ignored; you connect directly to:

http://<DGX-HOSTNAME>:7868

CUDA_VISIBLE_DEVICES=2 pins the app to GPU 2 on the DGX.

The Hugging Face cache is mounted from the host into the container.

### 3. Accessing the UI

Open a browser from your workstation:

http://<DGX-HOSTNAME>:7868


You should see “SDXL DGX Image Lab v12 🚀” and the Gradio interface.

## 🧪 Usage Overview
Txt2Img / Img2Img tab

Select model, sampler, style preset.

Adjust steps, guidance scale, and resolution.

Optionally enable:

VRAM pre-check

Threaded batch

Choose mode:

Text to Image:

Ignore init image

Image to Image:

Provide an init image

Adjust strength

Click Generate.

Outputs:

Gallery of images

Job information (HTML) with settings + saved paths

Warning box (OOM risks, lazy-load notes, etc.)

Negative prompt textbox updated with the effective negative

Animate Steps (Light Mode) tab

Enter Base Prompt.

Set resolution, number of frames, seed, and variation strength.

Click Generate Sequence.

Outputs:

Gallery of frames

Info block with:

Seeds

File paths

Settings used

Warnings if VRAM looks tight

## 🖥 DGX Environment & Status

The app is currently deployed in an environment similar to:

Hardware: NVIDIA DGX server

Container base: NVIDIA PyTorch container

Example banner:

NVIDIA Release 24.04

PyTorch Version 2.3.0a0+6ddf5cf

CUDA / drivers:

Managed via the NGC PyTorch image + DGX host drivers

Networking:

Gradio app exposed with --network host on port 7868

GPU selection:

Controlled with CUDA_VISIBLE_DEVICES inside Docker (e.g., 2 for GPU index 2)

Model storage:

Shared Hugging Face cache directory on host: /root/.cache/huggingface

Mounted read/write into the container

Known limitations in this DGX setup

Single-GPU only in this app:

Multi-GPU support was disabled because it:

Caused fragmentation / OOM

Added complexity without clear benefit for this use case

Shared GPU:

If other processes are using the same GPU, VRAM pre-checks and OOM behavior will vary.

SHMEM / ulimits:

Default SHMEM and memlock limits may be low.

For heavy workloads, NVIDIA recommends:

--ipc=host --ulimit memlock=-1 --ulimit stack=67108864


Offline-by-design:

Assumes models are pre-downloaded.

No automatic from_pretrained() downloads via the internet.

No authentication / TLS:

Gradio runs open on the DGX host port.

For production use, you’d typically:

Put this behind a reverse proxy with auth/TLS, or

Bind only to localhost and tunnel.

## 🧠 Insights, Challenges & Lessons Learned
1. Multi-GPU vs Single-GPU

Initial attempt: Multi-GPU pipeline (e.g., model parallelism / data parallelism).

Reality:

Fragmentation

OOM errors

More complexity than payoff for this interactive lab

Decision:

Embrace single-GPU execution for stability and predictability.

Use smarter batching and optional threading, but keep pipelines bound to one device.

Lesson learned:

For an interactive image lab, stability and responsiveness beat theoretical multi-GPU utilization.

2. Model selection & broken checkpoints

Some community SDXL models had:

Missing shards

Incomplete weights

Example: earlier Juggernaut variant issues

Fixed by switching to: stablediffusionapi/juggernautxl

Lesson learned:

Curate a small, known-good set of SDXL models. Fail fast on broken repos and keep the production list lean.

3. Gradio version mismatches

Container had an older Gradio version that:

Did not support gr.Blocks(css="...")

Was sensitive to some newer arguments

Fix:

Inject CSS using gr.HTML("<style>...</style>")

Avoid newer keyword-only args when not strictly needed

Also ran into:

TabError: inconsistent use of tabs and spaces during quick edits

Lesson learned:

Treat the container’s Gradio version as the source of truth. Favor the most compatible, minimal API usage and enforce spaces-only indentation.

4. VRAM heuristics & OOM behavior

Pure trial-and-error OOM handling is:

Frustrating

Time-consuming

Introduced a simple VRAM heuristic:

Uses free VRAM + resolution + batch + steps

Suggests a smaller resolution before it blows up

Lesson learned:

Even a rough heuristic is better than nothing when it comes to warning about OOM. Keep it conservative and treat it as guidance, not gospel.

5. Visibility & observability

Earlier versions had:

Hidden negative prompts

Weak status indication of:

Which model is loaded

Which scheduler is active

What was actually used to generate an image

v12 focuses on:

Making effective negative prompts visible in the UI

Displaying model + scheduler + seed for each job

Logging everything in JSON form

Lesson learned:

For iterative, experimental workflows, transparency and logging are crucial. If it’s not logged, it didn’t happen.

## 🔭 Roadmap / Future Ideas

Some directions for v13+:

Smarter job queue / scheduler:

Queue requests and process them one-at-a-time or in controlled batches

Preset management UI:

Add UI to edit and save custom style presets

More advanced “Animate Steps”:

Latent interpolation

Motion-consistent seeds / noise scheduling

Optional user authentication layer:

Token or basic auth if exposed beyond lab networks

Better VRAM-aware auto-tuning:

Automatically shrink resolution or batch size when near the edge

License & Notes

This app depends on:

NVIDIA PyTorch container license

Hugging Face model licenses for each SDXL checkpoint

Before sharing or commercializing:

Check each model’s license and TOS

Ensure compliance with NVIDIA / HF terms

If you’re reading this from the repo and want to get started fast, see the Quick Start and Usage Overview sections above. Happy generating! ✨



















I am continuing work on my multi-model SDXL Gradio image generation app. Previously, we built versions v1–v10 of the app. I want to move forward and build v11 / v12. Below is everything you need to know to continue seamlessly: 🚀 APP DESCRIPTION (CURRENT STATE – v10) I have a Docker-based Gradio SDXL application that: ✅ Supports multiple SDXL models (local-only) All models live under: /root/.cache/huggingface mounted with: -v /root/.cache/huggingface:/root/.cache/huggingface I pre-download all models via a dedicated script (download_sdxl_models.py). At runtime the app loads models offline-only unless I toggle it. ✅ Works with multiple GPUs The app loads the chosen model onto all visible GPUs, sequentially: cuda:0, cuda:1, cuda:2, cuda:3 Then, during generation: batch requests are split across GPUs each GPU processes its slice images are merged in correct order all generated images are auto-saved with timestamp-based filenames ✅ Features included Model selector (with descriptions) Scheduler selector (Default, Euler, DPM++, UniPC) Style presets: Photoreal Cinematic Anime Soft illustration B&W Pencil Sketch 35mm Film Rotoscoping R Rated Automatically applies style prompt + negative prompt Batch size (should not be auto-clamped unless model is flagged "heavy") Resolution presets (including custom widescreen default) Auto-detected aspect ratio from reference image Img2Img (working) Inpainting (working but currently deprioritized) Local logging: timestamp prompt settings used filenames generated ✅ UI behavior Model loading shows: Yellow “loading” state Green “model loaded in X seconds” Buttons disabled during loading Model is not preloaded on startup When model finishes loading, Generate button should be enabled ⚠️ CURRENT ISSUES TO FIX IN v11 These are important: 1️⃣ Generate button stays disabled until model is reselected Even after “model loaded” is printed. 2️⃣ Batch size sometimes ignores requested value (Seems to auto-limit to 2, or multi-GPU split not applied.) 3️⃣ Negative prompt text box does not update to reflect applied preset Style presets should inject text into the negative box visibly. 4️⃣ Clicking results in the gallery causes Gradio error ValueError: Cannot process this value as an Image, it is of type: dict This is likely because: gallery expects list[Image] but an image dict slipped in Need a strict return: return list_of_PIL_images. 5️⃣ Analytics summary error: TypeError: NDFrame.infer_objects() got an unexpected keyword argument 'copy' Solution: completely disable Gradio analytics internally. We already set: os.environ["GRADIO_ANALYTICS_ENABLED"] = "False" But may need: os.environ["GRADIO_TELEMETRY_ENABLED"] = "0" 🔥 MODELS WE WANT IN v11 Based on latest download testing, working models include: ✓ SDXL Base stabilityai/stable-diffusion-xl-base-1.0 ✓ SDXL Turbo stabilityai/sdxl-turbo ✓ RealVis XL v5.0 SG161222/RealVisXL_V5.0 ✓ CyberRealistic XL 5.8 John6666/cyberrealistic-xl-v58-sdxl ✓ Juggernaut XL (stablediffusionapi) stablediffusionapi/juggernautxl (This version worked — the previous "glides/juggernautxl" failed.) ✓ Animagine XL 4.0 cagliostrolab/animagine-xl-4.0 👉 Remove all models that failed to download (Including Clarity-SDXL and portrait-realistic-sdxl.) 🔥 NEXT FEATURES REQUESTED FOR v11 / v12 A. Multi-GPU Improvements Load model onto all GPUs, but show: "Loading on GPU 0..." "Loading on GPU 1..." etc. Try using torch.compile, channels_last, or xformers if supported. Ensure all GPUs participate immediately during generation: No staggering GPU load distribution proportional to batch size B. UI & Behavior Fix: After model loads, Generate button must become active Style preset must auto-fill the visible negative text box Add short description next to model names & schedulers More default style presets Compact "Loading model" progress output When loading, show ETA and which step is happening Automatically save all batch images, not only index 0 C. Logging Write to: ./logs/app_events.log Per job, include: [timestamp] MODEL=..., STEPS=..., SCHEDULER=..., WIDTHxHEIGHT, BATCH=..., STYLE=... PROMPT: ... NEG: ... SAVED FILES: - image_20250101_213033_xyz.png - ... 🧪 TESTING REQUIREMENTS Before running the full app, I must be able to check: python3 -m py_compile app_v11.py No: TabError SyntaxError Mixed tabs/spaces Non-UTF-8 characters We must ensure the next version is 100% clean. ❓ IMPORTANT QUESTIONS FOR THE NEXT VERSION I need the assistant to decide: Should model be loaded separately per-GPU or loaded once then copied? Should batch splitting be synchronous or parallel tasks? How to fix Gradio gallery dict error? Should we store metadata in JSON sidecar files? Should we implement model “warmup” step? 📌 DELIVERABLE IN NEW CHAT When continuing this project, please: Regenerate entire app code (v11) in ONE SINGLE file Remove dead models Fix all issues above Improve multi-GPU batch splitting logic Keep comments extremely clean and organized Ensure no syntax errors Make generation faster and errors cleaner





docker run --name image_gen_multi_gpu_v1 \
 --gpus all \
 --runtime=nvidia \
 --network host \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
 -e HF_HUB_OFFLINE=0 \
 -v /root/.cache/huggingface:/root/.cache/huggingface \
 -p 7867:7867 \
 -v "$(pwd)":/app \
 -w /app \
 gradio_app_generic:dude \
 bash -c "python3 gradio_app_multi_gpu_v1.py"
 
 
 
 docker run --name image_gen_single_gpu_v11 \
 --gpus all \
 --runtime=nvidia \
 --network host \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
 -e HF_HUB_OFFLINE=0 \
 -v /root/.cache/huggingface:/root/.cache/huggingface \
 -p 7867:7867 \
 -v "$(pwd)":/app \
 -w /app \
 gradio_app_generic:dude \
 bash -c "python3 gradio_app_multi-v10.py"