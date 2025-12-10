PROJECT: SDXL DGX Image Lab
SYSTEM: DGX-based image generation app using Docker + PyTorch + Diffusers + Gradio.

CURRENT MODELS INSTALLED:
- stabilityai/stable-diffusion-xl-base-1.0
- stabilityai/sdxl-turbo
- SG161222/RealVisXL_V5.0
- John6666/cyberrealistic-xl-v58-sdxl
- cagliostrolab/animagine-xl-4.0
- stablediffusionapi/juggernautxl

PLANNED NEW MODELS (v16):
- dreamshaperXL or dreamshaper variants (SDXL tuned)
- ZavyChroma XL
- Protogen/PhantasmXL (if offline-compatible)
- A curated set of specialty SDXL models that add variety

SCHEDULERS:
- Default, Euler, DPM++ 2M, UniPC

CURRENT CORE FEATURES:
- txt2img + img2img
- Lazy loading for pipelines (txt2img and img2img)
- Style profiles (Photoreal, Cinematic, Anime/Vibrant, Soft Illustration, Pencil, Rotoscoping, B&W, etc.)
- Automatic negative prompt injection
- Batch mode fully working, parallel-safe with unique seeds
- All images saved with timestamp, seed, slug
- jobs.log stores metadata for every generated image
- All-models × all-profiles batch sweep (v14)
- Auto resolution downgrade if CUDA OOM (v15)
- Improved verbose logging (v15.1)

PLANNED NEW ARTIST PROFILES (v17):
- Tim Burton
- Frank Frazetta
- Ralph Bakshi
- H.R. Giger

Each includes:
- Prompt suffix
- Negative suffix
- Optional default steps + scheduler

PLANNED UX / LOGIC ENHANCEMENTS:
- Mutually exclusive checkboxes:
  - “Run all profiles”
  - “Run all models + profiles”
  Only one may be active.

- Show live progress:
  - “Processing profile X of Y”
  - “Processing model A of B”

- Dedicated run folder auto-created for multiprofile/multimodel runs.

PLANNED ADVANCED FEATURES:
1. **Headless Mode (v17)**
   - Run full jobs without UI.
   - All parameters (model, prompt, style, steps, batch, etc.)
     can be passed via:
       • environment variables  
       • command-line args  
       • container runtime args
   - Supports full automation:
       RUN_ALL_MODELS=1  
       RUN_ALL_PROFILES=1  

2. **Per-Instance Logging (v17)**
   - Each container writes to:
     jobs_<containername>.log

3. **Dynamic GPU Assignment (v18)**
   - Detect idle GPUs using pynvml or nvidia-smi
   - Only execute job on a GPU with:
       • 0 running processes  
       • near-zero utilization  
       • near-zero VRAM allocation  
   - If no free GPU → wait/queue until one frees up

4. **Automated Prompt Generator (v19)**
   - Tool that generates prompt ideas based on:
       • genre  
       • notes  
       • parameters  
   - After each job finishes, generate a new prompt → start next job
   - Designed for overnight jobs (keep DGX fully utilized)

5. **Continuous Job Scheduler (v19)**
   - Loop:
       generate_prompt → run_job → analyze → repeat
   - Never idle unless user stops.

METADATA REQUIREMENT:
- Every image must store full metadata:
    {
      model,
      profile,
      scheduler,
      steps,
      seed,
      resolution,
      strength (img2img),
      timestamp,
      container_id
    }
- Future script will read favorite images → detect best configurations.

ADDITIONAL NOTES:
- User wants future exploration of 3D-generation models (Shap-E, DreamFusion, NeRF)
  → separate project, not mixed into SDXL Lab.

- Docker optimization:
  - Port chosen dynamically using:
      GRADIO_PORT env variable
  - Offline HF cache mounted from host
  - Multiple containers run simultaneously without conflicts.

DEVELOPMENT ROADMAP SUMMARY:

v17:
  - Add new artist profiles
  - Implement headless mode
  - Add exclusive checkboxes logic
  - Add per-container log file naming

v18:
  - GPU detection + idle GPU assignment
  - Ensure no GPU sharing
  - Job queue if GPUs busy

v19:
  - Automated prompt generator
  - Continuous overnight generation loop

v20+:
  - Extended analytics (favorite styles/models)
  - Optional 3D-model generation exploration
