#!/usr/bin/env python3
"""
SDXL DGX Image Lab v16

- Single GPU (no multi-GPU pipeline).
- Txt2Img + lazy Img2Img with deadlock-free loading.
- Rich style profiles (v12–v15 + new v16 profiles).
- Multi-profile (do_all_profiles) and all-models × all-profiles grid (do_all_models).
- Smart OOM handling with auto resolution downgrade.
- Verbose logging for long-running DGX jobs.

Designed to run in Docker on a DGX with models cached under /root/.cache/huggingface.
"""

import os
import gc
import math
import json
import time
import random
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image
import gradio as gr

from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

HF_CACHE_DIR = os.environ.get("HF_HOME", "/root/.cache/huggingface")
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
JOBS_LOG_PATH = os.path.join(OUTPUT_DIR, "jobs.log")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model registry (human name -> HF repo id)
AVAILABLE_MODELS: Dict[str, str] = {
    # Existing models
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "RealVis XL V5.0": "SG161222/RealVisXL_V5.0",
    "CyberRealistic XL v5.8": "John6666/cyberrealistic-xl-v58-sdxl",
    "Animagine XL 4.0": "cagliostrolab/animagine-xl-4.0",
    "Juggernaut XL": "stablediffusionapi/juggernautxl",
    # v16 additions
    "DreamShaper XL": "Lykon/dreamshaper-xl-1-0",
    "EpicRealism XL": "AiAF/epicrealismXL-vx1Finalkiss_Checkpoint_SDXL",
    "Pixel Art XL": "nerijs/pixel-art-xl",
    "Anime Illust Diffusion XL": "Eugeoter/anime_illust_diffusion_xl",
}

SCHEDULERS = {
    "Default": "default",
    "Euler": "euler",
    "DPM++ 2M": "dpmpp_2m",
    "UniPC": "unipc",
}

# ---------------------------------------------------------------------
# Style Profiles
# ---------------------------------------------------------------------

STYLE_PROFILES: Dict[str, Dict[str, object]] = {
    # Core profiles
    "None (raw)": {
        "prompt_suffix": "",
        "negative_suffix": "",
    },
    "Photoreal": {
        "prompt_suffix": "ultra-detailed, 8k, photorealistic, sharp focus, high dynamic range",
        "negative_suffix": "lowres, bad anatomy, text, error, extra limbs, worst quality, low quality, cartoon, flat lighting",
        "default_scheduler": "DPM++ 2M",
        "default_steps": 26,
    },
    "Cinematic": {
        "prompt_suffix": "cinematic lighting, volumetric light, high contrast, 35mm still, filmic color grading, depth of field",
        "negative_suffix": "flat lighting, washed out, low contrast, lowres, bad anatomy, text",
        "default_scheduler": "Euler",
        "default_steps": 30,
    },
    "Anime / Vibrant": {
        "prompt_suffix": "anime style, vibrant colors, crisp lineart, cel shading, highly detailed background",
        "negative_suffix": "photorealistic, grainy, lowres, messy lines",
        "default_scheduler": "UniPC",
        "default_steps": 24,
    },
    "Soft Illustration": {
        "prompt_suffix": "soft illustration, painterly, gentle shading, storybook style, textured brush strokes",
        "negative_suffix": "photorealistic, harsh shadows, grainy, high contrast",
        "default_scheduler": "Euler",
        "default_steps": 24,
    },
    "R-Rated": {
        "prompt_suffix": "mature themes, gritty atmosphere, harsh lighting, realistic textures",
        "negative_suffix": "childish, cartoonish, simplistic",
    },
    "Pencil Sketch": {
        "prompt_suffix": "pencil sketch, graphite lines, monochrome, cross hatching, rough shading",
        "negative_suffix": "color, digital painting, photorealistic",
    },
    "Black & White": {
        "prompt_suffix": "black and white, high contrast, strong shadows, monochrome film",
        "negative_suffix": "color, oversaturated",
    },
    "35mm Film": {
        "prompt_suffix": "shot on 35mm film, film grain, natural colors, cinematic framing, analogue feel",
        "negative_suffix": "digital noise, oversharpened, CGI, cartoon",
    },
    "Rotoscoping": {
        "prompt_suffix": "rotoscoped animation, outlined figures, limited color palette, stylized motion",
        "negative_suffix": "photorealistic, soft focus, 3d render",
    },
    # v12+ profiles
    "Watercolor": {
        "prompt_suffix": "watercolor painting, soft bleeding colors, paper texture, loose brush strokes",
        "negative_suffix": "hard edges, photorealistic, digital rendering",
        "default_steps": 22,
    },
    "Hyper-Realistic Portrait": {
        "prompt_suffix": "hyper realistic portrait, ultra detailed skin, pores, realistic lighting, shallow depth of field",
        "negative_suffix": "cartoon, illustration, low detail, plastic skin",
        "default_steps": 30,
    },
    "ISOTOPIA Sci-Fi Blueprint": {
        "prompt_suffix": "futuristic sci-fi blueprint, white lines on dark background, technical diagram, orthographic projection",
        "negative_suffix": "painterly, photorealistic, messy background",
        "default_steps": 26,
    },
    "Dark Fantasy / Grimdark": {
        "prompt_suffix": "dark fantasy, grimdark, moody lighting, cinematic, gothic atmosphere, detailed textures, chiaroscuro",
        "negative_suffix": "flat lighting, cartoon, washed out, low detail",
        "default_steps": 34,
    },
    "Pixar-ish Soft CG": {
        "prompt_suffix": "soft 3d CGI, pixar-style character, subsurface scattering, soft lighting, stylized proportions, colorful",
        "negative_suffix": "photorealistic, uncanny valley, harsh shadows, grainy",
        "default_steps": 28,
    },
    # v16 new profiles
    "Pixel Art / Isometric Game": {
        "prompt_suffix": "isometric pixel art, 16-bit game, crisp pixels, limited color palette, game mockup",
        "negative_suffix": "blurry, smooth shading, highres photo, gradients, realistic textures",
        "default_steps": 22,
    },
    "Low-Poly 3D / PS1": {
        "prompt_suffix": "low poly 3d render, ps1 graphics, visible polygons, flat shading, simple textures",
        "negative_suffix": "photorealistic, ray tracing, subsurface scattering, hyper detailed",
        "default_steps": 24,
    },
    "Product Render / Industrial": {
        "prompt_suffix": "studio product render, softbox lighting, reflective surface, clean background, design prototype",
        "negative_suffix": "grainy, cluttered background, text, watermark",
        "default_steps": 24,
    },
    "Isometric Tech Diagram": {
        "prompt_suffix": "clean isometric technical diagram, minimal color, thin lines, labeled components, infographic",
        "negative_suffix": "painterly, textured, noisy background",
        "default_steps": 24,
    },
    "Retro Comic / Halftone": {
        "prompt_suffix": "vintage comic book, halftone dots, bold line art, limited CMYK palette, ben-day dots",
        "negative_suffix": "smooth gradients, photorealistic, 3d render",
        "default_steps": 24,
    },
    "Vaporwave / Synthwave": {
        "prompt_suffix": "vaporwave, synthwave, neon grid, sunset, 80s retro, glowing lights, chrome elements",
        "negative_suffix": "muted colors, dull lighting, realistic color grading",
        "default_steps": 24,
    },
    "Children's Book Illustration": {
        "prompt_suffix": "storybook illustration, soft colors, friendly characters, expressive faces, textured brushes",
        "negative_suffix": "hyper realistic, horror, harsh shadows, high contrast",
        "default_steps": 24,
    },
    "Ink & Screentone Manga": {
        "prompt_suffix": "black and white manga illustration, clean line art, screentones, dynamic composition",
        "negative_suffix": "color, shading gradients, painterly",
        "default_steps": 24,
    },
    "Analog Horror / VHS": {
        "prompt_suffix": "analog horror, vhs scanlines, film grain, desaturated colors, eerie atmosphere, liminal space",
        "negative_suffix": "clean digital, sharp focus, bright colors",
        "default_steps": 26,
    },
    "Architectural Visualization": {
        "prompt_suffix": "architectural visualization, realistic materials, global illumination, ultra wide angle, studio quality",
        "negative_suffix": "cartoon, fantasy, extreme stylization",
        "default_steps": 26,
    },
}

# ---------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------

_state_lock = threading.Lock()
_txt2img_pipe = None
_img2img_pipe = None
_CURRENT_MODEL_KEY: Optional[str] = None
_CURRENT_MODEL_ID: Optional[str] = None
_CURRENT_SCHEDULER: Optional[str] = None

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def slugify(text: str) -> str:
    text = text.lower().replace("'", "")
    out = []
    for ch in text.replace("/", " ").replace("_", " "):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in [" ", "-", "."]:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "none"


def ensure_jobs_log():
    if not os.path.exists(JOBS_LOG_PATH):
        with open(JOBS_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")


def append_job_log(entry: Dict):
    ensure_jobs_log()
    entry["logged_at"] = datetime.utcnow().isoformat() + "Z"
    with open(JOBS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def apply_scheduler(pipe, scheduler_name: str):
    if scheduler_name == "Default" or scheduler_name not in SCHEDULERS:
        return pipe
    try:
        kind = SCHEDULERS[scheduler_name]
        if kind == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif kind == "dpmpp_2m":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif kind == "unipc":
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception as e:
        print(f"[SCHED] Failed to apply scheduler {scheduler_name}: {e}", flush=True)
    return pipe


def gpu_free_memory_mb() -> Optional[float]:
    if DEVICE != "cuda":
        return None
    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        return free_mem / (1024 * 1024)
    except Exception:
        return None


def estimate_job_vram_mb(width: int, height: int, batch_size: int, steps: int, model_factor: float = 1.0) -> float:
    base = (width * height / (1024 * 1024)) * batch_size * 150.0
    step_factor = 0.4 + steps / 50.0
    return base * step_factor * model_factor


def model_complexity_factor(model_key: str) -> float:
    if "Turbo" in model_key:
        return 0.7
    if "Pixel" in model_key:
        return 0.8
    if "CyberRealistic" in model_key or "Juggernaut" in model_key or "EpicRealism" in model_key:
        return 1.3
    if "RealVis" in model_key or "DreamShaper" in model_key:
        return 1.1
    return 1.0


# ---------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------


def _load_txt2img_pipeline(model_key: str, scheduler_name: str) -> None:
    global _txt2img_pipe, _CURRENT_MODEL_KEY, _CURRENT_MODEL_ID, _CURRENT_SCHEDULER

    model_id = AVAILABLE_MODELS[model_key]
    offline = os.environ.get("HF_HUB_OFFLINE", "") == "1"

    with _state_lock:
        if _txt2img_pipe is not None and _CURRENT_MODEL_ID == model_id and _CURRENT_SCHEDULER == scheduler_name:
            return

        print(f"[LOAD] Loading txt2img pipeline for model={model_key} ({model_id})", flush=True)
        t0 = time.time()
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                cache_dir=HF_CACHE_DIR,
                local_files_only=offline,
            )
        except Exception as e:
            if offline:
                print(f"[LOAD][WARN] local_files_only=True failed for {model_id}: {e}", flush=True)
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    cache_dir=HF_CACHE_DIR,
                    local_files_only=False,
                )
            else:
                raise

        pipe.to(DEVICE)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        pipe = apply_scheduler(pipe, scheduler_name)
        _txt2img_pipe = pipe
        _CURRENT_MODEL_KEY = model_key
        _CURRENT_MODEL_ID = model_id
        _CURRENT_SCHEDULER = scheduler_name
        t1 = time.time()
        print(f"[LOAD] txt2img model={model_key} loaded in {t1 - t0:.1f}s on {DEVICE}", flush=True)


def _load_img2img_pipeline(model_key: str, scheduler_name: str) -> Tuple[bool, float, str]:
    """
    Load (or reuse) the Img2Img pipeline for the given model.

    Avoids deadlock by not calling _load_txt2img_pipeline under _state_lock.
    """
    global _img2img_pipe, _CURRENT_MODEL_KEY, _CURRENT_MODEL_ID, _CURRENT_SCHEDULER

    model_id = AVAILABLE_MODELS[model_key]
    offline = os.environ.get("HF_HUB_OFFLINE", "") == "1"

    # Quick reuse check under lock
    with _state_lock:
        if (
            _img2img_pipe is not None
            and _CURRENT_MODEL_ID == model_id
            and _CURRENT_SCHEDULER == scheduler_name
        ):
            msg = (
                f"Img2Img pipeline already loaded for model={model_key} "
                f"({model_id}), scheduler={scheduler_name}"
            )
            print(
                f"[LOAD][IMG2IMG] Reusing already loaded img2img pipeline for "
                f"model={model_key} ({model_id}), scheduler={scheduler_name}",
                flush=True,
            )
            return False, 0.0, msg

    # Ensure txt2img is loaded (not under lock)
    _load_txt2img_pipeline(model_key, scheduler_name)

    # Build Img2Img pipeline
    t0 = time.time()
    print(
        f"[LOAD][IMG2IMG] About to create AutoPipelineForImage2Image for "
        f"model={model_key} ({model_id})",
        flush=True,
    )

    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            cache_dir=HF_CACHE_DIR,
            local_files_only=offline,
        )
        print(f"[LOAD][IMG2IMG] from_pretrained() returned for model={model_key}", flush=True)
    except Exception as e:
        t_fail = time.time()
        msg = (
            f"Failed to load Img2Img pipeline for model {model_key} ({model_id}) "
            f"in {t_fail - t0:.1f}s: {e}"
        )
        print(f"[LOAD][IMG2IMG][ERROR] {msg}", flush=True)
        with _state_lock:
            _img2img_pipe = None
        return False, 0.0, msg

    pipe.to(DEVICE)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe = apply_scheduler(pipe, scheduler_name)

    with _state_lock:
        _img2img_pipe = pipe
        _CURRENT_MODEL_KEY = model_key
        _CURRENT_MODEL_ID = model_id
        _CURRENT_SCHEDULER = scheduler_name

    t1 = time.time()
    msg = f"Img2Img pipeline loaded in {t1 - t0:.1f}s."
    print(
        f"[LOAD][IMG2IMG] img2img model={model_key} loaded in {t1 - t0:.1f}s on {DEVICE}",
        flush=True,
    )
    return True, t1 - t0, msg


def _ensure_img2img(model_key: str, scheduler_name: str) -> Tuple[str, str]:
    loaded, seconds, msg = _load_img2img_pipeline(model_key, scheduler_name)
    status = "Img2Img pipeline already loaded and ready." if not loaded else msg
    return status, ""


# ---------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------


def build_effective_prompts(
    base_prompt: str,
    base_negative: str,
    profile_name: str,
) -> Tuple[str, str]:
    profile = STYLE_PROFILES.get(profile_name, {})
    suffix = str(profile.get("prompt_suffix", "") or "").strip()
    neg_suffix = str(profile.get("negative_suffix", "") or "").strip()

    styled_prompt = base_prompt.strip()
    if suffix:
        styled_prompt = (styled_prompt + ", " + suffix) if styled_prompt else suffix

    effective_negative = base_negative.strip() if base_negative else ""
    if neg_suffix:
        effective_negative = (
            effective_negative + ", " + neg_suffix if effective_negative else neg_suffix
        )

    return styled_prompt, effective_negative


def auto_downgrade_sizes(width: int, height: int) -> List[Tuple[int, int]]:
    sizes = [(width, height)]
    sizes.append((int(width * 0.75), int(height * 0.75)))
    sizes.append((int(width * 0.66), int(height * 0.66)))
    tuned = []
    for w, h in sizes:
        w = int(w // 8 * 8)
        h = int(h // 8 * 8)
        tuned.append((max(w, 256), max(h, 256)))
    return tuned


def generate_single_batch_txt2img(
    model_key: str,
    scheduler_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    seed_base: int,
) -> Tuple[List[Image.Image], List[int], Tuple[int, int], Optional[str]]:
    _load_txt2img_pipeline(model_key, scheduler_name)
    pipe = _txt2img_pipe

    seeds: List[int] = []
    images: List[Image.Image] = []
    attempt_sizes = auto_downgrade_sizes(width, height)
    last_error: Optional[str] = None

    for attempt_idx, (w, h) in enumerate(attempt_sizes, start=1):
        try:
            print(
                f"[GEN][TXT2IMG] Attempt {attempt_idx}/{len(attempt_sizes)} at "
                f"{w}x{h}, batch={batch_size}",
                flush=True,
            )
            seeds = []
            images = []
            for i in range(batch_size):
                if seed_base is None or seed_base < 0:
                    seed = random.randint(0, 2**32 - 1)
                else:
                    seed = seed_base + i
                generator = torch.Generator(device=DEVICE).manual_seed(seed)
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=w,
                    height=h,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                )
                img = out.images[0]
                images.append(img)
                seeds.append(seed)
            return images, seeds, (w, h), None
        except torch.cuda.OutOfMemoryError as e:
            last_error = str(e)
            print(
                f"[GEN][OOM] Attempt {attempt_idx}/{len(attempt_sizes)} at {w}x{h} "
                f"failed. Trying lower resolution...",
                flush=True,
            )
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            last_error = str(e)
            print(f"[GEN][ERROR] Txt2Img generation failed: {e}", flush=True)
            break

    return [], [], attempt_sizes[-1], last_error


def generate_single_batch_img2img(
    model_key: str,
    scheduler_name: str,
    prompt: str,
    negative_prompt: str,
    init_image: Image.Image,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    strength: float,
    batch_size: int,
    seed_base: int,
) -> Tuple[List[Image.Image], List[int], Tuple[int, int], Optional[str]]:
    _ensure_img2img(model_key, scheduler_name)
    pipe = _img2img_pipe

    init_image_resized = init_image.convert("RGB").resize((width, height), Image.LANCZOS)

    seeds: List[int] = []
    images: List[Image.Image] = []
    attempt_sizes = auto_downgrade_sizes(width, height)
    last_error: Optional[str] = None

    for attempt_idx, (w, h) in enumerate(attempt_sizes, start=1):
        try:
            print(
                f"[GEN][IMG2IMG] Attempt {attempt_idx}/{len(attempt_sizes)} at "
                f"{w}x{h}, batch={batch_size}, strength={strength}",
                flush=True,
            )
            seeds = []
            images = []
            init_resized = init_image_resized.resize((w, h), Image.LANCZOS)
            for i in range(batch_size):
                if seed_base is None or seed_base < 0:
                    seed = random.randint(0, 2**32 - 1)
                else:
                    seed = seed_base + i
                generator = torch.Generator(device=DEVICE).manual_seed(seed)
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_resized,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                )
                img = out.images[0]
                images.append(img)
                seeds.append(seed)
            return images, seeds, (w, h), None
        except torch.cuda.OutOfMemoryError as e:
            last_error = str(e)
            print(
                f"[GEN][OOM] Attempt {attempt_idx}/{len(attempt_sizes)} at {w}x{h} "
                f"failed. Trying lower resolution...",
                flush=True,
            )
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            last_error = str(e)
            print(f"[GEN][ERROR] Img2Img generation failed: {e}", flush=True)
            break

    return [], [], attempt_sizes[-1], last_error


# ---------------------------------------------------------------------
# High-level job handler
# ---------------------------------------------------------------------


def generate_images(
    mode: str,
    prompt: str,
    negative_prompt: str,
    model_key: str,
    scheduler_name: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    batch_size: int,
    seed_base: int,
    style_profile: str,
    do_all_profiles: bool,
    do_all_models: bool,
    init_image: Optional[Image.Image],
    img2img_strength: float,
) -> Tuple[List[Image.Image], str, str]:
    """
    Main entry point called by Gradio.
    """
    t_start = time.time()
    ts = timestamp()

    is_img2img = mode == "Image to Image"
    if is_img2img and init_image is None:
        return [], "⚠️ Please upload an init image for Img2Img mode.", ""

    # VRAM check
    free_mb = gpu_free_memory_mb()
    warn_html = ""
    if free_mb is not None:
        est_mb = estimate_job_vram_mb(
            width, height, batch_size, steps, model_complexity_factor(model_key)
        )
        if est_mb > free_mb * 0.8:
            warn_html = (
                f"⚠️ VRAM warning: estimated job usage ~{est_mb:.0f} MB vs "
                f"free ~{free_mb:.0f} MB. Auto-downgrade will try smaller resolutions."
            )
            print(
                f"[VRAM] Warning: est={est_mb:.0f}MB free={free_mb:.0f}MB "
                f"model={model_key} size={width}x{height} batch={batch_size}",
                flush=True,
            )

    # Model + profile sets
    model_keys: List[str] = list(AVAILABLE_MODELS.keys()) if do_all_models else [model_key]
    if do_all_profiles:
        profile_names = list(STYLE_PROFILES.keys())
    else:
        profile_names = [style_profile or "None (raw)"]

    # Run directory
    if do_all_models:
        run_dir = os.path.join(OUTPUT_DIR, f"all_models_profiles_{ts}")
    elif do_all_profiles:
        run_dir = os.path.join(OUTPUT_DIR, f"multi_profiles_{ts}")
    else:
        run_dir = OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    all_images: List[Image.Image] = []
    status_lines: List[str] = []

    total_jobs = len(model_keys) * len(profile_names)
    job_index = 0

    for m_key in model_keys:
        model_id = AVAILABLE_MODELS[m_key]
        for prof in profile_names:
            job_index += 1
            styled_prompt, eff_negative = build_effective_prompts(prompt, negative_prompt, prof)

            profile_info = STYLE_PROFILES.get(prof, {})
            profile_steps = int(profile_info.get("default_steps", steps) or steps)
            eff_steps = profile_steps

            eff_scheduler = scheduler_name
            prof_sched = profile_info.get("default_scheduler")
            if prof_sched and not do_all_models:
                eff_scheduler = prof_sched

            print(
                f"[JOB] mode={mode} model={m_key} scheduler={eff_scheduler} "
                f"style={prof} size={width}x{height} batch={batch_size} "
                f"seed_base={seed_base} ({job_index}/{total_jobs})",
                flush=True,
            )

            status_lines.append(
                f"Running {mode} | Model {job_index}/{total_jobs}: {m_key} | "
                f"Profile: {prof} | Scheduler: {eff_scheduler} | Steps: {eff_steps}"
            )

            if is_img2img and init_image is not None:
                print(
                    f"[IMG2IMG] Single-profile: model={m_key}, style={prof}, "
                    f"size={width}x{height}, batch={batch_size}, strength={img2img_strength}, "
                    f"init_image_shape={init_image.size[0]}x{init_image.size[1]}",
                    flush=True,
                )
                status_msg, _ = _ensure_img2img(m_key, eff_scheduler)
                print(f"[JOB] Img2Img pipeline status: {status_msg}", flush=True)
                imgs, seeds, (eff_w, eff_h), err = generate_single_batch_img2img(
                    m_key,
                    eff_scheduler,
                    styled_prompt,
                    eff_negative,
                    init_image,
                    width,
                    height,
                    eff_steps,
                    guidance_scale,
                    img2img_strength,
                    batch_size,
                    seed_base,
                )
                mode_label = "img2img"
            else:
                imgs, seeds, (eff_w, eff_h), err = generate_single_batch_txt2img(
                    m_key,
                    eff_scheduler,
                    styled_prompt,
                    eff_negative,
                    width,
                    height,
                    eff_steps,
                    guidance_scale,
                    batch_size,
                    seed_base,
                )
                mode_label = "txt2img"

            if not imgs:
                append_job_log(
                    {
                        "timestamp": ts,
                        "mode": mode_label,
                        "multi_profile": do_all_profiles or do_all_models,
                        "profile_style": prof,
                        "prompt": prompt,
                        "styled_prompt": styled_prompt,
                        "negative_prompt": negative_prompt,
                        "effective_negative_prompt": eff_negative,
                        "model_key": m_key,
                        "model_id": model_id,
                        "scheduler": eff_scheduler,
                        "steps": eff_steps,
                        "guidance_scale": guidance_scale,
                        "width": width,
                        "height": height,
                        "batch_size": batch_size,
                        "seed_base": seed_base,
                        "seeds": [],
                        "paths": [],
                        "error": err or "Unknown generation error",
                        "run_dir": run_dir,
                    }
                )
                status_lines.append(
                    f"❌ Failed for model={m_key}, profile={prof}: {err or 'Unknown error'}"
                )
                continue

            paths: List[str] = []
            for idx, (img, seed) in enumerate(zip(imgs, seeds), start=1):
                prof_slug = slugify(prof)
                prompt_slug = slugify(prompt[:80] or "prompt")
                model_safe = slugify(m_key)
                model_dir = run_dir
                if do_all_models:
                    model_dir = os.path.join(run_dir, f"model_{model_safe}")
                    os.makedirs(model_dir, exist_ok=True)
                filename = f"{ts}_{prof_slug}_{prompt_slug}_seed{seed}_{idx:02d}.png"
                fpath = os.path.join(model_dir, filename)
                img.save(fpath)
                paths.append(fpath)
                all_images.append(img)

            append_job_log(
                {
                    "timestamp": ts,
                    "mode": mode_label,
                    "multi_profile": do_all_profiles or do_all_models,
                    "profile_style": prof,
                    "prompt": prompt,
                    "styled_prompt": styled_prompt,
                    "negative_prompt": negative_prompt,
                    "effective_negative_prompt": eff_negative,
                    "model_key": m_key,
                    "model_id": model_id,
                    "scheduler": eff_scheduler,
                    "steps": eff_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "batch_size": batch_size,
                    "seed_base": seed_base,
                    "seeds": seeds,
                    "paths": paths,
                    "error": None,
                    "run_dir": run_dir,
                }
            )

            print(
                f"[SINGLE] Generated {len(imgs)} images for model={m_key}, "
                f"style={prof}, mode={mode_label}.",
                flush=True,
            )

    t_end = time.time()
    total_time = t_end - t_start

    summary = (
        f"Model: {model_key} ({AVAILABLE_MODELS.get(model_key)})\n"
        f"Mode: {mode}\n"
        f"Resolution: {width}x{height}   |   Batch: {batch_size}\n"
        f"All models × profiles run: "
        f"{len(model_keys)} models × {len(profile_names)} profiles × batch {batch_size} "
        f"= {len(all_images)} images.\n"
        f"Run dir: {run_dir}\n"
        f"Total time: {total_time/60:.1f} minutes.\n"
    )

    print("[SUMMARY]\n" + summary, flush=True)

    status_html = "<br>".join(status_lines)
    if warn_html:
        status_html = warn_html + "<br>" + status_html

    return all_images, summary, status_html


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------


def build_ui():
    with gr.Blocks(title="SDXL DGX Image Lab v16") as demo:
        gr.Markdown("# SDXL DGX Image Lab v16 🚀")
        gr.Markdown(
            "Single-GPU SDXL lab for DGX. Txt2Img, Img2Img, multi-profile runs, "
            "and all-models × all-profiles grid sweeps. v16 adds new models and style profiles."
        )

        with gr.Row():
            with gr.Column(scale=2):
                mode = gr.Radio(
                    ["Text to Image", "Image to Image"],
                    value="Text to Image",
                    label="Mode",
                )
                prompt = gr.Textbox(
                    lines=3,
                    label="Prompt",
                    placeholder="Describe what you want to see...",
                )
                negative_prompt = gr.Textbox(
                    lines=2,
                    label="Negative Prompt",
                    placeholder="Things to avoid (e.g. lowres, bad anatomy, text, watermark...)",
                )
                init_image = gr.Image(
                    label="Init Image (for Img2Img)",
                    type="pil",
                )

            with gr.Column(scale=1):
                model = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="SDXL Base 1.0",
                    label="Model",
                )
                scheduler = gr.Dropdown(
                    choices=list(SCHEDULERS.keys()),
                    value="Default",
                    label="Sampler / Scheduler",
                )
                style_profile = gr.Dropdown(
                    choices=list(STYLE_PROFILES.keys()),
                    value="Photoreal",
                    label="Style Profile",
                )
                do_all_profiles = gr.Checkbox(
                    value=False,
                    label="Run ALL profiles for this model",
                )
                do_all_models = gr.Checkbox(
                    value=False,
                    label="Run ALL models × ALL profiles",
                )

                steps = gr.Slider(
                    minimum=4,
                    maximum=80,
                    value=26,
                    step=1,
                    label="Steps",
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.1,
                    label="CFG Scale",
                )

                width = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    value=1024,
                    step=8,
                    label="Width",
                )
                height = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    value=576,
                    step=8,
                    label="Height",
                )

                batch_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=8,
                    step=1,
                    label="Batch Size",
                )
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Base Seed (-1 for random)",
                )
                img2img_strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Img2Img Strength",
                )

                run_btn = gr.Button("Generate 🚀", variant="primary")

        gallery = gr.Gallery(
            label="Output Gallery",
            show_label=True,
            elem_id="gallery",
            columns=4,
            height="auto",
        )

        summary_box = gr.Textbox(
            label="Run Summary",
            lines=6,
        )
        status_html = gr.HTML()

        def _on_generate(
            mode_val,
            prompt_val,
            negative_val,
            model_val,
            scheduler_val,
            steps_val,
            guidance_val,
            width_val,
            height_val,
            batch_val,
            seed_val,
            style_val,
            do_all_profiles_val,
            do_all_models_val,
            init_image_val,
            img2img_strength_val,
        ):
            seed_int = int(seed_val) if seed_val is not None else -1
            return generate_images(
                mode=mode_val,
                prompt=prompt_val,
                negative_prompt=negative_val,
                model_key=model_val,
                scheduler_name=scheduler_val,
                steps=int(steps_val),
                guidance_scale=float(guidance_val),
                width=int(width_val),
                height=int(height_val),
                batch_size=int(batch_val),
                seed_base=seed_int,
                style_profile=style_val,
                do_all_profiles=bool(do_all_profiles_val),
                do_all_models=bool(do_all_models_val),
                init_image=init_image_val,
                img2img_strength=float(img2img_strength_val),
            )

        run_btn.click(
            fn=_on_generate,
            inputs=[
                mode,
                prompt,
                negative_prompt,
                model,
                scheduler,
                steps,
                guidance_scale,
                width,
                height,
                batch_size,
                seed,
                style_profile,
                do_all_profiles,
                do_all_models,
                init_image,
                img2img_strength,
            ],
            outputs=[gallery, summary_box, status_html],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7865")),
        share=False,
    )
