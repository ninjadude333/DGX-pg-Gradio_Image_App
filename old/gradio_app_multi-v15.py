import os
import time
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
import numpy as np
import gradio as gr
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
HF_CACHE_DIR = os.environ.get("HF_HOME", "/root/.cache/huggingface")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output_images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Models & Schedulers
# ------------------------------------------------------------

AVAILABLE_MODELS: Dict[str, str] = {
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "RealVis XL v5.0": "SG161222/RealVisXL_V5.0",
    "CyberRealistic XL v5.8": "John6666/cyberrealistic-xl-v58-sdxl",
    "Animagine XL 4.0": "cagliostrolab/animagine-xl-4.0",
    "Juggernaut XL": "stablediffusionapi/juggernautxl",
}

SCHEDULER_NAMES = ["Default", "Euler", "DPM++ 2M", "UniPC"]


def apply_scheduler(pipe, scheduler_name: str):
    if scheduler_name == "Default" or scheduler_name not in SCHEDULER_NAMES:
        return pipe
    config = pipe.scheduler.config
    if scheduler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(config)
    elif scheduler_name == "DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(config)
    elif scheduler_name == "UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(config)
    return pipe


# ------------------------------------------------------------
# Style presets
# ------------------------------------------------------------

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "None / Raw": {
        "prompt_suffix": "",
        "negative_suffix": "",
        "default_scheduler": None,
        "default_steps": None,
    },
    "Photoreal": {
        "prompt_suffix": "photorealistic, ultra-detailed, 8k, sharp focus, natural lighting",
        "negative_suffix": "blurry, lowres, cartoon, illustration, overexposed, underexposed",
        "default_scheduler": "DPM++ 2M",
        "default_steps": 28,
    },
    "Cinematic": {
        "prompt_suffix": "cinematic lighting, film still, volumetric light, dramatic shadows, 35mm",
        "negative_suffix": "flat lighting, low contrast, noisy, lowres",
        "default_scheduler": "Euler",
        "default_steps": 30,
    },
    "Anime / Vibrant": {
        "prompt_suffix": "anime style, vibrant colors, crisp lineart, highly detailed, masterpiece",
        "negative_suffix": "photorealistic, 3d render, low detail, muted colors",
        "default_scheduler": "UniPC",
        "default_steps": 26,
    },
    "Soft Illustration": {
        "prompt_suffix": "soft illustration, pastel colors, gentle shading, storybook art",
        "negative_suffix": "harsh shadows, photorealistic, noisy",
        "default_scheduler": "Euler",
        "default_steps": 24,
    },
    "R-Rated": {
        "prompt_suffix": "cinematic, dramatic, high contrast",
        "negative_suffix": "child, kid, young, lowres, bad anatomy, ugly, deformed",
        "default_scheduler": "DPM++ 2M",
        "default_steps": 25,
    },
    "Pencil Sketch": {
        "prompt_suffix": "pencil sketch, hand-drawn, line art, cross-hatching, monochrome",
        "negative_suffix": "colorful, 3d render, photorealistic",
        "default_scheduler": "Euler",
        "default_steps": 22,
    },
    "Black & White": {
        "prompt_suffix": "black and white, high contrast, film grain",
        "negative_suffix": "color, oversaturated",
        "default_scheduler": "DPM++ 2M",
        "default_steps": 24,
    },
    "35mm Film": {
        "prompt_suffix": "35mm film, grainy, cinematic, film still, analog style",
        "negative_suffix": "digital noise, CGI, over-sharpened",
        "default_scheduler": "Euler",
        "default_steps": 28,
    },
    "Rotoscoping": {
        "prompt_suffix": "rotoscoped animation, stylized outlines, limited color palette",
        "negative_suffix": "photorealistic, 3d render",
        "default_scheduler": "UniPC",
        "default_steps": 24,
    },
    # New profiles
    "Watercolor": {
        "prompt_suffix": "watercolor painting, soft edges, pigment bloom, paper texture, gentle gradients",
        "negative_suffix": "hard edges, 3d render, photorealistic, harsh contrast",
        "default_scheduler": "Euler",
        "default_steps": 30,
    },
    "Hyper-Realistic Portrait": {
        "prompt_suffix": "hyper-realistic portrait, ultra-detailed skin, pores, 50mm lens, shallow depth of field, studio lighting",
        "negative_suffix": "cartoon, anime, illustration, plastic skin, low detail",
        "default_scheduler": "DPM++ 2M",
        "default_steps": 32,
    },
    "ISOTOPIA Sci-Fi Blueprint": {
        "prompt_suffix": "isometric sci-fi city blueprint, glowing cyan lines on dark background, technical drawing, clean outlines, futuristic HUD",
        "negative_suffix": "photorealistic, messy background, painterly, soft shading",
        "default_scheduler": "UniPC",
        "default_steps": 26,
    },
    "Dark Fantasy / Grimdark": {
        "prompt_suffix": "dark fantasy, grimdark, moody lighting, cinematic, gothic atmosphere, detailed textures, chiaroscuro",
        "negative_suffix": "flat lighting, cartoon, washed out, low detail",
        "default_scheduler": "DPM++ 2M",
        "default_steps": 34,
    },
    "Pixar-ish Soft CG": {
        "prompt_suffix": "soft 3d CGI, pixar-style character, subsurface scattering, soft lighting, stylized proportions, colorful",
        "negative_suffix": "photorealistic, uncanny valley, harsh shadows, grainy",
        "default_scheduler": "Euler",
        "default_steps": 28,
    },
}

STYLE_NAMES = list(STYLE_PRESETS.keys())

# ------------------------------------------------------------
# Aspect ratios
# ------------------------------------------------------------

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "16:9 (1024x576)": (1024, 576),
    "9:16 (576x1024)": (576, 1024),
    "1:1 (1024x1024)": (1024, 1024),
    "3:4 Portrait (832x1104)": (832, 1104),
    "4:3 Landscape (1104x832)": (1104, 832),
    "Cinema 21:9 (1216x528)": (1216, 528),
    "Low-res square (768x768)": (768, 768),
    "Custom": (0, 0),
}

# ------------------------------------------------------------
# Backend state
# ------------------------------------------------------------

_state_lock = threading.Lock()
_txt2img_pipe = None
_img2img_pipe = None
_CURRENT_MODEL_KEY: Optional[str] = None
_CURRENT_MODEL_ID: Optional[str] = None
_CURRENT_SCHEDULER: str = "Default"


def _model_heavy_hint(model_key: str) -> Optional[str]:
    heavy_models = {
        "RealVis XL v5.0",
        "CyberRealistic XL v5.8",
        "Juggernaut XL",
    }
    if model_key in heavy_models:
        return (
            f"Model '{model_key}' is relatively heavy. Prefer resolutions ≤ 1024x1024 and smaller batch "
            f"sizes on GPUs with < 16GB VRAM."
        )
    return None


def _format_warning_html(messages: List[str]) -> str:
    if not messages:
        return ""
    items = "".join(f"<li>{m}</li>" for m in messages)
    return f"""
<div style='color:orange; font-weight:bold;'>
  ⚠ Warnings:
  <ul style="font-weight:normal; margin-top:4px;">{items}</ul>
</div>
""".strip()


def _estimate_vram_and_suggest(
    width: int,
    height: int,
    batch_size: int,
    steps: int,
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    if not torch.cuda.is_available():
        return None, None
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return None, None

    free_gb = free_bytes / (1024 ** 3)
    megapixels = (width * height) / 1_000_000.0
    est_gb = megapixels * 1.6 * batch_size
    est_gb *= (0.8 + (steps / 40.0) * 0.4)

    if est_gb > free_gb * 0.8:
        scale = math.sqrt((free_gb * 0.6) / max(est_gb, 1e-6))
        suggested_w = max(512, int(width * scale) // 64 * 64)
        suggested_h = max(512, int(height * scale) // 64 * 64)
        msg = (
            f"Requested resolution {width}x{height} (batch {batch_size}) may exceed available VRAM. "
            f"Estimated need ≈ {est_gb:.1f} GB vs free ≈ {free_gb:.1f} GB. "
            f"Consider using around {suggested_w}x{suggested_h} or smaller."
        )
        return msg, (suggested_w, suggested_h)
    return None, None


def _load_txt2img_pipeline(model_key: str, scheduler_name: str) -> Tuple[bool, float, str]:
    global _txt2img_pipe, _CURRENT_MODEL_KEY, _CURRENT_MODEL_ID, _CURRENT_SCHEDULER

    model_id = AVAILABLE_MODELS[model_key]
    with _state_lock:
        if (
            _txt2img_pipe is not None
            and _CURRENT_MODEL_ID == model_id
            and _CURRENT_SCHEDULER == scheduler_name
        ):
            return False, 0.0, (
                f"Re-using already loaded model: <b>{model_key}</b> with scheduler <b>{scheduler_name}</b>."
            )

        t0 = time.time()
        print(f"[LOAD] Loading txt2img pipeline for model={model_key} ({model_id})", flush=True)
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            cache_dir=HF_CACHE_DIR,
        )
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
        print(
            f"[LOAD] txt2img model={model_key} loaded in {t1 - t0:.1f}s on {DEVICE}",
            flush=True,
        )
        return True, t1 - t0, (
            f"Loaded model <b>{model_key}</b> with scheduler <b>{scheduler_name}</b> "
            f"in {t1 - t0:.1f}s on {DEVICE}."
        )


def _ensure_txt2img(model_key: str, scheduler_name: str) -> Tuple[str, str]:
    loaded, _dt, msg = _load_txt2img_pipeline(model_key, scheduler_name)
    warnings: List[str] = []
    if loaded:
        warnings.append("First load of a model can be slow; subsequent generations will be faster.")
    hint = _model_heavy_hint(model_key)
    if hint:
        warnings.append(hint)
    warn_html = _format_warning_html(warnings)
    return msg, warn_html


def _load_img2img_pipeline(model_key: str, scheduler_name: str) -> Tuple[bool, float, str]:
    global _img2img_pipe, _CURRENT_MODEL_KEY, _CURRENT_MODEL_ID, _CURRENT_SCHEDULER

    model_id = AVAILABLE_MODELS[model_key]
    with _state_lock:
        if (
            _img2img_pipe is not None
            and _CURRENT_MODEL_ID == model_id
            and _CURRENT_SCHEDULER == scheduler_name
        ):
            return False, 0.0, "Img2Img pipeline already loaded and ready."
        _load_txt2img_pipeline(model_key, scheduler_name)
        t0 = time.time()
        print(f"[LOAD] Loading img2img pipeline for model={model_key} ({model_id})", flush=True)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            cache_dir=HF_CACHE_DIR,
        )
        pipe.to(DEVICE)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        pipe = apply_scheduler(pipe, scheduler_name)
        _img2img_pipe = pipe
        t1 = time.time()
        print(
            f"[LOAD] img2img model={model_key} loaded in {t1 - t0:.1f}s on {DEVICE}",
            flush=True,
        )
        return True, t1 - t0, f"Img2Img pipeline loaded in {t1 - t0:.1f}s."


def _ensure_img2img(model_key: str, scheduler_name: str) -> Tuple[str, str]:
    loaded, _dt, msg = _load_img2img_pipeline(model_key, scheduler_name)
    warnings: List[str] = []
    if loaded:
        warnings.append("Img2Img pipeline loaded lazily. Subsequent Img2Img runs will be much faster.")
    hint = _model_heavy_hint(model_key)
    if hint:
        warnings.append(hint)
    warn_html = _format_warning_html(warnings)
    return msg, warn_html


def _sanitize_slug(text: str, max_len: int = 64) -> str:
    if not text:
        return "prompt"
    text = text.lower()
    out_chars: List[str] = []
    for ch in text:
        if ch.isalnum():
            out_chars.append(ch)
        elif ch in (" ", "-", "_", "."):
            out_chars.append("-")
    slug = "".join(out_chars).strip("-")
    slug = "-".join(filter(None, slug.split("-")))
    return slug[:max_len] or "prompt"


def _append_style_suffix(base: str, suffix: str) -> str:
    base = (base or "").strip()
    suffix = (suffix or "").strip()
    if not suffix:
        return base
    if not base:
        return suffix
    if suffix.lower() in base.lower():
        return base
    return base + ", " + suffix


def _log_job(entry: Dict[str, Any]):
    try:
        log_path = os.path.join(OUTPUT_DIR, "jobs.log")
        entry_with_ts = dict(entry)
        entry_with_ts.setdefault("logged_at", datetime.utcnow().isoformat() + "Z")
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(entry_with_ts, f)
            f.write("\n")
    except Exception:
        pass


# ------------------------------------------------------------
# UI helper callbacks
# ------------------------------------------------------------

def on_style_change(style_name: str, current_neg: str, current_sampler: str, current_steps: int):
    cfg = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None / Raw"])
    new_neg = _append_style_suffix(current_neg or "", cfg.get("negative_suffix", ""))
    sampler = current_sampler
    steps = current_steps
    if cfg.get("default_scheduler"):
        sampler = cfg["default_scheduler"]
    if cfg.get("default_steps"):
        steps = int(cfg["default_steps"])
    return new_neg, sampler, steps


def on_aspect_ratio_change(ratio_label: str):
    w, h = ASPECT_RATIOS.get(ratio_label, (0, 0))
    if w <= 0 or h <= 0:
        return gr.update(), gr.update()
    w = int(w // 64 * 64)
    h = int(h // 64 * 64)
    return w, h


def ui_load_model(model_key: str, scheduler_name: str):
    status, warn_html = _ensure_txt2img(model_key, scheduler_name)
    return status, warn_html


def ui_preload_img2img(model_key: str, scheduler_name: str):
    status, warn_html = _ensure_img2img(model_key, scheduler_name)
    return status, warn_html


def init_on_app_load():
    if _CURRENT_MODEL_KEY is not None and _txt2img_pipe is not None:
        msg = (
            f"Re-using already loaded model <b>{_CURRENT_MODEL_KEY}</b> "
            f"with scheduler <b>{_CURRENT_SCHEDULER}</b> in this backend session."
        )
        warn_html = _format_warning_html(
            [
                "Backend model is already loaded. You can start generating immediately.",
                "If UI showed 'no model loaded' before, this has now been synchronized.",
            ]
        )
        return _CURRENT_MODEL_KEY, msg, warn_html
    else:
        return (
            list(AVAILABLE_MODELS.keys())[0],
            "No model loaded yet. Click 'Load / Switch Model' or just generate to auto-load.",
            "",
        )


# ------------------------------------------------------------
# OOM-safe generation helper
# ------------------------------------------------------------

def _safe_generate_images_with_retries(
    pipe,
    mode_used: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    batch_size: int,
    base_seed: int,
    init_image: Optional[Image.Image],
    strength: float,
    max_retries: int = 3,
) -> Tuple[List[Image.Image], int, int, Optional[str]]:
    """
    Run generation with automatic OOM handling by downscaling resolution.

    Returns:
        images, used_width, used_height, downgrade_message
    Raises:
        RuntimeError if all retries fail.
    """
    requested_w = width
    requested_h = height
    current_w = width
    current_h = height
    downgrade_msg: Optional[str] = None

    for attempt in range(max_retries):
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(base_seed)
            if mode_used == "img2img":
                if init_image is None:
                    raise RuntimeError("Img2Img mode requires an init_image.")
                out = pipe(
                    prompt=prompt,
                    image=init_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    strength=float(strength),
                    width=current_w,
                    height=current_h,
                    generator=generator,
                    num_images_per_prompt=batch_size,
                )
            else:
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    width=current_w,
                    height=current_h,
                    generator=generator,
                    num_images_per_prompt=batch_size,
                )
            images = list(out.images)
            if (current_w != requested_w) or (current_h != requested_h):
                downgrade_msg = (
                    f"Resolution auto-downgraded due to OOM from {requested_w}x{requested_h} "
                    f"to {current_w}x{current_h}."
                )
            return images, current_w, current_h, downgrade_msg
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" not in msg:
                raise
            # OOM: try lower resolution
            print(
                f"[GEN][OOM] Attempt {attempt+1}/{max_retries} at {current_w}x{current_h} failed. "
                "Trying lower resolution...",
                flush=True,
            )
            if attempt == max_retries - 1:
                raise
            new_w = max(512, int(current_w * 0.8) // 64 * 64)
            new_h = max(512, int(current_h * 0.8) // 64 * 64)
            if new_w == current_w and new_h == current_h:
                raise
            current_w = new_w
            current_h = new_h

    raise RuntimeError("Unexpected: _safe_generate_images_with_retries exhausted loop.")


# ------------------------------------------------------------
# Core helpers for multi-profile / multi-model
# ------------------------------------------------------------

def _generate_single_profile(
    pipe,
    mode_used: str,
    model_key: str,
    scheduler_name: str,
    steps: int,
    guidance_scale: float,
    prompt: str,
    negative_prompt_in: str,
    style_name: str,
    width: int,
    height: int,
    batch_size: int,
    base_seed: int,
    init_image: Optional[np.ndarray],
    strength: float,
    run_dir: str,
    run_timestamp: str,
) -> Tuple[List[Image.Image], List[int], List[str], str, Optional[str]]:
    """
    Helper: generate images for a single style profile.
    Includes OOM-safe downscaling via _safe_generate_images_with_retries.
    """
    style_cfg = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None / Raw"])
    styled_prompt = _append_style_suffix(prompt or "", style_cfg.get("prompt_suffix", ""))
    effective_negative = _append_style_suffix(
        negative_prompt_in or "",
        style_cfg.get("negative_suffix", ""),
    )

    profile_scheduler = style_cfg.get("default_scheduler") or scheduler_name
    profile_steps = int(style_cfg.get("default_steps") or steps)

    apply_scheduler(pipe, profile_scheduler)

    seeds = [base_seed + i for i in range(batch_size)]
    saved_paths: List[str] = []
    downgrade_msg: Optional[str] = None

    if mode_used == "img2img" and init_image is not None:
        init_pil = Image.fromarray(init_image.astype("uint8")) if isinstance(
            init_image, np.ndarray
        ) else init_image

        images, used_w, used_h, downgrade_msg = _safe_generate_images_with_retries(
            pipe=pipe,
            mode_used="img2img",
            prompt=styled_prompt,
            negative_prompt=effective_negative,
            steps=profile_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            batch_size=batch_size,
            base_seed=base_seed,
            init_image=init_pil,
            strength=strength,
        )
    else:
        images, used_w, used_h, downgrade_msg = _safe_generate_images_with_retries(
            pipe=pipe,
            mode_used="txt2img",
            prompt=styled_prompt,
            negative_prompt=effective_negative,
            steps=profile_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            batch_size=batch_size,
            base_seed=base_seed,
            init_image=None,
            strength=strength,
        )

    slug_prompt = _sanitize_slug(prompt if prompt else style_name)
    slug_profile = _sanitize_slug(style_name)

    for idx, img in enumerate(images):
        seed_val = seeds[idx] if idx < len(seeds) else base_seed
        filename = (
            f"{run_timestamp}_{slug_profile}_{slug_prompt}_seed{seed_val}_{idx+1:02d}.png"
        )
        out_path = os.path.join(run_dir, filename)
        try:
            img.save(out_path)
            saved_paths.append(out_path)
        except Exception:
            saved_paths.append(out_path + " (save failed)")

    job_entry = {
        "timestamp": run_timestamp,
        "mode": mode_used,
        "multi_profile": True,
        "profile_style": style_name,
        "prompt": prompt,
        "styled_prompt": styled_prompt,
        "negative_prompt": negative_prompt_in,
        "effective_negative_prompt": effective_negative,
        "model_key": model_key,
        "model_id": AVAILABLE_MODELS.get(model_key),
        "scheduler": profile_scheduler,
        "steps": profile_steps,
        "guidance_scale": guidance_scale,
        "width": used_w,
        "height": used_h,
        "batch_size": batch_size,
        "seed_base": base_seed,
        "seeds": seeds,
        "paths": saved_paths,
        "error": None,
        "run_dir": run_dir,
    }
    _log_job(job_entry)

    return images, seeds, saved_paths, effective_negative, downgrade_msg


# ------------------------------------------------------------
# Main generation (single / all profiles / all models×profiles)
# ------------------------------------------------------------

def generate_images(
    model_key: str,
    scheduler_name: str,
    steps: int,
    guidance_scale: float,
    style_name: str,
    prompt: str,
    negative_prompt_in: str,
    aspect_ratio_label: str,
    width: int,
    height: int,
    batch_size: int,
    seed: int,
    mode: str,
    init_image: Optional[np.ndarray],
    strength: float,
    enable_vram_check: bool,
    use_parallel_batch: bool,
    do_all_profiles: bool,
    do_all_models_profiles: bool,
    progress=gr.Progress(track_tqdm=True),
):
    global _txt2img_pipe, _img2img_pipe

    model_status, warn_html1 = _ensure_txt2img(model_key, scheduler_name)

    if aspect_ratio_label in ASPECT_RATIOS and aspect_ratio_label != "Custom":
        width, height = ASPECT_RATIOS[aspect_ratio_label]
    width = int(max(512, width // 64 * 64))
    height = int(max(512, height // 64 * 64))

    warnings: List[str] = []
    if warn_html1:
        pass

    vram_suggestion = None
    if enable_vram_check:
        msg, suggestion = _estimate_vram_and_suggest(width, height, batch_size, steps)
        if msg:
            warnings.append(msg)
        if suggestion:
            vram_suggestion = suggestion

    if vram_suggestion:
        sw, sh = vram_suggestion
        warnings.append(f"Suggested safe resolution (heuristic): around {sw}x{sh} or lower.")

    if mode == "Image to Image":
        if init_image is None:
            warnings.append("Img2Img mode selected but no input image provided. Falling back to Txt2Img.")
        else:
            img_status, warn_html_img = _ensure_img2img(model_key, scheduler_name)
            if warn_html_img:
                warnings.append("Img2Img lazy load: completed.")
            warnings.append("Using Img2Img pipeline.")
            print(f"[JOB] Img2Img pipeline status: {img_status}", flush=True)

    if seed is None or seed < 0:
        base_seed = random.randint(0, 2 ** 32 - 1)
    else:
        base_seed = int(seed)

    # Job header log
    mode_label = (
        "all_models_profiles" if do_all_models_profiles
        else ("all_profiles" if do_all_profiles else mode)
    )
    print(
        f"[JOB] mode={mode_label} model={model_key} "
        f"scheduler={scheduler_name} style={style_name} "
        f"size={width}x{height} batch={batch_size} seed_base={base_seed}",
        flush=True,
    )

    all_images: List[Image.Image] = []
    all_saved_paths: List[str] = []
    all_seeds: List[int] = []
    effective_negative_final = negative_prompt_in or ""
    mode_used = "txt2img"
    error_msg = None
    profile_progress_text = ""
    per_job_failures: List[str] = []

    try:
        # Select initial pipeline for the selected model (used in non-all-models branch)
        if mode == "Image to Image" and init_image is not None and _img2img_pipe is not None:
            mode_used = "img2img"
            pipe = _img2img_pipe
        else:
            mode_used = "txt2img"
            pipe = _txt2img_pipe

        # ---------------------------
        # v15: ALL models × ALL profiles (with OOM-safe handling)
        # ---------------------------
        if do_all_models_profiles:
            model_names = list(AVAILABLE_MODELS.keys())
            total_models = len(model_names)
            total_profiles = len(STYLE_NAMES)
            total_jobs = total_models * total_profiles

            print(
                f"[GRID] Starting all-models × all-profiles run: "
                f"{total_models} models × {total_profiles} profiles × batch {batch_size}",
                flush=True,
            )

            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_run_dir_name = f"all_models_profiles_{run_timestamp}"
            base_run_dir = os.path.join(OUTPUT_DIR, base_run_dir_name)
            os.makedirs(base_run_dir, exist_ok=True)

            summary_lines: List[str] = []

            for mi, mk in enumerate(model_names):
                print(f"[GRID] Model {mi+1}/{total_models}: {mk}", flush=True)

                if mode == "Image to Image" and init_image is not None:
                    _ensure_img2img(mk, scheduler_name)
                    local_mode_used = "img2img"
                    local_pipe = _img2img_pipe
                else:
                    _ensure_txt2img(mk, scheduler_name)
                    local_mode_used = "txt2img"
                    local_pipe = _txt2img_pipe

                model_slug = _sanitize_slug(mk)
                model_run_dir = os.path.join(base_run_dir, f"model_{model_slug}")
                os.makedirs(model_run_dir, exist_ok=True)

                per_model_lines = [f"<b>Model {mi+1}/{total_models}:</b> {mk}"]

                for pi, prof_name in enumerate(STYLE_NAMES):
                    global_idx = mi * total_profiles + pi
                    if progress is not None:
                        progress(
                            (global_idx + 1) / float(total_jobs),
                            desc=(
                                f"Model {mi+1}/{total_models}: {mk} | "
                                f"Profile {pi+1}/{total_profiles}: {prof_name}"
                            ),
                        )

                    profile_seed_base = base_seed + global_idx * 1000000
                    print(
                        f"[GRID]   Profile {pi+1}/{total_profiles}: {prof_name} "
                        f"(seed_base={profile_seed_base})",
                        flush=True,
                    )

                    try:
                        images, seeds, paths, effective_negative, downgrade_msg = _generate_single_profile(
                            pipe=local_pipe,
                            mode_used=local_mode_used,
                            model_key=mk,
                            scheduler_name=scheduler_name,
                            steps=steps,
                            guidance_scale=guidance_scale,
                            prompt=prompt,
                            negative_prompt_in=negative_prompt_in,
                            style_name=prof_name,
                            width=width,
                            height=height,
                            batch_size=batch_size,
                            base_seed=profile_seed_base,
                            init_image=init_image if local_mode_used == "img2img" else None,
                            strength=strength,
                            run_dir=model_run_dir,
                            run_timestamp=run_timestamp,
                        )
                        all_images.extend(images)
                        all_saved_paths.extend(paths)
                        all_seeds.extend(seeds)
                        effective_negative_final = effective_negative

                        num_imgs = len(images)
                        if num_imgs > 0:
                            w0, h0 = images[0].size
                        else:
                            w0, h0 = width, height

                        print(
                            f"[GRID]   ✔ Profile {pi+1}/{total_profiles}: {prof_name} "
                            f"→ {num_imgs} images at ~{w0}x{h0}",
                            flush=True,
                        )
                        if downgrade_msg:
                            print(
                                f"[GRID]     (auto-resize note: {downgrade_msg})",
                                flush=True,
                            )

                        info_line = (
                            f"&nbsp;&nbsp;Profile {pi+1}/{total_profiles}: <b>{prof_name}</b> "
                            f"→ {len(images)} images"
                        )
                        if downgrade_msg:
                            info_line += f" <span style='color:orange;'>(auto-resize: {downgrade_msg})</span>"
                        per_model_lines.append(info_line)

                    except RuntimeError as e:
                        msg = str(e)
                        if "CUDA out of memory" in msg:
                            per_job_failures.append(
                                f"OOM for model <b>{mk}</b>, profile <b>{prof_name}</b> "
                                f"at requested {width}x{height}, batch {batch_size}."
                            )
                            print(
                                f"[GRID][OOM] model={mk} profile={prof_name} "
                                f"requested={width}x{height} batch={batch_size}",
                                flush=True,
                            )
                        else:
                            per_job_failures.append(
                                f"Error for model <b>{mk}</b>, profile <b>{prof_name}</b>: {msg}"
                            )
                            print(
                                f"[GRID][FAIL] model={mk} profile={prof_name}: {msg}",
                                flush=True,
                            )
                        continue

                summary_lines.append("<br>".join(per_model_lines))

            total_generated = len(all_images)
            print(
                f"[GRID] Completed all-models × all-profiles run: "
                f"{total_generated} images total.",
                flush=True,
            )

            profile_progress_text = (
                f"<b>All models × all profiles run completed (with OOM-safe handling).</b><br>"
                f"Base folder: {base_run_dir}<br><br>"
                + "<br><br>".join(summary_lines)
                + f"<br><br><b>Total images generated:</b> {total_generated} "
                f"(models: {total_models}, profiles: {total_profiles}, batch size: {batch_size})"
            )

            if per_job_failures:
                warnings.append(
                    f"{len(per_job_failures)} profiles/models were skipped after OOM or other errors."
                )
                profile_progress_text += (
                    "<br><br><b>Failures / Skipped jobs:</b><br>" + "<br>".join(per_job_failures)
                )

        # ---------------------------
        # v13: ALL profiles for selected model (OOM-safe per profile)
        # ---------------------------
        elif do_all_profiles:
            profile_names = STYLE_NAMES
            total_profiles = len(profile_names)

            print(
                f"[MULTI] Starting multi-profile run for model {model_key}: "
                f"{total_profiles} profiles × batch {batch_size}",
                flush=True,
            )

            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir_name = f"multi_profiles_{run_timestamp}"
            run_dir = os.path.join(OUTPUT_DIR, run_dir_name)
            os.makedirs(run_dir, exist_ok=True)

            profile_progress_lines: List[str] = []

            for idx, prof_name in enumerate(profile_names):
                if progress is not None:
                    progress(
                        (idx + 1) / float(total_profiles),
                        desc=f"Profile {idx + 1}/{total_profiles}: {prof_name}",
                    )

                profile_seed_base = base_seed + idx * 1000000
                print(
                    f"[MULTI] Profile {idx+1}/{total_profiles}: {prof_name} "
                    f"(seed_base={profile_seed_base})",
                    flush=True,
                )

                try:
                    images, seeds, paths, effective_negative, downgrade_msg = _generate_single_profile(
                        pipe=pipe,
                        mode_used=mode_used,
                        model_key=model_key,
                        scheduler_name=scheduler_name,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        prompt=prompt,
                        negative_prompt_in=negative_prompt_in,
                        style_name=prof_name,
                        width=width,
                        height=height,
                        batch_size=batch_size,
                        base_seed=profile_seed_base,
                        init_image=init_image if mode_used == "img2img" else None,
                        strength=strength,
                        run_dir=run_dir,
                        run_timestamp=run_timestamp,
                    )
                    all_images.extend(images)
                    all_saved_paths.extend(paths)
                    all_seeds.extend(seeds)
                    effective_negative_final = effective_negative

                    line = (
                        f"Profile {idx+1} / {total_profiles}: <b>{prof_name}</b> "
                        f"generated {len(images)} images."
                    )
                    if downgrade_msg:
                        line += f" <span style='color:orange;'>(auto-resize: {downgrade_msg})</span>"
                    profile_progress_lines.append(line)

                    print(
                        f"[MULTI] ✔ Profile {idx+1}/{total_profiles}: {prof_name} "
                        f"→ {len(images)} images",
                        flush=True,
                    )
                    if downgrade_msg:
                        print(
                            f"[MULTI]   (auto-resize note: {downgrade_msg})",
                            flush=True,
                        )

                except RuntimeError as e:
                    msg = str(e)
                    if "CUDA out of memory" in msg:
                        per_job_failures.append(
                            f"OOM for profile <b>{prof_name}</b> at requested {width}x{height}, "
                            f"batch {batch_size}."
                        )
                        print(
                            f"[MULTI][OOM] profile={prof_name} "
                            f"requested={width}x{height} batch={batch_size}",
                            flush=True,
                        )
                    else:
                        per_job_failures.append(
                            f"Error for profile <b>{prof_name}</b>: {msg}"
                        )
                        print(
                            f"[MULTI][FAIL] profile={prof_name}: {msg}",
                            flush=True,
                        )
                    continue

            profile_progress_text = (
                f"<b>Multi-profile run completed into folder:</b><br>{run_dir}<br><br>"
                + "<br>".join(profile_progress_lines)
                + f"<br><br><b>Total images generated:</b> {len(all_images)} "
                f"(profiles: {total_profiles}, batch size: {batch_size})"
            )
            print(
                f"[MULTI] Completed multi-profile run for {model_key}: "
                f"{len(all_images)} images total.",
                flush=True,
            )
            if per_job_failures:
                warnings.append(
                    f"{len(per_job_failures)} profiles were skipped after OOM or other errors."
                )
                profile_progress_text += (
                    "<br><br><b>Failures / Skipped profiles:</b><br>"
                    + "<br>".join(per_job_failures)
                )

        # ---------------------------
        # v12: single profile, single model (OOM-safe)
        # ---------------------------
        else:
            style_cfg = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None / Raw"])
            styled_prompt = _append_style_suffix(prompt or "", style_cfg.get("prompt_suffix", ""))
            effective_negative = _append_style_suffix(
                negative_prompt_in or "",
                style_cfg.get("negative_suffix", ""),
            )
            effective_negative_final = effective_negative

            profile_scheduler = style_cfg.get("default_scheduler") or scheduler_name
            profile_steps = int(style_cfg.get("default_steps") or steps)

            apply_scheduler(pipe, profile_scheduler)

            seeds = [base_seed + i for i in range(batch_size)]

            if mode == "Image to Image" and init_image is not None and _img2img_pipe is not None:
                mode_used = "img2img"
                init_pil = Image.fromarray(init_image.astype("uint8")) if isinstance(
                    init_image, np.ndarray
                ) else init_image

                images, used_w, used_h, downgrade_msg = _safe_generate_images_with_retries(
                    pipe=pipe,
                    mode_used="img2img",
                    prompt=styled_prompt,
                    negative_prompt=effective_negative,
                    steps=profile_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    base_seed=base_seed,
                    init_image=init_pil,
                    strength=strength,
                )
                if downgrade_msg:
                    warnings.append(downgrade_msg)
                all_images = images
                actual_w = used_w
                actual_h = used_h
            else:
                mode_used = "txt2img"
                if use_parallel_batch and batch_size > 1:
                    # Threaded batch remains as-is; OOM here will bubble to outer except.
                    def worker(one_seed: int) -> Image.Image:
                        gen = torch.Generator(device=DEVICE).manual_seed(one_seed)
                        result = pipe(
                            prompt=styled_prompt,
                            negative_prompt=effective_negative,
                            num_inference_steps=profile_steps,
                            guidance_scale=float(guidance_scale),
                            width=width,
                            height=height,
                            generator=gen,
                            num_images_per_prompt=1,
                        )
                        return result.images[0]

                    max_workers = min(batch_size, 4)
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        all_images = list(ex.map(worker, seeds))
                    warnings.append(
                        "Experimental threaded batch enabled. On a single GPU this may or may not be faster; "
                        "disable it if you see instability or slower performance."
                    )
                    actual_w = width
                    actual_h = height
                else:
                    images, used_w, used_h, downgrade_msg = _safe_generate_images_with_retries(
                        pipe=pipe,
                        mode_used="txt2img",
                        prompt=styled_prompt,
                        negative_prompt=effective_negative,
                        steps=profile_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        batch_size=batch_size,
                        base_seed=base_seed,
                        init_image=None,
                        strength=strength,
                    )
                    if downgrade_msg:
                        warnings.append(downgrade_msg)
                    all_images = images
                    actual_w = used_w
                    actual_h = used_h

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            slug = _sanitize_slug(prompt if prompt else style_name)
            saved_paths: List[str] = []
            for idx, img in enumerate(all_images):
                seed_val = seeds[idx] if idx < len(seeds) else base_seed
                filename = f"{timestamp}_{slug}_seed{seed_val}_{idx+1:02d}.png"
                out_path = os.path.join(OUTPUT_DIR, filename)
                try:
                    img.save(out_path)
                    saved_paths.append(out_path)
                except Exception:
                    saved_paths.append(out_path + " (save failed)")

            all_saved_paths = saved_paths
            all_seeds = seeds

            job_entry = {
                "timestamp": timestamp,
                "mode": mode_used,
                "multi_profile": False,
                "profile_style": style_name,
                "prompt": prompt,
                "styled_prompt": styled_prompt,
                "negative_prompt": negative_prompt_in,
                "effective_negative_prompt": effective_negative,
                "model_key": model_key,
                "model_id": AVAILABLE_MODELS.get(model_key),
                "scheduler": profile_scheduler,
                "steps": profile_steps,
                "guidance_scale": guidance_scale,
                "width": actual_w,
                "height": actual_h,
                "batch_size": batch_size,
                "seed_base": base_seed,
                "seeds": seeds,
                "paths": saved_paths,
                "error": None,
            }
            _log_job(job_entry)
            profile_progress_text = "Single profile mode."
            print(
                f"[SINGLE] Generated {len(all_images)} images "
                f"for model={model_key}, style={style_name}, mode={mode_used}.",
                flush=True,
            )

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            error_msg = (
                "CUDA out of memory during generation. Some parts of the job may have failed. "
                "Try lowering resolution, batch size, or steps, or switch to a lighter model for future runs."
            )
            warnings.append(error_msg)
            print(f"[JOB][OOM] {error_msg}", flush=True)
        else:
            error_msg = f"Generation failed: {e}"
            warnings.append(error_msg)
            print(f"[JOB][FAIL] {error_msg}", flush=True)
        # do NOT clear all_images

    if all_images:
        gallery_images = all_images
    else:
        gallery_images = None

    if not all_seeds:
        all_seeds = []

    job_lines: List[str] = [
        f"<b>Model:</b> {model_key} ({AVAILABLE_MODELS.get(model_key)})",
        f"<b>Mode:</b> {mode}",
        f"<b>Requested resolution:</b> {width}x{height} &nbsp; | &nbsp; <b>Batch:</b> {batch_size}",
    ]

    if do_all_models_profiles:
        job_lines.append(
            f"<b>All models × profiles run (OOM-safe):</b> "
            f"{len(AVAILABLE_MODELS)} models × {len(STYLE_NAMES)} profiles "
            f"× batch {batch_size} = {len(all_images)} images."
        )
        if all_saved_paths:
            job_lines.append(
                "<b>Example saved files (first 10):</b><br>"
                + "<br>".join(all_saved_paths[:10])
            )
    elif do_all_profiles:
        job_lines.append(
            f"<b>Multi-profile run:</b> {len(STYLE_NAMES)} profiles × batch {batch_size} "
            f"= {len(all_images)} images."
        )
        if all_saved_paths:
            job_lines.append(
                "<b>Example saved files (first 10):</b><br>"
                + "<br>".join(all_saved_paths[:10])
            )
    else:
        job_lines.append(
            f"<b>Style:</b> {style_name} &nbsp; | &nbsp; <b>Seed base:</b> {base_seed} "
            f"&nbsp; | &nbsp; <b>Seeds:</b> {', '.join(str(s) for s in all_seeds)}"
        )
        if all_saved_paths:
            job_lines.append("<b>Saved files:</b><br>" + "<br>".join(all_saved_paths))

    if per_job_failures:
        job_lines.append(
            f"<b>Failures / Skipped jobs:</b><br>" + "<br>".join(per_job_failures)
        )

    if error_msg:
        job_lines.append(f"<b>Error:</b> {error_msg}")

    job_html = "<br>".join(job_lines)

    all_warnings = warnings
    warn_html = _format_warning_html(all_warnings)

    return gallery_images, job_html, warn_html, effective_negative_final, model_status, profile_progress_text


# ------------------------------------------------------------
# Animate Steps - Light Mode
# ------------------------------------------------------------

MOTION_PHRASES = [
    "subtle motion blur",
    "slight camera movement",
    "gentle parallax shift",
    "animated feel",
    "frame-by-frame motion",
    "dynamic composition",
]


def generate_animate_sequence(
    model_key: str,
    scheduler_name: str,
    steps: int,
    guidance_scale: float,
    style_name: str,
    base_prompt: str,
    negative_prompt_in: str,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    variation_strength: float,
    enable_vram_check: bool,
):
    global _txt2img_pipe

    model_status, warn_html1 = _ensure_txt2img(model_key, scheduler_name)

    style_cfg = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None / Raw"])
    styled_prompt_base = _append_style_suffix(base_prompt or "", style_cfg.get("prompt_suffix", ""))
    effective_negative = _append_style_suffix(
        negative_prompt_in or "",
        style_cfg.get("negative_suffix", ""),
    )

    width = int(max(512, width // 64 * 64))
    height = int(max(512, height // 64 * 64))

    warnings: List[str] = []
    if warn_html1:
        pass

    if enable_vram_check:
        msg, suggestion = _estimate_vram_and_suggest(width, height, num_frames, steps)
        if msg:
            warnings.append(msg)
        if suggestion:
            sw, sh = suggestion
            warnings.append(f"Suggested safe resolution (heuristic): around {sw}x{sh} or lower.")

    if seed is None or seed < 0:
        base_seed = random.randint(0, 2 ** 32 - 1)
    else:
        base_seed = int(seed)

    print(
        f"[ANIMATE] Starting sequence: model={model_key} style={style_name} "
        f"frames={num_frames} size={width}x{height} seed_base={base_seed}",
        flush=True,
    )

    pipe = _txt2img_pipe
    frames: List[Image.Image] = []
    seeds_used: List[int] = []
    error_msg = None

    try:
        for idx in range(num_frames):
            frame_seed = base_seed + idx
            seeds_used.append(frame_seed)
            print(
                f"[ANIMATE]   Frame {idx+1}/{num_frames} seed={frame_seed}",
                flush=True,
            )
            gen = torch.Generator(device=DEVICE).manual_seed(frame_seed)

            frame_prompt = styled_prompt_base
            if variation_strength > 0:
                num_phrases = 1 if variation_strength < 0.5 else 2
                phrases = random.sample(MOTION_PHRASES, k=min(num_phrases, len(MOTION_PHRASES)))
                frame_prompt = _append_style_suffix(frame_prompt, ", ".join(phrases))

            out = pipe(
                prompt=frame_prompt,
                negative_prompt=effective_negative,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                width=width,
                height=height,
                generator=gen,
                num_images_per_prompt=1,
            )
            frames.append(out.images[0])
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            error_msg = (
                "CUDA out of memory during Animate Steps generation. "
                "Try lowering resolution, frame count, or steps."
            )
            warnings.append(error_msg)
            print(f"[ANIMATE][OOM] {error_msg}", flush=True)
        else:
            error_msg = f"Animate Steps generation failed: {e}"
            warnings.append(error_msg)
            print(f"[ANIMATE][FAIL] {error_msg}", flush=True)
        frames = []

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _sanitize_slug(base_prompt if base_prompt else style_name)
    saved_paths: List[str] = []
    for idx, img in enumerate(frames):
        frame_idx = idx + 1
        filename = f"{timestamp}_{slug}_frame_{frame_idx:03d}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        try:
            img.save(out_path)
            saved_paths.append(out_path)
        except Exception:
            saved_paths.append(out_path + " (save failed)")

    print(
        f"[ANIMATE] Completed sequence: {len(frames)} frames saved.",
        flush=True,
    )

    job_entry = {
        "timestamp": timestamp,
        "mode": "animate_steps",
        "prompt": base_prompt,
        "styled_prompt_base": styled_prompt_base,
        "negative_prompt": negative_prompt_in,
        "effective_negative_prompt": effective_negative,
        "style": style_name,
        "model_key": model_key,
        "model_id": AVAILABLE_MODELS.get(model_key),
        "scheduler": scheduler_name,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "seed_base": base_seed,
        "seeds": seeds_used,
        "paths": saved_paths,
        "error": error_msg,
    }
    _log_job(job_entry)

    if frames:
        gallery_frames = frames
    else:
        gallery_frames = None

    info_lines = [
        f"<b>Animate Steps:</b> {num_frames} frames generated.",
        f"<b>Model:</b> {model_key} | <b>Style:</b> {style_name}",
        f"<b>Resolution:</b> {width}x{height} | <b>Steps:</b> {steps} | <b>Guidance:</b> {guidance_scale}",
        f"<b>Seed base:</b> {base_seed} &nbsp; | &nbsp; <b>Seeds:</b> {', '.join(str(s) for s in seeds_used)}",
        "Saved sequential frames like: frame_001, frame_002, ... (with timestamp + slug prefix).",
    ]
    if saved_paths:
        info_lines.append("<b>Saved files:</b><br>" + "<br>".join(saved_paths))
    if error_msg:
        info_lines.append(f"<b>Error:</b> {error_msg}")

    warn_html = _format_warning_html(warnings)
    info_html = "<br>".join(info_lines)
    return gallery_frames, info_html, warn_html, effective_negative, model_status


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------

def build_ui():
    with gr.Blocks() as demo:
        gr.HTML("""
        <style>
            .model-status {
                color: green;
            }
        </style>
        """)

        gr.Markdown("# SDXL DGX Image Lab v15 🚀")
        gr.Markdown(
            "Single-GPU SDXL image lab running on DGX. "
            "Models are loaded from the local HuggingFace cache at `/root/.cache/huggingface`.\n\n"
            "**New in v15:** OOM-aware auto resolution downscaling and resilient all-models × all-profiles "
            "batch runs (failed profiles are skipped, successful ones preserved)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0],
                )
            with gr.Column(scale=1):
                scheduler_dropdown = gr.Dropdown(
                    label="Sampler / Scheduler",
                    choices=SCHEDULER_NAMES,
                    value="Default",
                )
            with gr.Column(scale=1):
                steps_slider = gr.Slider(
                    label="Steps",
                    minimum=4,
                    maximum=60,
                    step=1,
                    value=30,
                )

        with gr.Row():
            with gr.Column(scale=2):
                style_dropdown = gr.Dropdown(
                    label="Style Preset",
                    choices=STYLE_NAMES,
                    value="Photoreal",
                )
            with gr.Column(scale=1):
                guidance_slider = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=7.5,
                )
            with gr.Column(scale=1):
                vram_check_checkbox = gr.Checkbox(
                    label="Enable VRAM pre-check (heuristic)",
                    value=True,
                )

        with gr.Row():
            with gr.Column(scale=2):
                negative_prompt_box = gr.Textbox(
                    label="Negative Prompt (auto-augmented by style)",
                    lines=2,
                    value="lowres, bad anatomy, text, error, extra limbs, cropped, worst quality, low quality",
                )
            with gr.Column(scale=1):
                threaded_batch_checkbox = gr.Checkbox(
                    label="Threaded Batch (experimental, txt2img only)",
                    value=False,
                )

        with gr.Row():
            do_all_profiles_checkbox = gr.Checkbox(
                label="Do ALL Profiles (multi-style batch for selected model)",
                value=False,
            )
            do_all_models_profiles_checkbox = gr.Checkbox(
                label="Do ALL Models × Profiles (v15 grid, overnight mode)",
                value=False,
            )

        model_status_md = gr.Markdown(value="No model loaded yet.")
        warnings_md = gr.HTML(value="")
        profile_progress_md = gr.HTML(value="")

        with gr.Tab("Txt2Img / Img2Img"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_box = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe what you want to see...",
                        lines=4,
                    )
                    aspect_ratio_dropdown = gr.Dropdown(
                        label="Aspect Ratio Preset",
                        choices=list(ASPECT_RATIOS.keys()),
                        value="16:9 (1024x576)",
                    )
                    with gr.Row():
                        width_slider = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=1024,
                        )
                        height_slider = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=576,
                        )
                    with gr.Row():
                        batch_slider = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=1,
                        )
                        seed_box = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                        )
                    mode_radio = gr.Radio(
                        label="Mode",
                        choices=["Text to Image", "Image to Image"],
                        value="Text to Image",
                    )
                    with gr.Row(visible=False) as img2img_row:
                        init_image = gr.Image(
                            label="Init Image for Img2Img",
                            type="numpy",
                        )
                        strength_slider = gr.Slider(
                            label="Img2Img Strength",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                        )
                    img2img_status_md = gr.Markdown("Img2Img pipeline: not loaded yet.")
                    with gr.Row():
                        load_model_btn = gr.Button("Load / Switch Model")
                        preload_img2img_btn = gr.Button("Preload Img2Img (lazy-load)")
                    generate_btn = gr.Button("Generate")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Output Images", columns=2, height=512)
                    job_info_md = gr.HTML(label="Job Info")

        with gr.Tab("Animate Steps (Light Mode)"):
            with gr.Row():
                with gr.Column(scale=2):
                    animate_prompt_box = gr.Textbox(
                        label="Base Prompt",
                        placeholder="Base description for your motion-like sequence...",
                        lines=3,
                    )
                    with gr.Row():
                        animate_width_slider = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=1024,
                        )
                        animate_height_slider = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=576,
                        )
                    with gr.Row():
                        num_frames_slider = gr.Slider(
                            label="Number of Frames",
                            minimum=2,
                            maximum=24,
                            step=1,
                            value=8,
                        )
                        animate_seed_box = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                        )
                    variation_slider = gr.Slider(
                        label="Prompt Variation per Frame",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                    )
                    animate_btn = gr.Button("Generate Sequence")
                with gr.Column(scale=2):
                    animate_gallery = gr.Gallery(label="Frames", columns=4, height=512)
                    animate_info_md = gr.HTML(label="Sequence Info")

        def on_mode_change(mode_val: str):
            return gr.update(visible=(mode_val == "Image to Image"))

        demo.load(
            fn=init_on_app_load,
            inputs=None,
            outputs=[model_dropdown, model_status_md, warnings_md],
        )

        style_dropdown.change(
            fn=on_style_change,
            inputs=[style_dropdown, negative_prompt_box, scheduler_dropdown, steps_slider],
            outputs=[negative_prompt_box, scheduler_dropdown, steps_slider],
        )

        aspect_ratio_dropdown.change(
            fn=on_aspect_ratio_change,
            inputs=[aspect_ratio_dropdown],
            outputs=[width_slider, height_slider],
        )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[img2img_row],
        )

        load_model_btn.click(
            fn=ui_load_model,
            inputs=[model_dropdown, scheduler_dropdown],
            outputs=[model_status_md, warnings_md],
        )

        preload_img2img_btn.click(
            fn=ui_preload_img2img,
            inputs=[model_dropdown, scheduler_dropdown],
            outputs=[img2img_status_md, warnings_md],
        )

        generate_btn.click(
            fn=generate_images,
            inputs=[
                model_dropdown,
                scheduler_dropdown,
                steps_slider,
                guidance_slider,
                style_dropdown,
                prompt_box,
                negative_prompt_box,
                aspect_ratio_dropdown,
                width_slider,
                height_slider,
                batch_slider,
                seed_box,
                mode_radio,
                init_image,
                strength_slider,
                vram_check_checkbox,
                threaded_batch_checkbox,
                do_all_profiles_checkbox,
                do_all_models_profiles_checkbox,
            ],
            outputs=[
                gallery,
                job_info_md,
                warnings_md,
                negative_prompt_box,
                model_status_md,
                profile_progress_md,
            ],
        )

        animate_btn.click(
            fn=generate_animate_sequence,
            inputs=[
                model_dropdown,
                scheduler_dropdown,
                steps_slider,
                guidance_slider,
                style_dropdown,
                animate_prompt_box,
                negative_prompt_box,
                animate_width_slider,
                animate_height_slider,
                num_frames_slider,
                animate_seed_box,
                variation_slider,
                vram_check_checkbox,
            ],
            outputs=[
                animate_gallery,
                animate_info_md,
                warnings_md,
                negative_prompt_box,
                model_status_md,
            ],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7868)
