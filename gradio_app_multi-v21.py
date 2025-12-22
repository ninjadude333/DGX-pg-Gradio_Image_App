#!/usr/bin/env python3
"""
SDXL DGX Image Lab v21 🚀

New in v21:
- Model name in output folder
- Select/Deselect All buttons for models and profiles
- New adult content profiles (Sexy/Adult, Porn, LucasArts)
"""

import os
import sys
import time
import json
import random
import threading
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re

# Disable Gradio analytics completely
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "0"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Suppress HuggingFace offline warnings
import warnings
warnings.filterwarnings("ignore", message=".*offline mode is enabled.*")
warnings.filterwarnings("ignore", message=".*Couldn't connect to the Hub.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*clean_caption.*")
warnings.filterwarnings("ignore", message=".*legacy behaviour.*T5Tokenizer.*")

import torch
import gradio as gr
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    PixArtSigmaPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

# Global state
_txt2img_pipe = None
_img2img_pipe = None
_current_model_key = None
_current_model_id = None
_current_scheduler = None
_state_lock = threading.Lock()
_abort_flag = False

# Small in-process cache so switching back/forth between a couple of models is fast.
# (Keeps behavior identical when you don't switch models.)
_pipe_cache: Dict[Tuple[str, str], Any] = {}
_pipe_cache_order: List[Tuple[str, str]] = []
_PIPE_CACHE_MAX = int(os.environ.get("PIPE_CACHE_MAX", "2"))

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"[INIT] Using device: {DEVICE}")
    print(f"[INIT] Available GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"[INIT]   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print(f"[INIT] Using device: {DEVICE}")

# Output directory
OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Instance-specific logging
INSTANCE_ID = os.environ.get("INSTANCE_ID", os.environ.get("HOSTNAME", "default"))
JOBS_LOG_PATH = OUTPUT_DIR / f"jobs_{INSTANCE_ID}.log"

# Available models (diffusers-compatible only)
AVAILABLE_MODELS = {
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "RealVis XL v5.0": "SG161222/RealVisXL_V5.0",
    "CyberRealistic XL 5.8": "John6666/cyberrealistic-xl-v58-sdxl",
    "Animagine XL 4.0": "cagliostrolab/animagine-xl-4.0",
    "Juggernaut XL": "stablediffusionapi/juggernautxl",
    "PixArt Sigma XL 1024": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "SD3 Medium": "stabilityai/stable-diffusion-3-medium-diffusers",  # Requires multi-GPU or >24GB VRAM
}

# Model loader types (keeps existing logic, only routes PixArt to the correct pipeline)
MODEL_TYPES = {
    **{k: "auto" for k in AVAILABLE_MODELS.keys()},
    "PixArt Sigma XL 1024": "pixart",
    "SD3 Medium": "sd3",
}

# Schedulers
SCHEDULERS = {
    "Default": None,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "UniPC": UniPCMultistepScheduler,
}

# Enhanced style profiles with new artist/genre profiles
STYLE_PROFILES = {
    "None / Raw": {
        "prompt_suffix": "",
        "negative_suffix": "",
        "scheduler": None,
        "steps": None,
    },
    "Photoreal": {
        "prompt_suffix": ", photorealistic, highly detailed, sharp focus, professional photography",
        "negative_suffix": "cartoon, anime, painting, drawing, illustration, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Cinematic": {
        "prompt_suffix": ", cinematic lighting, dramatic composition, film grain, movie still, professional cinematography",
        "negative_suffix": "amateur, snapshot, phone camera, lowres, bad composition, overexposed, underexposed, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Anime / Vibrant": {
        "prompt_suffix": ", anime style, vibrant colors, cel shading, detailed illustration, high quality anime art",
        "negative_suffix": "photorealistic, realistic, 3d render, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Soft Illustration": {
        "prompt_suffix": ", soft illustration, pastel colors, gentle lighting, artistic, painterly style",
        "negative_suffix": "harsh lighting, high contrast, photorealistic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Black & White": {
        "prompt_suffix": ", black and white, monochrome, high contrast, dramatic lighting, film noir style",
        "negative_suffix": "color, colorful, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Pencil Sketch": {
        "prompt_suffix": ", pencil sketch, hand drawn, artistic sketch, detailed line art, graphite drawing",
        "negative_suffix": "photorealistic, color, painting, digital art, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "35mm Film": {
        "prompt_suffix": ", 35mm film photography, film grain, vintage look, analog photography, natural lighting",
        "negative_suffix": "digital, HDR, oversaturated, lowres, bad composition, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Rotoscoping": {
        "prompt_suffix": ", rotoscoped animation style, A Scanner Darkly style, traced animation, unique visual style",
        "negative_suffix": "traditional animation, photorealistic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "R-Rated": {
        "prompt_suffix": ", mature themes, dramatic, intense, adult content, sophisticated composition",
        "negative_suffix": "childish, cartoon, lowres, bad anatomy, text, watermark, children, blurry",
        "scheduler": None,
        "steps": None,
    },
    # New v17 Artist/Genre Profiles
    "Tim Burton Style": {
        "prompt_suffix": ", Tim Burton style, gothic, dark whimsical, striped patterns, pale skin, dark circles under eyes, twisted architecture, Burton-esque character design",
        "negative_suffix": "bright colors, cheerful, normal proportions, realistic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": 30,
    },
    "Frank Frazetta Fantasy": {
        "prompt_suffix": ", Frank Frazetta style, fantasy art, muscular heroes, dramatic poses, rich colors, painterly brushstrokes, epic fantasy illustration, barbarian aesthetic",
        "negative_suffix": "modern, sci-fi, cartoon, anime, lowres, bad anatomy, text, watermark, blurry, weak composition",
        "scheduler": "DPM++ 2M",
        "steps": 35,
    },
    "Ralph Bakshi Animation": {
        "prompt_suffix": ", Ralph Bakshi animation style, rotoscoped, 1970s animation, gritty urban fantasy, adult animation, detailed backgrounds, atmospheric",
        "negative_suffix": "Disney style, cute, childish, clean animation, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": 28,
    },
    "H.R. Giger Biomechanical": {
        "prompt_suffix": ", H.R. Giger style, biomechanical, alien, dark surreal, organic machinery, xenomorph aesthetic, airbrushed, monochromatic, nightmarish beauty",
        "negative_suffix": "colorful, cheerful, organic only, mechanical only, lowres, bad anatomy, text, watermark, blurry, cute",
        "scheduler": "Euler",
        "steps": 40,
    },
    "Dark Fantasy / Grimdark": {
        "prompt_suffix": ", dark fantasy, grimdark, ominous atmosphere, muted colors, gothic horror, medieval darkness, foreboding mood, dramatic shadows",
        "negative_suffix": "bright, cheerful, colorful, happy, lowres, bad anatomy, text, watermark, blurry, cartoon",
        "scheduler": "DPM++ 2M",
        "steps": 32,
    },
    # v16 Extended Profiles
    "Watercolor": {
        "prompt_suffix": ", watercolor painting, soft brushstrokes, flowing colors, artistic, painterly, delicate washes",
        "negative_suffix": "digital art, sharp edges, harsh lines, photorealistic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Hyper-Realistic Portrait": {
        "prompt_suffix": ", hyper-realistic portrait, extremely detailed skin, perfect lighting, studio photography, high resolution",
        "negative_suffix": "cartoon, anime, painting, illustration, lowres, bad anatomy, bad hands, text, watermark, blurry",
        "scheduler": None,
        "steps": 35,
    },
    "ISOTOPIA Sci-Fi Blueprint": {
        "prompt_suffix": ", technical blueprint, sci-fi schematic, isometric view, clean lines, technical drawing, futuristic design",
        "negative_suffix": "organic, natural, hand-drawn, sketchy, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Pixar-ish Soft CG": {
        "prompt_suffix": ", Pixar style, soft 3D rendering, colorful, family-friendly, smooth surfaces, appealing character design",
        "negative_suffix": "realistic, dark, gritty, harsh lighting, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Pixel Art / Isometric Game": {
        "prompt_suffix": ", pixel art, isometric game art, retro gaming, 16-bit style, clean pixels, game sprite",
        "negative_suffix": "smooth, anti-aliased, photorealistic, high resolution, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Low-Poly 3D / PS1": {
        "prompt_suffix": ", low-poly 3D, PS1 style, retro gaming, simple geometry, flat shading, nostalgic 90s aesthetic",
        "negative_suffix": "high-poly, smooth surfaces, modern graphics, photorealistic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Product Render / Industrial": {
        "prompt_suffix": ", product photography, industrial design, clean background, studio lighting, commercial render",
        "negative_suffix": "cluttered, messy, artistic, painterly, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Isometric Tech Diagram": {
        "prompt_suffix": ", isometric technical diagram, clean lines, technical illustration, blueprint style, engineering drawing",
        "negative_suffix": "perspective, organic, artistic, sketchy, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Retro Comic / Halftone": {
        "prompt_suffix": ", retro comic book style, halftone dots, vintage colors, pop art, comic book illustration",
        "negative_suffix": "modern, digital, smooth gradients, photorealistic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Vaporwave / Synthwave": {
        "prompt_suffix": ", vaporwave aesthetic, synthwave, neon colors, retro futuristic, 80s nostalgia, cyberpunk vibes",
        "negative_suffix": "modern, realistic, muted colors, natural lighting, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Children's Book Illustration": {
        "prompt_suffix": ", children's book illustration, whimsical, colorful, friendly, soft art style, storybook art",
        "negative_suffix": "dark, scary, realistic, adult themes, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Ink & Screentone Manga": {
        "prompt_suffix": ", manga style, ink drawing, screentone, black and white, Japanese comic art, detailed linework",
        "negative_suffix": "color, photorealistic, western comic, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Analog Horror / VHS": {
        "prompt_suffix": ", analog horror, VHS aesthetic, grainy, distorted, eerie atmosphere, found footage style",
        "negative_suffix": "clean, high quality, bright, cheerful, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    "Architectural Visualization": {
        "prompt_suffix": ", architectural visualization, clean render, professional presentation, realistic materials, proper lighting",
        "negative_suffix": "sketchy, artistic, fantastical, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": None,
    },
    # v21 Adult Content Profiles
    "Sexy / Adult": {
        "prompt_suffix": ", sexy, sensual, alluring, attractive body, seductive pose, intimate, erotic atmosphere, beautiful curves, revealing, provocative",
        "negative_suffix": "clothed, covered, modest, sfw, censored, lowres, bad anatomy, text, watermark, blurry, children",
        "scheduler": None,
        "steps": 35,
    },
    "Porn / Explicit": {
        "prompt_suffix": ", explicit, hardcore, pornographic, nude, naked, sexual act, genitals visible, xxx, nsfw, uncensored",
        "negative_suffix": "clothed, sfw, censored, covered, modest, lowres, bad anatomy, text, watermark, blurry, children",
        "scheduler": None,
        "steps": 40,
    },
    "LucasArts Point & Click": {
        "prompt_suffix": ", LucasArts style, point and click adventure game, 1990s pixel art, VGA graphics, dithered colors, adventure game aesthetic, SCUMM engine style",
        "negative_suffix": "modern, 3d, photorealistic, high resolution, smooth gradients, lowres, bad anatomy, text, watermark, blurry",
        "scheduler": None,
        "steps": 28,
    },
}


def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[-\s]+', '_', text).strip('-_')[:50]


def append_job_log(job_data: Dict[str, Any]) -> None:
    """Append job data to instance-specific log file."""
    try:
        with open(JOBS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(job_data, default=str) + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to write to job log: {e}")


def _ensure_pixart_compat_env() -> None:
    """Apply safe defaults for V100-era GPUs (prevents xformers from attempting flash/triton paths).

    These are no-ops on systems where they don't matter, but prevent brittle import-time failures.
    """
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("XFORMERS_DISABLE_FLASH_ATTN", "1")
    os.environ.setdefault("DISABLE_FLASH_ATTN", "1")


def _cache_get(model_key: str, scheduler_name: str):
    key = (model_key, scheduler_name)
    return _pipe_cache.get(key)


def _cache_put(model_key: str, scheduler_name: str, pipe) -> None:
    key = (model_key, scheduler_name)
    _pipe_cache[key] = pipe
    if key in _pipe_cache_order:
        _pipe_cache_order.remove(key)
    _pipe_cache_order.append(key)
    # Evict LRU
    while len(_pipe_cache_order) > _PIPE_CACHE_MAX:
        oldest = _pipe_cache_order.pop(0)
        try:
            del _pipe_cache[oldest]
        except KeyError:
            pass


def load_model(model_key: str, scheduler_name: str = "Default") -> Tuple[bool, str]:
    """Load model with scheduler. Returns (success, message)."""
    global _txt2img_pipe, _img2img_pipe, _current_model_key, _current_model_id, _current_scheduler
    
    if model_key not in AVAILABLE_MODELS:
        return False, f"Unknown model: {model_key}"
    
    model_id = AVAILABLE_MODELS[model_key]
    model_type = MODEL_TYPES.get(model_key, "auto")
    
    with _state_lock:
        # Check if already loaded
        if (
            _current_model_key == model_key
            and _current_scheduler == scheduler_name
            and _txt2img_pipe is not None
        ):
            return True, f"Model {model_key} already loaded with {scheduler_name} scheduler"

        # Cache hit: reuse the previous pipeline if present
        cache_key = (model_key, scheduler_name)
        if cache_key in _pipe_cache:
            _txt2img_pipe = _pipe_cache[cache_key]
            _img2img_pipe = None
            _current_model_key = model_key
            _current_model_id = model_id
            _current_scheduler = scheduler_name
            # Refresh LRU order
            try:
                _pipe_cache_order.remove(cache_key)
            except ValueError:
                pass
            _pipe_cache_order.append(cache_key)
            return True, f"✅ Model {model_key} loaded from cache with {scheduler_name} scheduler"
        
        try:
            print(f"[LOAD] Loading {model_key} ({model_id}) with {scheduler_name} scheduler...")
            start_time = time.time()
            
            # Clear existing pipelines
            _txt2img_pipe = None
            _img2img_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Load txt2img pipeline (model-type aware)
            if model_type == "pixart":
                _ensure_pixart_compat_env()
                # PixArt: Use float32 (APEX RMSNorm requires it) + single GPU
                import logging
                logging.getLogger("diffusers").setLevel(logging.ERROR)
                print(f"[LOAD] PixArt using single GPU: {DEVICE} (float32 for APEX compatibility)")
                _txt2img_pipe = PixArtSigmaPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                ).to(DEVICE)
                # Enable all memory optimizations
                try:
                    _txt2img_pipe.enable_attention_slicing()
                    _txt2img_pipe.enable_vae_slicing()
                    _txt2img_pipe.enable_vae_tiling()
                except Exception:
                    pass
            elif model_type == "sd3":
                _ensure_pixart_compat_env()
                # SD3: Use device_map for multi-GPU, float32 to avoid dtype issues
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    print(f"[LOAD] SD3 using device_map='balanced' across {gpu_count} GPUs")
                    _txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
                        model_id, torch_dtype=torch.float32, device_map="balanced"
                    )
                else:
                    print(f"[LOAD] SD3 using single GPU: {DEVICE}")
                    _txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
                        model_id, torch_dtype=torch.float32
                    ).to(DEVICE)
                try:
                    _txt2img_pipe.enable_attention_slicing()
                    _txt2img_pipe.enable_vae_slicing()
                    _txt2img_pipe.enable_vae_tiling()
                except Exception:
                    pass

            else:
                try:
                    _txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                    ).to(DEVICE)
                except Exception as e:
                    print(f"[ERROR] Failed to load {model_id} with fp16 variant: {e}")
                    print(f"[RETRY] Trying without variant and safetensors...")
                    _txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                    ).to(DEVICE)
            
            # Enable memory optimizations for all models
            try:
                _txt2img_pipe.enable_vae_slicing()
            except Exception:
                pass
            
            # Apply scheduler if specified
            if scheduler_name != "Default" and scheduler_name in SCHEDULERS:
                scheduler_class = SCHEDULERS[scheduler_name]
                if scheduler_class:
                    _txt2img_pipe.scheduler = scheduler_class.from_config(_txt2img_pipe.scheduler.config)
            
            # Enable memory efficient attention if available
            if hasattr(_txt2img_pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    _txt2img_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            # Update cache (LRU)
            _pipe_cache[cache_key] = _txt2img_pipe
            _pipe_cache_order.append(cache_key)
            while len(_pipe_cache_order) > _PIPE_CACHE_MAX:
                old_key = _pipe_cache_order.pop(0)
                if old_key == cache_key:
                    continue
                try:
                    old_pipe = _pipe_cache.pop(old_key, None)
                    del old_pipe
                except Exception:
                    pass
            torch.cuda.empty_cache()
            
            _current_model_key = model_key
            _current_model_id = model_id
            _current_scheduler = scheduler_name
            
            load_time = time.time() - start_time
            message = f"✅ Model {model_key} loaded in {load_time:.1f}s with {scheduler_name} scheduler"
            print(f"[LOAD] {message}")
            return True, message
            
        except Exception as e:
            error_msg = f"❌ Failed to load {model_key}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return False, error_msg


def load_img2img_pipeline() -> bool:
    """Lazy load img2img pipeline. Returns success."""
    global _img2img_pipe
    
    if _img2img_pipe is not None:
        return True
    
    if _current_model_id is None:
        return False

    # PixArt and SD3 are text-to-image only in this app
    model_type = MODEL_TYPES.get(_current_model_key)
    if _current_model_key and model_type in ["pixart", "sd3"]:
        print(f"[LOAD] Img2Img is not supported for {_current_model_key} in this app")
        return False
    
    try:
        print("[LOAD] Loading Img2Img pipeline...")
        try:
            _img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
                _current_model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(DEVICE)
        except Exception as e:
            print(f"[ERROR] Failed to load img2img with fp16 variant: {e}")
            print(f"[RETRY] Trying without variant and safetensors...")
            _img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
                _current_model_id,
                torch_dtype=torch.float16,
            ).to(DEVICE)
        
        # Apply same scheduler as txt2img
        if _current_scheduler != "Default" and _current_scheduler in SCHEDULERS:
            scheduler_class = SCHEDULERS[_current_scheduler]
            if scheduler_class:
                _img2img_pipe.scheduler = scheduler_class.from_config(_img2img_pipe.scheduler.config)
        
        if hasattr(_img2img_pipe, "enable_xformers_memory_efficient_attention"):
            try:
                _img2img_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        print("[LOAD] ✅ Img2Img pipeline loaded")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load Img2Img pipeline: {e}")
        return False


def apply_style_profile(prompt: str, negative_prompt: str, profile_name: str) -> Tuple[str, str, Optional[str], Optional[int]]:
    """Apply style profile to prompts. Returns (styled_prompt, styled_negative, scheduler, steps)."""
    if profile_name not in STYLE_PROFILES:
        return prompt, negative_prompt, None, None
    
    profile = STYLE_PROFILES[profile_name]
    
    styled_prompt = prompt
    if profile["prompt_suffix"]:
        styled_prompt = f"{prompt}{profile['prompt_suffix']}"
    
    styled_negative = negative_prompt
    if profile["negative_suffix"]:
        if negative_prompt:
            styled_negative = f"{negative_prompt}, {profile['negative_suffix']}"
        else:
            styled_negative = profile["negative_suffix"]
    
    return styled_prompt, styled_negative, profile["scheduler"], profile["steps"]




def _save_image_any(img, path: Path) -> bool:
    """Robustly save an image-like object to disk. Returns True if saved."""
    try:
        # PIL Image
        if hasattr(img, "save"):
            img.save(str(path))
        else:
            # Try numpy array -> PIL
            import numpy as np
            from PIL import Image as PILImage
            if isinstance(img, np.ndarray):
                PILImage.fromarray(img).save(str(path))
            else:
                return False
        return path.exists()
    except Exception as e:
        print(f"[SAVE] ❌ Failed to save {path}: {e}")
        return False

def generate_images(
    mode: str,
    prompt: str,
    negative_prompt: str,
    selected_models: List[str],
    scheduler_name: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    batch_size: int,
    seed_base: int,
    selected_profiles: List[str],
    init_image: Optional[Image.Image] = None,
    img2img_strength: float = 0.6,
) -> Tuple[List[Image.Image], str, str]:
    """Main generation function."""
    
    if not prompt.strip():
        return [], "❌ Please enter a prompt", ""
    
    # Use selected models and profiles
    model_keys = selected_models if selected_models else ["SDXL Base 1.0"]
    profile_names = selected_profiles if selected_profiles else ["Photoreal"]
    
    # Create run directory with model name(s)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(model_keys) == 1:
        model_slug = slugify(model_keys[0])
        run_dir = OUTPUT_DIR / f"run_{timestamp}_{model_slug}"
    else:
        run_dir = OUTPUT_DIR / f"run_{timestamp}_multi_models"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    global _abort_flag
    _abort_flag = False
    
    all_images = []
    status_lines = []
    t_start = time.time()
    
    # Ensure dimensions are divisible by 8
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    if width < 256:
        width = 256
    if height < 256:
        height = 256
    
    # PixArt/SD3: reduce resolution if too large to avoid OOM
    for mk in model_keys:
        mt = MODEL_TYPES.get(mk, "auto")
        if mt in ["pixart", "sd3"] and (width > 1024 or height > 1024):
            scale = min(1024 / width, 1024 / height)
            width = int(width * scale // 8) * 8
            height = int(height * scale // 8) * 8
            print(f"[GENERATE] {mk} resolution reduced to {width}×{height} (VRAM limit)")
            break
    
    print(f"[GENERATE] Starting generation: {len(model_keys)} models × {len(profile_names)} profiles")
    print(f"[GENERATE] Adjusted dimensions: {width}×{height} (must be divisible by 8)")
    
    # Generate for each model/profile combination
    for m_key in model_keys:
        if _abort_flag:
            status_lines.append("❌ Job aborted by user")
            print("[GENERATE] ❌ Job aborted by user")
            break
        
        # Clear VRAM before loading new model (critical for multi-model runs)
        if len(model_keys) > 1:
            # Aggressive cleanup between models
            global _txt2img_pipe, _img2img_pipe, _pipe_cache, _pipe_cache_order
            _txt2img_pipe = None
            _img2img_pipe = None
            # Clear cache to prevent dtype contamination between float32/float16 models
            _pipe_cache.clear()
            _pipe_cache_order.clear()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            time.sleep(0.5)  # Allow GPU to fully release
        
        # Load model
        print(f"[GENERATE] Loading model: {m_key}")
        success, load_msg = load_model(m_key, scheduler_name)
        if not success:
            error_msg = f"❌ {m_key}: {load_msg}"
            status_lines.append(error_msg)
            print(f"[ERROR] {error_msg}")
            continue
        
        print(f"[GENERATE] ✅ Model loaded: {m_key}")
        
        for prof in profile_names:
            if _abort_flag:
                status_lines.append("❌ Job aborted by user")
                print("[GENERATE] ❌ Job aborted by user")
                break
            
            try:
                # Apply style profile
                styled_prompt, eff_negative, prof_scheduler, prof_steps = apply_style_profile(
                    prompt, negative_prompt, prof
                )
                
                # Use profile overrides if available
                eff_scheduler = prof_scheduler or scheduler_name
                eff_steps = prof_steps or steps
                
                # Generate seeds
                current_seed_base = seed_base
                if current_seed_base == -1:
                    current_seed_base = random.randint(0, 2**32 - 1)
                
                # Generate seeds (no forced batch limits - let user control)
                model_type = MODEL_TYPES.get(m_key, "auto")
                seeds = [current_seed_base + i for i in range(batch_size)]
                
                # Generate images
                print(f"[GENERATE] Generating {batch_size} images for {m_key} + {prof}")
                
                if mode == "Image to Image" and init_image is not None:
                    print(f"[GENERATE] Loading Img2Img pipeline...")
                    if not load_img2img_pipeline():
                        error_msg = f"❌ Failed to load Img2Img pipeline for {m_key}"
                        status_lines.append(error_msg)
                        print(f"[ERROR] {error_msg}")
                        continue
                    print(f"[GENERATE] ✅ Img2Img pipeline ready")
                    
                    imgs = []
                    for i, seed in enumerate(seeds):
                        # Img2img always uses single GPU, safe to use DEVICE
                        generator = torch.Generator(device=DEVICE).manual_seed(seed)
                        result = _img2img_pipe(
                            prompt=styled_prompt,
                            negative_prompt=eff_negative,
                            image=init_image,
                            strength=img2img_strength,
                            num_inference_steps=eff_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            width=width,
                            height=height,
                        )
                        imgs.append(result.images[0])
                else:
                    # Text to Image
                    # SD3 with device_map: generate one at a time
                    uses_device_map = model_type == "sd3" and hasattr(_txt2img_pipe, "hf_device_map")
                    if uses_device_map:
                        imgs = []
                        for seed in seeds:
                            generator = torch.Generator(device="cpu").manual_seed(seed)
                            result = _txt2img_pipe(
                                prompt=styled_prompt,
                                negative_prompt=eff_negative,
                                num_inference_steps=eff_steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                num_images_per_prompt=1,
                                generator=generator,
                            )
                            imgs.extend(result.images)
                            torch.cuda.synchronize()  # Ensure completion before next
                    else:
                        # PixArt and SDXL: batch generation on single device
                        generators = [torch.Generator(device=DEVICE).manual_seed(seed) for seed in seeds]
                        result = _txt2img_pipe(
                            prompt=styled_prompt,
                            negative_prompt=eff_negative,
                            num_inference_steps=eff_steps,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            num_images_per_prompt=len(seeds),
                            generator=generators,
                        )
                        imgs = result.images
                
                print(f"[GENERATE] Generated {len(imgs)} images, saving...")
                
                # Validate images
                if not imgs or len(imgs) == 0:
                    error_msg = f"❌ {m_key} + {prof}: No images generated"
                    status_lines.append(error_msg)
                    print(f"[ERROR] {error_msg}")
                    continue
                
                # Save images
                paths = []
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                for idx, img in enumerate(imgs):
                    prof_slug = slugify(prof)
                    prompt_slug = slugify(prompt[:50])
                    model_slug = slugify(m_key)
                    
                    if len(model_keys) > 1:
                        model_dir = run_dir / f"model_{model_slug}"
                        model_dir.mkdir(exist_ok=True)
                        filename = f"{ts}_{prof_slug}_{prompt_slug}_seed{seeds[idx]}_{idx+1:02d}.png"
                        fpath = model_dir / filename
                    else:
                        filename = f"{ts}_{prof_slug}_{prompt_slug}_seed{seeds[idx]}_{idx+1:02d}.png"
                        fpath = run_dir / filename
                    
                    saved_ok = _save_image_any(img, fpath)
                    if not saved_ok:
                        print(f"[SAVE] ⚠️ Save reported failure: {fpath}")
                    else:
                        paths.append(str(fpath))
                    all_images.append(img)
                
                # Log job
                mode_label = "img2img" if mode == "Image to Image" else "txt2img"
                append_job_log({
                    "timestamp": ts,
                    "mode": mode_label,
                    "multi_profile": len(model_keys) > 1 or len(profile_names) > 1,
                    "profile_style": prof,
                    "prompt": prompt,
                    "styled_prompt": styled_prompt,
                    "negative_prompt": negative_prompt,
                    "effective_negative_prompt": eff_negative,
                    "model_key": m_key,
                    "model_id": AVAILABLE_MODELS[m_key],
                    "scheduler": eff_scheduler,
                    "steps": eff_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "batch_size": batch_size,
                    "seed_base": current_seed_base,
                    "seeds": seeds,
                    "paths": paths,
                    "run_dir": str(run_dir),
                    "instance_id": INSTANCE_ID,
                })
                
                success_msg = f"✅ {m_key} + {prof}: {len(imgs)} images"
                status_lines.append(success_msg)
                print(f"[GENERATE] {success_msg}")
                
            except Exception as e:
                error_msg = f"❌ {m_key} + {prof}: {str(e)}"
                status_lines.append(error_msg)
                print(f"[ERROR] {error_msg}")
                # Log the full traceback for debugging
                import traceback
                print(f"[ERROR] Full traceback: {traceback.format_exc()}")
    
    t_end = time.time()
    total_time = t_end - t_start
    
    # Summary
    summary = (
        f"Generated {len(all_images)} images in {total_time/60:.1f} minutes\n"
        f"Models: {len(model_keys)} | Profiles: {len(profile_names)}\n"
        f"Run directory: {run_dir}\n"
        f"Instance: {INSTANCE_ID}\n"
    )
    
    print(f"[GENERATE] ✅ Complete: {summary.replace(chr(10), ' | ')}")
    
    status_text = "\n".join(status_lines)
    status_html = "<br>".join(status_lines)
    
    return all_images, summary, status_text


# Model metadata for tooltips and time estimates
MODEL_INFO = {
    "SDXL Base 1.0": {"type": "SDXL", "vram": "8-12 GB", "speed": "Medium", "time_per_step": 0.8},
    "SDXL Turbo": {"type": "SDXL", "vram": "8-10 GB", "speed": "Fast", "time_per_step": 0.5},
    "RealVis XL v5.0": {"type": "SDXL", "vram": "8-12 GB", "speed": "Medium", "time_per_step": 0.8},
    "CyberRealistic XL 5.8": {"type": "SDXL", "vram": "8-12 GB", "speed": "Medium", "time_per_step": 0.8},
    "Animagine XL 4.0": {"type": "SDXL", "vram": "8-12 GB", "speed": "Medium", "time_per_step": 0.8},
    "Juggernaut XL": {"type": "SDXL", "vram": "8-12 GB", "speed": "Medium", "time_per_step": 0.8},
    "PixArt Sigma XL 1024": {"type": "PixArt", "vram": "12-22 GB", "speed": "Slow", "time_per_step": 1.5},
    "SD3 Medium": {"type": "SD3", "vram": "16-28 GB", "speed": "Slow", "time_per_step": 2.0},
}

def estimate_time(models, steps, batch):
    """Estimate generation time in seconds."""
    if not models: return 0
    total = 0
    for m in models:
        info = MODEL_INFO.get(m, {"time_per_step": 1.0})
        total += steps * info["time_per_step"] * batch
    return total

def format_time(sec):
    """Format seconds as readable time."""
    if sec < 60: return f"~{int(sec)}s"
    return f"~{int(sec/60)}m {int(sec%60)}s"

def build_ui():
    """Build Gradio UI."""
    with gr.Blocks(title="SDXL DGX Image Lab v21") as demo:
        gr.HTML("<style>body { font-family: 'Segoe UI', sans-serif; }</style>")
        gr.Markdown("# SDXL DGX Image Lab v21 🚀")
        gr.Markdown("Select multiple models and profiles using checkboxes below.")
        
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
                    placeholder="Things to avoid...",
                )
                init_image = gr.Image(
                    label="Init Image (for Img2Img)",
                    type="pil",
                    visible=False,
                )
            
            with gr.Column(scale=1):
                scheduler = gr.Dropdown(
                    choices=list(SCHEDULERS.keys()),
                    value="Default",
                    label="Scheduler",
                )
                
                gr.Markdown("### Select Models")
                with gr.Row():
                    select_all_models_btn = gr.Button("Select All", size="sm", scale=1)
                    deselect_all_models_btn = gr.Button("Deselect All", size="sm", scale=1)
                model_checkboxes = gr.CheckboxGroup(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=["SDXL Base 1.0"],
                    label="Models to Generate",
                )
                model_info_md = gr.Markdown("**Selected:** SDXL Base 1.0 (Type: SDXL, VRAM: 8-12 GB, Speed: Medium)")
                
                gr.Markdown("### Select Profiles")
                with gr.Row():
                    select_all_profiles_btn = gr.Button("Select All", size="sm", scale=1)
                    deselect_all_profiles_btn = gr.Button("Deselect All", size="sm", scale=1)
                profile_checkboxes = gr.CheckboxGroup(
                    choices=list(STYLE_PROFILES.keys()),
                    value=["Photoreal"],
                    label="Profiles to Apply",
                )
                
                steps = gr.Slider(4, 80, 26, step=1, label="Steps")
                guidance_scale = gr.Slider(0.0, 20.0, 7.5, step=0.1, label="CFG Scale")
                time_estimate_md = gr.Markdown("**Est. time:** ~21s (1 model × 1 profile × 26 steps × 4 batch)")
                with gr.Row():
                    width = gr.Slider(256, 1536, 1024, step=8, label="Width")
                    height = gr.Slider(256, 1536, 576, step=8, label="Height")
                
                with gr.Row():
                    gr.Button("768×432 (16:9)", size="sm").click(lambda: (768, 432), outputs=[width, height])
                    gr.Button("1024×576 (16:9)", size="sm").click(lambda: (1024, 576), outputs=[width, height])
                    gr.Button("1280×720 (16:9)", size="sm").click(lambda: (1280, 720), outputs=[width, height])
                with gr.Row():
                    gr.Button("1024×440 (21:9)", size="sm").click(lambda: (1024, 440), outputs=[width, height])
                    gr.Button("1280×544 (21:9)", size="sm").click(lambda: (1280, 544), outputs=[width, height])
                    gr.Button("1536×656 (21:9)", size="sm").click(lambda: (1536, 656), outputs=[width, height])
                with gr.Row():
                    gr.Button("1280×392 (32:9)", size="sm").click(lambda: (1280, 392), outputs=[width, height])
                    gr.Button("1536×472 (32:9)", size="sm").click(lambda: (1536, 472), outputs=[width, height])
                    gr.Button("1024×432 (2.35:1)", size="sm").click(lambda: (1024, 432), outputs=[width, height])
                with gr.Row():
                    gr.Button("768×768 (1:1)", size="sm").click(lambda: (768, 768), outputs=[width, height])
                    gr.Button("1024×1024 (1:1)", size="sm").click(lambda: (1024, 1024), outputs=[width, height])
                    gr.Button("512×512 (1:1)", size="sm").click(lambda: (512, 512), outputs=[width, height])
                batch_size = gr.Slider(1, 10, 4, step=1, label="Batch Size")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                img2img_strength = gr.Slider(0.1, 1.0, 0.6, step=0.05, label="Img2Img Strength")
                
                with gr.Row():
                    run_btn = gr.Button("Generate 🚀", variant="primary", scale=3)
                    abort_btn = gr.Button("❌ Abort", variant="stop", scale=1)
        
        gallery = gr.Gallery(
            label="Generated Images",
            show_label=True,
            columns=4,
            height="auto",
        )
        
        with gr.Row():
            summary_box = gr.Textbox(label="Summary", lines=4, scale=2)
            status_box = gr.Textbox(label="Status", lines=4, scale=1)
        
        progress_html = gr.HTML()
        
        # Mode change handler for image visibility
        def on_mode_change(mode_val):
            return gr.update(visible=(mode_val == "Image to Image"))
        
        mode.change(
            fn=on_mode_change,
            inputs=[mode],
            outputs=[init_image],
        )
        
        # Select/Deselect All handlers
        select_all_models_btn.click(
            fn=lambda: gr.update(value=list(AVAILABLE_MODELS.keys())),
            inputs=[],
            outputs=[model_checkboxes],
        )
        
        deselect_all_models_btn.click(
            fn=lambda: gr.update(value=[]),
            inputs=[],
            outputs=[model_checkboxes],
        )
        
        select_all_profiles_btn.click(
            fn=lambda: gr.update(value=list(STYLE_PROFILES.keys())),
            inputs=[],
            outputs=[profile_checkboxes],
        )
        
        deselect_all_profiles_btn.click(
            fn=lambda: gr.update(value=[]),
            inputs=[],
            outputs=[profile_checkboxes],
        )
        
        # Update model info
        def update_model_info(sel):
            if not sel: return "No models selected"
            parts = []
            for m in sel[:2]:
                info = MODEL_INFO.get(m, {})
                parts.append(f"{m} ({info.get('type','?')}, {info.get('vram','?')}, {info.get('speed','?')})")  
            if len(sel) > 2: parts.append(f"...+{len(sel)-2} more")
            return "**Selected:** " + " | ".join(parts)
        
        model_checkboxes.change(update_model_info, [model_checkboxes], [model_info_md])
        
        # Update time estimate
        def update_time(mods, profs, st, bat):
            if not mods or not profs: return "**Est. time:** N/A"
            sec = estimate_time(mods, st, bat) * len(profs)
            return f"**Est. time:** {format_time(sec)} ({len(mods)} model × {len(profs)} profile × {st} steps × {bat} batch)"
        
        for c in [model_checkboxes, profile_checkboxes, steps, batch_size]:
            c.change(update_time, [model_checkboxes, profile_checkboxes, steps, batch_size], [time_estimate_md])
        
        # Abort handler
        def on_abort():
            global _abort_flag
            _abort_flag = True
            return "❌ Aborting job..."
        
        abort_btn.click(
            fn=on_abort,
            inputs=[],
            outputs=[status_box],
        )
        
        # Generation handler with progress
        def on_generate(*args, progress=gr.Progress()):
            try:
                progress(0, desc="Starting...")
                result = generate_images(*args)
                progress(1.0, desc="Complete!")
                return result[0], result[1], result[2], "<div style='color: green;'>✅ Generation completed successfully!</div>"
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return [], error_msg, error_msg, f"<div style='color: red;'>{error_msg}</div>"
        
        run_btn.click(
            fn=on_generate,
            inputs=[
                mode, prompt, negative_prompt, model_checkboxes, scheduler,
                steps, guidance_scale, width, height, batch_size,
                seed, profile_checkboxes,
                init_image, img2img_strength,
            ],
            outputs=[gallery, summary_box, status_box, progress_html],
            show_progress=True,
        )
        
        # Disable abort when not generating
        run_btn.click(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[abort_btn],
        )
    
    return demo


def run_headless():
    """Run in headless mode using environment variables."""
    print("[HEADLESS] Starting headless generation...")
    
    # Parse environment variables
    prompt = os.environ.get("PROMPT", "")
    negative_prompt = os.environ.get("NEGATIVE_PROMPT", "")
    model_key = os.environ.get("MODEL", "SDXL Base 1.0")
    scheduler_name = os.environ.get("SCHEDULER", "Default")
    style_profile = os.environ.get("STYLE_PROFILE", "Photoreal")
    
    steps = int(os.environ.get("STEPS", "26"))
    guidance_scale = float(os.environ.get("CFG_SCALE", "7.5"))
    width = int(os.environ.get("WIDTH", "1024"))
    height = int(os.environ.get("HEIGHT", "576"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    seed_base = int(os.environ.get("SEED", "-1"))
    
    # Boolean flags
    do_all_profiles = os.environ.get("RUN_ALL_PROFILES", "false").lower() == "true"
    do_all_models = os.environ.get("RUN_ALL_MODELS", "false").lower() == "true"
    
    # Safety: mutual exclusion
    if do_all_models and do_all_profiles:
        print("[HEADLESS] Both RUN_ALL_MODELS and RUN_ALL_PROFILES set. Using RUN_ALL_MODELS.")
        do_all_profiles = False
    
    if not prompt:
        print("[HEADLESS] ERROR: PROMPT environment variable is required")
        sys.exit(1)
    
    # Run generation
    images, summary, status = generate_images(
        mode="Text to Image",
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_key=model_key,
        scheduler_name=scheduler_name,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        batch_size=batch_size,
        seed_base=seed_base,
        style_profile=style_profile,
        do_all_profiles=do_all_profiles,
        do_all_models=do_all_models,
    )
    
    print(f"[HEADLESS] {summary}")
    print(f"[HEADLESS] Generated {len(images)} images")


if __name__ == "__main__":
    # Check for headless mode
    if os.environ.get("HEADLESS", "false").lower() == "true":
        run_headless()
    else:
        # Run UI
        ui = build_ui()
        try:
            ui.queue()
        except Exception as e:
            print(f"[WARNING] Queue setup failed: {e}")
        ui.launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("GRADIO_PORT", "7865")),
            share=False,
        )