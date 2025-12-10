#!/usr/bin/env python3
"""
SDXL DGX Image Lab v17 🚀

Key v17 Features:
- Artist/Genre profiles (Tim Burton, Frank Frazetta, Ralph Bakshi, H.R. Giger)
- Headless mode via environment variables
- Mutually exclusive checkboxes for safety
- Per-instance log files
- Single-GPU optimized for DGX
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "0"

import torch
import gradio as gr
from PIL import Image
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
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

# Device setup - single GPU only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {DEVICE}")

# Output directory
OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Instance-specific logging
INSTANCE_ID = os.environ.get("INSTANCE_ID", os.environ.get("HOSTNAME", "default"))
JOBS_LOG_PATH = OUTPUT_DIR / f"jobs_{INSTANCE_ID}.log"

# Available models
AVAILABLE_MODELS = {
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "RealVis XL v5.0": "SG161222/RealVisXL_V5.0",
    "CyberRealistic XL 5.8": "John6666/cyberrealistic-xl-v58-sdxl",
    "Animagine XL 4.0": "cagliostrolab/animagine-xl-4.0",
    "Juggernaut XL": "stablediffusionapi/juggernautxl",
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
        "negative_suffix": "childish, cartoon, lowres, bad anatomy, text, watermark, blurry",
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


def load_model(model_key: str, scheduler_name: str = "Default") -> Tuple[bool, str]:
    """Load model with scheduler. Returns (success, message)."""
    global _txt2img_pipe, _img2img_pipe, _current_model_key, _current_model_id, _current_scheduler
    
    if model_key not in AVAILABLE_MODELS:
        return False, f"Unknown model: {model_key}"
    
    model_id = AVAILABLE_MODELS[model_key]
    
    with _state_lock:
        # Check if already loaded
        if (_current_model_key == model_key and 
            _current_scheduler == scheduler_name and 
            _txt2img_pipe is not None):
            return True, f"Model {model_key} already loaded with {scheduler_name} scheduler"
        
        try:
            print(f"[LOAD] Loading {model_key} ({model_id}) with {scheduler_name} scheduler...")
            start_time = time.time()
            
            # Clear existing pipelines
            _txt2img_pipe = None
            _img2img_pipe = None
            torch.cuda.empty_cache()
            
            # Load txt2img pipeline
            _txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(DEVICE)
            
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
    
    try:
        print("[LOAD] Loading Img2Img pipeline...")
        _img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            _current_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
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
    init_image: Optional[Image.Image] = None,
    img2img_strength: float = 0.6,
) -> Tuple[List[Image.Image], str, str]:
    """Main generation function."""
    
    if not prompt.strip():
        return [], "❌ Please enter a prompt", ""
    
    # Determine what to run
    if do_all_models:
        model_keys = list(AVAILABLE_MODELS.keys())
        profile_names = list(STYLE_PROFILES.keys())
    elif do_all_profiles:
        model_keys = [model_key]
        profile_names = list(STYLE_PROFILES.keys())
    else:
        model_keys = [model_key]
        profile_names = [style_profile]
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    all_images = []
    status_lines = []
    t_start = time.time()
    
    # Generate for each model/profile combination
    for m_key in model_keys:
        # Load model
        success, load_msg = load_model(m_key, scheduler_name)
        if not success:
            status_lines.append(f"❌ {m_key}: {load_msg}")
            continue
        
        for prof in profile_names:
            try:
                # Apply style profile
                styled_prompt, eff_negative, prof_scheduler, prof_steps = apply_style_profile(
                    prompt, negative_prompt, prof
                )
                
                # Use profile overrides if available
                eff_scheduler = prof_scheduler or scheduler_name
                eff_steps = prof_steps or steps
                
                # Generate seeds
                if seed_base == -1:
                    seed_base = random.randint(0, 2**32 - 1)
                
                seeds = [seed_base + i for i in range(batch_size)]
                
                # Generate images
                if mode == "Image to Image" and init_image is not None:
                    if not load_img2img_pipeline():
                        status_lines.append(f"❌ Failed to load Img2Img pipeline for {m_key}")
                        continue
                    
                    imgs = []
                    for i, seed in enumerate(seeds):
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
                        imgs.extend(result.images)
                else:
                    # Text to Image
                    generators = [torch.Generator(device=DEVICE).manual_seed(seed) for seed in seeds]
                    result = _txt2img_pipe(
                        prompt=styled_prompt,
                        negative_prompt=eff_negative,
                        num_inference_steps=eff_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        num_images_per_prompt=batch_size,
                        generator=generators,
                    )
                    imgs = result.images
                
                # Save images
                paths = []
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                for idx, img in enumerate(imgs):
                    prof_slug = slugify(prof)
                    prompt_slug = slugify(prompt[:50])
                    model_slug = slugify(m_key)
                    
                    if do_all_models:
                        model_dir = run_dir / f"model_{model_slug}"
                        model_dir.mkdir(exist_ok=True)
                        filename = f"{ts}_{prof_slug}_{prompt_slug}_seed{seeds[idx]}_{idx+1:02d}.png"
                        fpath = model_dir / filename
                    else:
                        filename = f"{ts}_{prof_slug}_{prompt_slug}_seed{seeds[idx]}_{idx+1:02d}.png"
                        fpath = run_dir / filename
                    
                    img.save(fpath)
                    paths.append(str(fpath))
                    all_images.append(img)
                
                # Log job
                mode_label = "img2img" if mode == "Image to Image" else "txt2img"
                append_job_log({
                    "timestamp": ts,
                    "mode": mode_label,
                    "multi_profile": do_all_profiles or do_all_models,
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
                    "seed_base": seed_base,
                    "seeds": seeds,
                    "paths": paths,
                    "run_dir": str(run_dir),
                    "instance_id": INSTANCE_ID,
                })
                
                status_lines.append(f"✅ {m_key} + {prof}: {len(imgs)} images")
                
            except Exception as e:
                error_msg = f"❌ {m_key} + {prof}: {str(e)}"
                status_lines.append(error_msg)
                print(f"[ERROR] {error_msg}")
    
    t_end = time.time()
    total_time = t_end - t_start
    
    # Summary
    summary = (
        f"Generated {len(all_images)} images in {total_time/60:.1f} minutes\n"
        f"Models: {len(model_keys)} | Profiles: {len(profile_names)}\n"
        f"Run directory: {run_dir}\n"
        f"Instance: {INSTANCE_ID}\n"
    )
    
    status_html = "<br>".join(status_lines)
    
    return all_images, summary, status_html


def build_ui():
    """Build Gradio UI."""
    with gr.Blocks(title="SDXL DGX Image Lab v17") as demo:
        gr.HTML("<style>body { font-family: 'Segoe UI', sans-serif; }</style>")
        gr.Markdown("# SDXL DGX Image Lab v17 🚀")
        gr.Markdown("Enhanced with artist/genre profiles, headless mode, and improved safety features.")
        
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
                    label="Scheduler",
                )
                style_profile = gr.Dropdown(
                    choices=list(STYLE_PROFILES.keys()),
                    value="Photoreal",
                    label="Style Profile",
                )
                
                # Mutually exclusive checkboxes
                do_all_profiles = gr.Checkbox(
                    value=False,
                    label="🎨 Run ALL profiles for this model",
                )
                do_all_models = gr.Checkbox(
                    value=False,
                    label="🚀 Run ALL models × ALL profiles",
                )
                
                steps = gr.Slider(4, 80, 26, step=1, label="Steps")
                guidance_scale = gr.Slider(0.0, 20.0, 7.5, step=0.1, label="CFG Scale")
                width = gr.Slider(256, 1536, 1024, step=8, label="Width")
                height = gr.Slider(256, 1536, 576, step=8, label="Height")
                batch_size = gr.Slider(1, 10, 4, step=1, label="Batch Size")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                img2img_strength = gr.Slider(0.1, 1.0, 0.6, step=0.05, label="Img2Img Strength")
                
                run_btn = gr.Button("Generate 🚀", variant="primary")
        
        gallery = gr.Gallery(
            label="Generated Images",
            show_label=True,
            columns=4,
            height="auto",
        )
        
        summary_box = gr.Textbox(label="Summary", lines=4)
        status_html = gr.HTML()
        
        # Mutual exclusion logic
        def on_all_models_change(value):
            return gr.update(value=False) if value else gr.update()
        
        def on_all_profiles_change(value):
            return gr.update(value=False) if value else gr.update()
        
        do_all_models.change(
            fn=on_all_models_change,
            inputs=[do_all_models],
            outputs=[do_all_profiles],
        )
        
        do_all_profiles.change(
            fn=on_all_profiles_change,
            inputs=[do_all_profiles],
            outputs=[do_all_models],
        )
        
        # Generation handler
        def on_generate(*args):
            return generate_images(*args)
        
        run_btn.click(
            fn=on_generate,
            inputs=[
                mode, prompt, negative_prompt, model, scheduler,
                steps, guidance_scale, width, height, batch_size,
                seed, style_profile, do_all_profiles, do_all_models,
                init_image, img2img_strength,
            ],
            outputs=[gallery, summary_box, status_html],
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
        ui.queue()
        ui.launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("GRADIO_PORT", "7865")),
            share=False,
        )