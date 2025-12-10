import os
import time
import json
from datetime import datetime

import torch
import numpy as np
from PIL import Image as PilImage

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
import gradio as gr

# Disable Gradio analytics / telemetry to avoid pandas / NDFrame issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "0"

# -------------------------------------------------------------------------
# Paths and logging
# -------------------------------------------------------------------------

OUTPUT_DIR = "./output_images"
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "app_events.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# CUDA / multi-GPU
# -------------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this app.")

DEVICE_IDS = list(range(torch.cuda.device_count()))

# Honor HF offline mode if set (models must already be cached)
HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

# -------------------------------------------------------------------------
# Models and schedulers
# -------------------------------------------------------------------------

MODEL_CONFIGS = {
    "SDXL Base 1.0 (StabilityAI)": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "is_turbo": False,
        "heavy": False,
        "max_devices": None,  # use all visible GPUs
    },
    "SDXL Turbo (Fast preview)": {
        "repo_id": "stabilityai/sdxl-turbo",
        "is_turbo": True,
        "heavy": False,
        "max_devices": None,
    },
    "RealVis XL 5.0 (Photoreal)": {
        "repo_id": "SG161222/RealVisXL_V5.0",
        "is_turbo": False,
        "heavy": True,
        "max_devices": 2,  # limit to first 2 GPUs
    },
    "CyberRealistic XL v5.8 (Photoreal)": {
        "repo_id": "John6666/cyberrealistic-xl-v58-sdxl",
        "is_turbo": False,
        "heavy": True,
        "max_devices": 2,
    },
    "Juggernaut XL (stablediffusionapi)": {
        "repo_id": "stablediffusionapi/juggernautxl",
        "is_turbo": False,
        "heavy": True,
        "max_devices": 2,
    },
    "Animagine XL 4.0 (Anime)": {
        "repo_id": "cagliostrolab/animagine-xl-4.0",
        "is_turbo": False,
        "heavy": False,
        "max_devices": None,
    },
}

SCHEDULER_CLASSES = {
    "Default (from model)": None,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "UniPC": UniPCMultistepScheduler,
}

# -------------------------------------------------------------------------
# Style presets and aspect presets
# -------------------------------------------------------------------------

STYLE_PRESETS = {
    "None (raw prompt)": {
        "prompt_suffix": "",
        "negative_suffix": "",
    },
    "Photoreal": {
        "prompt_suffix": ", ultra realistic, 8k photograph, detailed, natural lighting, sharp focus",
        "negative_suffix": ", cartoon, anime, illustration, cgi, 3d render, oversaturated colors, painting",
    },
    "Cinematic": {
        "prompt_suffix": ", cinematic lighting, film still, depth of field, 35mm photography, high dynamic range",
        "negative_suffix": ", flat lighting, low detail, washed out",
    },
    "Soft illustration": {
        "prompt_suffix": ", soft illustration, gentle colors, subtle shading, artstation",
        "negative_suffix": ", harsh lighting, heavy contrast, photoreal skin pores",
    },
    "Anime / vibrant": {
        "prompt_suffix": ", anime style, vibrant colors, clean line art",
        "negative_suffix": ", photo, realistic skin, gritty, grainy",
    },
    "Black & White": {
        "prompt_suffix": ", black and white, high contrast, monochrome, fine grain film",
        "negative_suffix": ", color, oversaturated, neon colors",
    },
    "Pencil Sketch": {
        "prompt_suffix": ", pencil sketch, line drawing, cross hatching, hand drawn",
        "negative_suffix": ", full color, digital painting, 3d render",
    },
    "35mm Film": {
        "prompt_suffix": ", 35mm film photo, natural grain, subtle color, filmic look",
        "negative_suffix": ", overly sharp, cgi, hdr, oversaturated",
    },
    "Rotoscoping": {
        "prompt_suffix": ", rotoscoped style, semi realistic cel shading, outlined shapes, animation frame",
        "negative_suffix": ", flat cartoon, 3d render, low detail, stick figure",
    },
    "R Rated": {
        "prompt_suffix": ", gritty, mature tone, realistic lighting",
        "negative_suffix": ", childish, cartoon, toy-like",
    },
}

ASPECT_PRESETS = {
    "Keep sliders": None,
    "Square 1:1 (1024x1024)": (1024, 1024),
    "Portrait 3:4 (832x1216)": (832, 1216),
    "Landscape 4:3 (1216x832)": (1216, 832),
    "Widescreen 16:9 (1152x648)": (1152, 648),
    "Vertical 9:16 (648x1152)": (648, 1152),
    "Match reference image": "match_ref",
}

# -------------------------------------------------------------------------
# Global state: per-device pipelines
# -------------------------------------------------------------------------

pipelines_txt2img = {}  # gpu_id -> pipeline
pipelines_img2img = {}  # gpu_id -> pipeline
current_model_name = None

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def log_event(data):
    """Append a JSON line to the local log file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print("[WARN] Failed to write log:", e)


def apply_style(prompt, negative_prompt, style_name):
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None (raw prompt)"])
    ps = style.get("prompt_suffix", "")
    ns = style.get("negative_suffix", "")

    full_prompt = (prompt or "") + (ps if ps else "")
    neg_base = (negative_prompt or "").strip()
    if ns:
        if neg_base:
            full_negative = neg_base + ns
        else:
            full_negative = ns
    else:
        full_negative = neg_base or None

    return full_prompt.strip(), (full_negative.strip() if full_negative else None)


def round_to_multiple(x, base=64):
    return int(base * round(float(x) / base))


def compute_size_from_ref(ref_image, max_dim=1024, min_dim=512):
    """Compute width/height from a reference image (preserve aspect, round to 64)."""
    if ref_image is None:
        return None

    w, h = ref_image.size
    if w <= 0 or h <= 0:
        return None

    scale = float(max_dim) / float(max(w, h))
    new_w = max(min_dim, min(max_dim, int(w * scale)))
    new_h = max(min_dim, min(max_dim, int(h * scale)))

    new_w = round_to_multiple(new_w, 64)
    new_h = round_to_multiple(new_h, 64)
    return new_w, new_h


def get_effective_size(aspect_preset, slider_w, slider_h, ref_image):
    if aspect_preset == "Match reference image" and ref_image is not None:
        ref_size = compute_size_from_ref(ref_image)
        if ref_size is not None:
            return ref_size

    preset_val = ASPECT_PRESETS.get(aspect_preset)
    if isinstance(preset_val, tuple):
        return preset_val

    return int(slider_w), int(slider_h)


def clamp_for_heavy_model(model_name, width, height, batch_size):
    """Clamp resolution and batch size for heavy models to reduce OOM chance."""
    cfg = MODEL_CONFIGS.get(model_name, {})
    if not cfg.get("heavy", False):
        return width, height, batch_size

    max_pixels = 1024 * 640
    pixels = width * height
    if pixels > max_pixels:
        scale = (max_pixels / float(pixels)) ** 0.5
        width = round_to_multiple(width * scale, 64)
        height = round_to_multiple(height * scale, 64)
        print("Heavy model: clamped resolution to {}x{}".format(width, height))

    if batch_size > 4:
        batch_size = 4
        print("Heavy model: clamped batch size to 4")

    return width, height, batch_size


def set_scheduler(pipe, scheduler_name):
    if pipe is None:
        return
    scheduler_class = SCHEDULER_CLASSES.get(scheduler_name)
    if scheduler_class is None:
        return
    try:
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    except Exception as e:
        print("[WARN] Failed to set scheduler {}: {}".format(scheduler_name, e))


def enable_memory_saving(pipe):
    if pipe is None:
        return
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass


# -------------------------------------------------------------------------
# Model loading (multi-GPU)
# -------------------------------------------------------------------------


def load_pipelines(model_name):
    """
    Load txt2img and img2img pipelines for the given model on multiple GPUs.
    For heavy models, restrict to fewer devices.
    Returns (ok: bool, message: str)
    """
    global pipelines_txt2img, pipelines_img2img, current_model_name

    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["repo_id"]
    heavy = cfg.get("heavy", False)
    max_devices = cfg.get("max_devices")
    if max_devices is None:
        max_devices = len(DEVICE_IDS)

    target_device_ids = DEVICE_IDS[:max_devices]
    if not target_device_ids:
        return False, "No CUDA devices visible."

    print("\n=== Loading model: {} ({}) , offline={} ===".format(
        model_name, model_id, HF_OFFLINE
    ))
    print("[INFO] Multi-GPU: {} devices will be used for batches.".format(len(target_device_ids)))

    start_time = time.time()
    new_txt = {}
    new_img = {}

    try:
        for gpu_id in target_device_ids:
            device = "cuda:{}".format(gpu_id)

            print("  -> Loading txt2img on {}".format(device))
            pipe_txt = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=HF_OFFLINE,
            )
            enable_memory_saving(pipe_txt)
            pipe_txt.to(device)
            new_txt[gpu_id] = pipe_txt

            print("  -> Loading img2img on {}".format(device))
            try:
                pipe_img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=HF_OFFLINE,
                )
                enable_memory_saving(pipe_img)
                pipe_img.to(device)
                new_img[gpu_id] = pipe_img
            except Exception as e:
                print("  [WARN] Could not load img2img on {}: {}".format(device, e))
                new_img[gpu_id] = None

    except Exception as e:
        print("[ERROR] Failed loading model {}: {}".format(model_id, e))
        for p in list(new_txt.values()) + list(new_img.values()):
            try:
                del p
            except Exception:
                pass
        torch.cuda.empty_cache()
        offline_note = ""
        if HF_OFFLINE:
            offline_note = " (HF_HUB_OFFLINE=1, so only cached models can be used)"
        return False, "Failed to load model {}: {}{}".format(model_name, e, offline_note)

    # Free old pipelines
    for p in list(pipelines_txt2img.values()) + list(pipelines_img2img.values()):
        try:
            del p
        except Exception:
            pass
    torch.cuda.empty_cache()

    pipelines_txt2img = new_txt
    pipelines_img2img = new_img
    current_model_name = model_name

    elapsed = time.time() - start_time
    msg = "Loaded model {} on {} device(s) in {:.1f} seconds.".format(
        model_name, len(target_device_ids), elapsed
    )
    print(msg)
    log_event({
        "ts": datetime.utcnow().isoformat(),
        "event": "model_loaded",
        "model_name": model_name,
        "repo_id": model_id,
        "devices": target_device_ids,
        "elapsed_sec": elapsed,
        "heavy": heavy,
    })
    return True, msg


# -------------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------------


def generate_image(
    prompt,
    ref_image,
    strength,
    guidance,
    steps,
    negative_prompt,
    seed,
    width,
    height,
    batch_size,
    scheduler_name,
    style_name,
    aspect_preset,
):
    """
    Text-to-image or Img2Img, multi-GPU batch splitting.
    Returns a list of PIL images for the Gradio Gallery.
    """
    global pipelines_txt2img, pipelines_img2img, current_model_name

    if current_model_name is None or not pipelines_txt2img:
        print("No model loaded yet.")
        return None

    prompt = (prompt or "").strip()
    if not prompt:
        return None

    negative_prompt = (negative_prompt or "").strip()

    # Apply style preset (also updates negative)
    styled_prompt, styled_negative = apply_style(prompt, negative_prompt, style_name)

    # Resolve size
    effective_width, effective_height = get_effective_size(
        aspect_preset, width, height, ref_image
    )

    # Clamp for heavy models
    batch_size = int(batch_size) if batch_size else 1
    batch_size = max(1, batch_size)
    effective_width, effective_height, batch_size = clamp_for_heavy_model(
        current_model_name, effective_width, effective_height, batch_size
    )

    # Decide mode and pipeline map
    has_ref = ref_image is not None
    if has_ref:
        # Img2Img mode
        active_items = [(gpu_id, pipe) for gpu_id, pipe in pipelines_img2img.items() if pipe is not None]
        mode = "img2img"
    else:
        active_items = list(pipelines_txt2img.items())
        mode = "text2img"

    if not active_items:
        print("No suitable pipeline available for this model/mode.")
        return None

    # Set scheduler on each active pipeline
    for _, pipe in active_items:
        set_scheduler(pipe, scheduler_name)

    print("\n=== New generation request ===")
    print("Model: {}".format(current_model_name))
    print("Mode: {}".format(mode))
    print("Style: {}".format(style_name))
    print("Scheduler: {}".format(scheduler_name))
    print("Prompt: {!r}".format(styled_prompt))
    print("Negative prompt: {!r}".format(styled_negative))
    print("Seed: {}, Batch size: {}".format(seed, batch_size))
    print("Size: {}x{}".format(effective_width, effective_height))
    print("Guidance: {}, Steps: {}".format(guidance, steps))

    # Base seed (if given)
    base_seed = int(seed) if seed and int(seed) > 0 else None

    # Split batch across devices
    num_devices = len(active_items)
    base = batch_size // num_devices
    rem = batch_size % num_devices

    all_images = []
    saved_files = []

    for idx, (gpu_id, pipe) in enumerate(active_items):
        local_batch = base + (1 if idx < rem else 0)
        if local_batch <= 0:
            continue

        device = "cuda:{}".format(gpu_id)
        if base_seed is not None:
            local_seed = base_seed + idx
            generator = torch.Generator(device=device).manual_seed(local_seed)
        else:
            generator = torch.Generator(device=device)

        prompts = [styled_prompt] * local_batch
        negs = [styled_negative] * local_batch if styled_negative else None

        with torch.no_grad():
            if mode == "text2img":
                result = pipe(
                    prompt=prompts,
                    negative_prompt=negs,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(effective_width),
                    height=int(effective_height),
                    generator=generator,
                    num_images_per_prompt=1,
                )
            else:
                imgs = [ref_image] * local_batch
                result = pipe(
                    prompt=prompts,
                    negative_prompt=negs,
                    image=imgs,
                    strength=float(strength),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(effective_width),
                    height=int(effective_height),
                    generator=generator,
                    num_images_per_prompt=1,
                )

        images = result.images
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        for j, img in enumerate(images):
            safe_model = current_model_name.replace(" ", "_").replace("/", "_")
            filename = "img_{}_{}_gpu{}_{}.png".format(
                ts_str, safe_model, gpu_id, j
            )
            path = os.path.join(OUTPUT_DIR, filename)
            try:
                img.save(path)
                saved_files.append(path)
            except Exception as e:
                print("[WARN] Failed to save image {}: {}".format(path, e))
            all_images.append(img)

    # Log the job
    log_event({
        "ts": datetime.utcnow().isoformat(),
        "event": "generation",
        "model_name": current_model_name,
        "mode": mode,
        "style": style_name,
        "scheduler": scheduler_name,
        "prompt": prompt,
        "styled_prompt": styled_prompt,
        "negative_prompt": negative_prompt,
        "styled_negative": styled_negative,
        "seed": base_seed,
        "batch_size": batch_size,
        "width": effective_width,
        "height": effective_height,
        "guidance": guidance,
        "steps": steps,
        "saved_files": saved_files,
    })

    if not all_images:
        return None

    print("Saved {} image(s) for this job.".format(len(saved_files)))
    return all_images


# -------------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------------


def on_model_change(model_name):
    ok, msg = load_pipelines(model_name)
    # Enable Generate button only if load succeeded
    return msg, gr.update(interactive=ok)


def build_ui():
    with gr.Blocks(title="DGX SDXL Multi-GPU Image Lab") as demo:
        gr.Markdown(
            """
            # DGX SDXL Multi-GPU Image Lab

            - Multiple SDXL models (Base, Turbo, RealVis, CyberRealistic, Juggernaut, Animagine)
            - Text-to-image and Img2Img
            - Style presets, schedulers, negative prompts, batch size
            - Multi-GPU batch splitting
            - All images auto-saved to ./output_images
            """
        )

        with gr.Row():
            # Column 1: model and scheduler
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="SDXL Base 1.0 (StabilityAI)",
                    label="Model",
                    info="Choose which SDXL checkpoint to use.",
                )
                scheduler_dropdown = gr.Dropdown(
                    choices=list(SCHEDULER_CLASSES.keys()),
                    value="Default (from model)",
                    label="Scheduler / Sampler",
                    info="Try UniPC or DPM++ 2M for quality / speed tradeoff.",
                )
                style_dropdown = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="Photoreal",
                    label="Style preset",
                )
                model_status = gr.Markdown(
                    "No model loaded yet. Select a model above."
                )

            # Column 2: negative prompt, seed, aspect
            with gr.Column(scale=1):
                negative_prompt_in = gr.Textbox(
                    lines=3,
                    label="Negative prompt",
                    placeholder="Things you DO NOT want (e.g. ugly, blurry, extra fingers)...",
                )
                seed_in = gr.Number(
                    value=0,
                    label="Seed (0 = random each run)",
                    precision=0,
                )
                aspect_dropdown = gr.Dropdown(
                    choices=list(ASPECT_PRESETS.keys()),
                    value="Widescreen 16:9 (1152x648)",
                    label="Aspect ratio preset",
                )

            # Column 3: width/height, batch size
            with gr.Column(scale=1):
                width_in = gr.Slider(
                    minimum=512,
                    maximum=1280,
                    value=1152,
                    step=64,
                    label="Width",
                )
                height_in = gr.Slider(
                    minimum=512,
                    maximum=1280,
                    value=648,
                    step=64,
                    label="Height",
                )
                batch_size_in = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="Batch size (number of images)",
                )

        # Update sliders when aspect preset changes (except Keep sliders / Match ref)
        def on_aspect_change(preset, w, h):
            preset_val = ASPECT_PRESETS.get(preset)
            if isinstance(preset_val, tuple):
                return gr.update(value=preset_val[0]), gr.update(value=preset_val[1])
            return gr.update(value=w), gr.update(value=h)

        aspect_dropdown.change(
            fn=on_aspect_change,
            inputs=[aspect_dropdown, width_in, height_in],
            outputs=[width_in, height_in],
        )

        # Main generation tab
        with gr.Tab("Text / Img2Img"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_in = gr.Textbox(
                        lines=3,
                        label="Prompt",
                        placeholder="Describe the image you want...",
                    )
                    ref_image_in = gr.Image(
                        label="Optional reference image (Img2Img)",
                        type="pil",
                    )
                    strength_in = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Img2Img strength (how much to change the reference image)",
                    )
                    guidance_in = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance scale",
                    )
                    steps_in = gr.Slider(
                        minimum=5,
                        maximum=80,
                        value=30,
                        step=1,
                        label="Number of inference steps",
                    )
                    generate_btn = gr.Button(
                        "Generate",
                        variant="primary",
                        interactive=False,
                    )
                with gr.Column(scale=3):
                    output_gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=True,
                        columns=4,
                        height="auto",
                    )

            generate_btn.click(
                fn=generate_image,
                inputs=[
                    prompt_in,
                    ref_image_in,
                    strength_in,
                    guidance_in,
                    steps_in,
                    negative_prompt_in,
                    seed_in,
                    width_in,
                    height_in,
                    batch_size_in,
                    scheduler_dropdown,
                    style_dropdown,
                    aspect_dropdown,
                ],
                outputs=[output_gallery],
            )

        # Model load hook
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[model_status, generate_btn],
        )

        return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7867)

