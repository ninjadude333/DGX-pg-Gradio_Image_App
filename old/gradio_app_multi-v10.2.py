import os
import time
import json
import torch
import numpy as np

# Disable Gradio analytics BEFORE importing gradio to avoid pandas issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
import gradio as gr
from PIL import Image as PilImage

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUTPUT_DIR = "./output_images"
LOG_DIR = "./logs"
LOG_PATH = os.path.join(LOG_DIR, "image_jobs.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = "cuda"

HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

# Human-friendly model names -> HF repo IDs
MODEL_CONFIGS = {
    "SDXL Base 1.0 (StabilityAI)": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "SDXL Turbo (Fast preview)": {
        "repo_id": "stabilityai/sdxl-turbo",
        "kind": "sdxl",
        "is_turbo": True,
    },
    "RealVis XL 5.0 (Photoreal)": {
        "repo_id": "SG161222/RealVisXL_V5.0",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "CyberRealistic XL v5.8 (Photoreal)": {
        "repo_id": "John6666/cyberrealistic-xl-v58-sdxl",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "Juggernaut XL (Photoreal)": {
        "repo_id": "glides/juggernautxl",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "Animagine XL 4.0 (Anime)": {
        "repo_id": "cagliostrolab/animagine-xl-4.0",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "Portrait Realistic SDXL": {
        "repo_id": "stablediffusionapi/portrait-realistic-sdxl",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "Clarity SDXL (Cinematic)": {
        "repo_id": "nDimensional/Clarity-SDXL",
        "kind": "sdxl",
        "is_turbo": False,
    },
}

HEAVY_MODELS = {
    "RealVis XL 5.0 (Photoreal)",
    "CyberRealistic XL v5.8 (Photoreal)",
    "Juggernaut XL (Photoreal)",
    "Animagine XL 4.0 (Anime)",
    "Portrait Realistic SDXL",
    "Clarity SDXL (Cinematic)",
}

SCHEDULER_CLASSES = {
    "Default (as loaded)": None,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M (DPMSolverMultistep)": DPMSolverMultistepScheduler,
    "UniPC (fast / modern)": UniPCMultistepScheduler,
}

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
        "negative_suffix": ", cartoon, anime, flat lighting, low detail",
    },
    "Soft illustration": {
        "prompt_suffix": ", soft illustration, gentle colors, subtle shading, artstation",
        "negative_suffix": ", harsh lighting, heavy contrast, photoreal skin pores",
    },
    "Anime / vibrant": {
        "prompt_suffix": ", anime style, vibrant colors, clean line art",
        "negative_suffix": ", photo, realistic skin, gritty, grainy",
    },
    "R Rated": {
        "prompt_suffix": ", gritty, realistic lighting, film grain, moody atmosphere",
        "negative_suffix": ", cartoon, anime, flat lighting, childish, low detail",
    },
    "Pencil Sketch": {
        "prompt_suffix": ", pencil sketch, monochrome drawing, cross hatching, hand drawn lines",
        "negative_suffix": ", full color, 3d render, glossy, photorealistic skin, digital painting",
    },
    "B&W": {
        "prompt_suffix": ", black and white, high contrast, monochrome, dramatic shadows",
        "negative_suffix": ", colorful, oversaturated, pastel colors",
    },
    "35mm Film": {
        "prompt_suffix": ", 35mm film photograph, film grain, natural colors, soft focus, analog look",
        "negative_suffix": ", cgi, hdr, oversharp, digital noise",
    },
    "Rotoscoping": {
        "prompt_suffix": ", rotoscoped style, semi realistic cel shading, outlined shapes, animation frame",
        "negative_suffix": ", flat cartoon, 3d render, low detail, stick figure",
    },
}

ASPECT_PRESETS = {
    "Keep sliders": None,
    "Square 1:1 - 1024x1024": (1024, 1024),
    "Portrait 3:4 - 832x1216": (832, 1216),
    "Landscape 4:3 - 1216x832": (1216, 832),
    "Widescreen 16:9 - 1152x648": (1152, 648),
    "Small Widescreen 16:9 - 768x432": (768, 432),
    "Vertical 9:16 - 648x1152": (648, 1152),
    "Match reference image": "match_ref",
}

pipe_txt2img = None
pipe_img2img = None
current_model_name = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_scheduler(pipe, scheduler_name: str):
    if pipe is None:
        return
    scheduler_class = SCHEDULER_CLASSES.get(scheduler_name)
    if scheduler_class is None:
        return
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)


def _enable_memory_saving_features(pipe):
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


def load_pipelines(model_name: str):
    global pipe_txt2img, pipe_img2img, current_model_name

    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["repo_id"]

    print("\n=== Loading model: {} ({}), offline={} ===".format(
        model_name, model_id, HF_OFFLINE
    ))

    try:
        new_txt2img = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=HF_OFFLINE,
        )
        _enable_memory_saving_features(new_txt2img)
        new_txt2img.to(DEVICE)

        try:
            new_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=HF_OFFLINE,
            )
            _enable_memory_saving_features(new_img2img)
            new_img2img.to(DEVICE)
        except Exception as e:
            print("WARNING: Could not load Img2Img pipeline for {}: {}".format(model_id, e))
            new_img2img = None

    except Exception as e:
        print("ERROR loading model {}: {}".format(model_id, e))
        offline_note = " (HF_HUB_OFFLINE=1, so no network access)" if HF_OFFLINE else ""
        return False, "Failed to load model {}{}: {}".format(model_name, offline_note, e)

    for p in [pipe_txt2img, pipe_img2img]:
        try:
            del p
        except Exception:
            pass
    torch.cuda.empty_cache()

    pipe_txt2img = new_txt2img
    pipe_img2img = new_img2img
    current_model_name = model_name

    print("Pipelines loaded.")
    return True, "Loaded model: {}".format(model_name)


def apply_style(prompt: str, negative_prompt: str, style_name: str):
    """
    Apply style preset to prompt and (optionally) negative prompt.

    - Prompt: always append the style's prompt_suffix.
    - Negative prompt: we assume the textbox already contains what the user wants.
      The style preset can auto-fill it via the UI when empty, so we do NOT append
      negative_suffix again here (to avoid double application).
    """
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None (raw prompt)"])
    ps = style.get("prompt_suffix", "")

    prompt = prompt or ""
    negative_prompt = negative_prompt or ""

    full_prompt = prompt + ps if ps else prompt
    full_negative = negative_prompt.strip() or None

    return full_prompt, full_negative


def round_to_multiple(x, base=64):
    return int(base * round(float(x) / base))


def compute_size_from_ref(ref_image, max_dim=1024, min_dim=512):
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


def get_effective_size(aspect_preset: str, slider_w: int, slider_h: int, ref_image):
    if aspect_preset == "Match reference image" and ref_image is not None:
        ref_size = compute_size_from_ref(ref_image)
        if ref_size is not None:
            return ref_size

    preset_val = ASPECT_PRESETS.get(aspect_preset, None)
    if isinstance(preset_val, tuple):
        return preset_val

    return int(slider_w), int(slider_h)


def clamp_for_heavy_models(width, height, batch_size):
    """
    For very heavy SDXL finetunes, clamp resolution a bit
    to reduce chances of CUDA OOM. Do NOT touch batch_size here,
    so the UI batch slider is respected.
    """
    max_pixels = 1024 * 640
    pixels = width * height

    if pixels > max_pixels:
        scale = (max_pixels / float(pixels)) ** 0.5
        width = round_to_multiple(width * scale, 64)
        height = round_to_multiple(height * scale, 64)
        print("Heavy model: clamped resolution to {}x{}".format(width, height))

    return width, height, batch_size


def make_filename_base(prompt: str, model_name: str, mode: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt = (prompt or "").strip().lower()
    model_name = (model_name or "model").strip().lower()
    mode = (mode or "gen").strip().lower()

    def slugify(text):
        chars = []
        for ch in text:
            if ch.isalnum() or ch.isspace():
                chars.append(ch)
            else:
                chars.append(" ")
        cleaned = "".join(chars)
        words = [w for w in cleaned.split() if w]
        if not words:
            return "image"
        slug = "_".join(words[:6])
        if len(slug) > 60:
            slug = slug[:60]
        return slug

    prompt_slug = slugify(prompt)
    model_slug = slugify(model_name)
    mode_slug = slugify(mode)

    base = "{}_{}_{}".format(timestamp, model_slug, mode_slug)
    if prompt_slug:
        base += "_" + prompt_slug
    return base


def log_event(event_type: str, data: dict):
    try:
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_type,
            "data": data,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print("Failed to write log entry: {}".format(e))


# -----------------------------------------------------------------------------
# Generation and saving
# -----------------------------------------------------------------------------

def generate_image(
    prompt: str,
    ref_image,
    strength: float,
    guidance: float,
    steps: int,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    batch_size: int,
    scheduler_name: str,
    style_name: str,
    aspect_preset: str,
):
    global pipe_txt2img, pipe_img2img, current_model_name

    if pipe_txt2img is None and ref_image is None:
        print("No Text2Img pipeline loaded yet.")
        return None
    if pipe_img2img is None and ref_image is not None:
        print("No Img2Img pipeline loaded yet for this model.")
        return None

    prompt = (prompt or "").strip()
    if not prompt:
        return None

    neg_prompt = (negative_prompt or "").strip() or None

    styled_prompt, styled_negative = apply_style(prompt, neg_prompt, style_name)

    effective_width, effective_height = get_effective_size(
        aspect_preset, width, height, ref_image
    )

    batch_size = max(1, int(batch_size))
    if current_model_name in HEAVY_MODELS:
        effective_width, effective_height, batch_size = clamp_for_heavy_models(
            effective_width, effective_height, batch_size
        )

    mode = "img2img" if ref_image is not None else "text2img"

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

    generator = None
    if seed and int(seed) > 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    if ref_image is None:
        set_scheduler(pipe_txt2img, scheduler_name)
        pipe = pipe_txt2img
    else:
        set_scheduler(pipe_img2img, scheduler_name)
        pipe = pipe_img2img

    try:
        with torch.no_grad():
            if ref_image is None:
                result = pipe(
                    prompt=[styled_prompt] * batch_size,
                    negative_prompt=[styled_negative] * batch_size
                    if styled_negative
                    else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=effective_width,
                    height=effective_height,
                    generator=generator,
                    num_images_per_prompt=1,
                )
            else:
                result = pipe(
                    prompt=[styled_prompt] * batch_size,
                    negative_prompt=[styled_negative] * batch_size
                    if styled_negative
                    else None,
                    image=[ref_image] * batch_size,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=effective_width,
                    height=effective_height,
                    generator=generator,
                    num_images_per_prompt=1,
                )
    except Exception as e:
        print("ERROR during generation: {}".format(e))
        log_event("generation_error", {
            "model": current_model_name,
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "styled_prompt": styled_prompt,
            "styled_negative": styled_negative,
            "seed": seed,
            "batch_size": batch_size,
            "width": effective_width,
            "height": effective_height,
            "guidance": guidance,
            "steps": steps,
            "scheduler": scheduler_name,
            "style": style_name,
            "aspect_preset": aspect_preset,
            "error": str(e),
        })
        return None

    images = result.images

    if not images:
        print("Pipeline returned no images.")
        return None

    base_name = make_filename_base(prompt, current_model_name, mode)
    saved_files = []

    for idx, img in enumerate(images, start=1):
        filename = "{}_{:02d}.png".format(base_name, idx)
        path = os.path.join(OUTPUT_DIR, filename)
        img.save(path)
        saved_files.append(path)
        print("Saved image {} to {}".format(idx, path))

    log_event("generation", {
        "model": current_model_name,
        "mode": mode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "styled_prompt": styled_prompt,
        "styled_negative": styled_negative,
        "seed": seed,
        "batch_size": batch_size,
        "width": effective_width,
        "height": effective_height,
        "guidance": guidance,
        "steps": steps,
        "scheduler": scheduler_name,
        "style": style_name,
        "aspect_preset": aspect_preset,
        "has_reference_image": ref_image is not None,
        "saved_files": saved_files,
    })

    return images


def on_gallery_select(evt: gr.SelectData):
    """
    Gallery.select in newer Gradio returns a SelectData where evt.value
    is often a dict like {"image": <PIL.Image>, "index": int}.
    We need to return the actual image for the Image component.
    """
    value = evt.value
    if isinstance(value, dict):
        if "image" in value:
            return value["image"]
        if "value" in value:
            return value["value"]
    return value


def save_selected_image(selected_image, prompt, model_name):
    if selected_image is None:
        return "No image selected to save."

    prompt = prompt or ""
    model_name = model_name or "model"

    def slugify(text):
        chars = []
        for ch in text:
            if ch.isalnum() or ch in (" ", "-", "_"):
                chars.append(ch)
            else:
                chars.append(" ")
        cleaned = "".join(chars)
        words = [w for w in cleaned.split() if w]
        if not words:
            return "image"
        slug = "_".join(words[:6])
        if len(slug) > 60:
            slug = slug[:60]
        return slug

    safe_prompt = slugify(prompt)
    safe_model = slugify(model_name)

    ts = int(time.time())
    filename = "{}_{}_{}.png".format(safe_model, ts, safe_prompt)
    path = os.path.join(OUTPUT_DIR, filename)

    selected_image.save(path)
    print("Saved selected image to {}".format(path))

    log_event("manual_save", {
        "model": model_name,
        "prompt": prompt,
        "saved_file": path,
    })

    return "Saved image to {}".format(path)


# -----------------------------------------------------------------------------
# Gradio UI utilities
# -----------------------------------------------------------------------------

print("Starting DGX SDXL Image Lab v10.2 (loading initial model)...")
initial_model_name = "SDXL Base 1.0 (StabilityAI)"
initial_ok, initial_msg_plain = load_pipelines(initial_model_name)

if initial_ok:
    initial_status_html = (
        "<span style='color: #22c55e;'>"
        "Loaded model {}"
        "</span>".format(initial_msg_plain)
    )
else:
    initial_status_html = (
        "<span style='color: #ef4444;'>"
        "Warning: {}"
        "</span>".format(initial_msg_plain)
    )


def on_model_change(model_name):
    global current_model_name, pipe_txt2img, pipe_img2img

    if model_name == current_model_name and pipe_txt2img is not None:
        msg_html = (
            "<span style='color: #22c55e;'>"
            "Model <b>{}</b> is already loaded. Reusing existing pipelines."
            "</span>".format(model_name)
        )
        yield gr.update(value=msg_html), gr.update(interactive=True)
        return

    loading_html = (
        "<span style='color: #eab308;'>"
        "Loading model <b>{}</b> (offline={}), please wait..."
        "</span>".format(model_name, "yes" if HF_OFFLINE else "no")
    )
    yield gr.update(value=loading_html), gr.update(interactive=False)

    start = time.time()
    ok, msg_plain = load_pipelines(model_name)
    elapsed = time.time() - start

    if ok:
        msg_html = (
            "<span style='color: #22c55e;'>"
            "Loaded model <b>{}</b> in {:.1f} seconds."
            "</span>".format(model_name, elapsed)
        )
        yield gr.update(value=msg_html), gr.update(interactive=True)
    else:
        any_loaded = pipe_txt2img is not None or pipe_img2img is not None
        msg_html = (
            "<span style='color: #ef4444;'>"
            "Failed to load model: {}"
            "</span>".format(msg_plain)
        )
        if current_model_name:
            msg_html += (
                "<br>Still using previously loaded model: <b>{}</b>."
            ).format(current_model_name)
        else:
            msg_html += "<br>No model is currently loaded."
        yield gr.update(value=msg_html), gr.update(interactive=any_loaded)


def on_aspect_change(aspect_preset, current_w, current_h, ref_image):
    if aspect_preset == "Match reference image":
        if ref_image is None:
            return gr.update(value=current_w), gr.update(value=current_h)
        size = compute_size_from_ref(ref_image)
        if size is None:
            return gr.update(value=current_w), gr.update(value=current_h)
        w, h = size
        return gr.update(value=w), gr.update(value=h)

    preset_val = ASPECT_PRESETS.get(aspect_preset, None)
    if isinstance(preset_val, tuple):
        w, h = preset_val
        return gr.update(value=w), gr.update(value=h)

    return gr.update(value=current_w), gr.update(value=current_h)


def on_style_change(style_name, current_negative):
    """
    When the style preset changes:
    - If the negative prompt box is empty, auto-fill it with the style's
      default negative text (cleaned).
    - If the user already typed something, do NOT override it.
    """
    if current_negative is not None and current_negative.strip():
        return gr.update()

    style = STYLE_PRESETS.get(style_name, {})
    neg_suf = style.get("negative_suffix", "") or ""
    neg_suf = neg_suf.lstrip(" ,")

    if not neg_suf:
        return gr.update()

    return gr.update(value=neg_suf)


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

with gr.Blocks(title="DGX SDXL Image Lab v10.2") as demo:
    gr.Markdown(
        """
        # DGX SDXL Image Lab v10.2

        - Text / Img2Img generation with multiple SDXL models
        - All images in each batch are saved to `./output_images`
        - Every generation and manual save is logged to `./logs/image_jobs.log`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=initial_model_name,
                label="Model",
                info=(
                    "Choose base / turbo / anime / portrait / cinematic SDXL model. "
                    "Models are loaded from local HF cache (offline={}).".format(
                        HF_OFFLINE
                    )
                ),
            )
            scheduler_dropdown = gr.Dropdown(
                choices=list(SCHEDULER_CLASSES.keys()),
                value="DPM++ 2M (DPMSolverMultistep)",
                label="Scheduler / Sampler",
                info=(
                    "Default: model's own scheduler. "
                    "Euler: classic sampler. "
                    "DPM++ 2M: high quality. "
                    "UniPC: modern fast sampler."
                ),
            )
            style_dropdown = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value="Photoreal",
                label="Style preset",
            )
            model_status = gr.Markdown(initial_status_html)
        with gr.Column(scale=1):
            negative_prompt_in = gr.Textbox(
                lines=2,
                label="Negative prompt",
                placeholder="Things you DO NOT want (e.g. ugly, blurry, extra fingers, cartoon)...",
            )
            seed_in = gr.Number(
                value=0,
                label="Seed (0 = random each run)",
                precision=0,
            )
        with gr.Column(scale=1):
            aspect_dropdown = gr.Dropdown(
                choices=list(ASPECT_PRESETS.keys()),
                value="Widescreen 16:9 - 1152x648",
                label="Aspect ratio preset",
            )
            width_in = gr.Slider(
                minimum=512,
                maximum=1536,
                value=1152,
                step=64,
                label="Width",
            )
            height_in = gr.Slider(
                minimum=512,
                maximum=1536,
                value=648,
                step=64,
                label="Height",
            )
            batch_size_in = gr.Slider(
                minimum=1,
                maximum=8,
                value=1,
                step=1,
                label="Batch size (number of variations)",
            )

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
                    maximum=15.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance scale (higher = more literal prompt following)",
                )
                steps_in = gr.Slider(
                    minimum=10,
                    maximum=80,
                    value=30,
                    step=2,
                    label="Number of inference steps",
                )
                generate_btn = gr.Button(
                    "Generate",
                    variant="primary",
                    interactive=bool(initial_ok),
                )
            with gr.Column(scale=3):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    columns=4,
                    height="auto",
                    allow_preview=True,
                )
                selected_image = gr.Image(
                    label="Selected image for saving",
                    interactive=False,
                )
                save_btn = gr.Button("Save selected image to disk")
                save_status = gr.Markdown("")

        aspect_dropdown.change(
            fn=on_aspect_change,
            inputs=[aspect_dropdown, width_in, height_in, ref_image_in],
            outputs=[width_in, height_in],
        )

        ref_image_in.change(
            fn=on_aspect_change,
            inputs=[aspect_dropdown, width_in, height_in, ref_image_in],
            outputs=[width_in, height_in],
        )

        style_dropdown.change(
            fn=on_style_change,
            inputs=[style_dropdown, negative_prompt_in],
            outputs=[negative_prompt_in],
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

        output_gallery.select(
            fn=on_gallery_select,
            inputs=None,
            outputs=selected_image,
        )

        save_btn.click(
            fn=save_selected_image,
            inputs=[selected_image, prompt_in, model_dropdown],
            outputs=[save_status],
        )

    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[model_status, generate_btn],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861
)

