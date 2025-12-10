import os
import time
import json
import random
from typing import List, Optional

# Make sure Gradio doesn't try to phone home / use pandas analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import torch
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "jobs.log")

DEVICE = "cuda"

# Respect offline mode if you run the container with HF_HUB_OFFLINE=1
HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

# Only models we know are present in your cache
MODEL_CONFIGS = {
    "SDXL Base 1.0 (StabilityAI)": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "tag": "sdxlbase",
    },
    "SDXL Turbo (Fast preview)": {
        "repo_id": "stabilityai/sdxl-turbo",
        "tag": "sdxlturbo",
    },
    "RealVis XL 5.0 (Photoreal)": {
        "repo_id": "SG161222/RealVisXL_V5.0",
        "tag": "realvis5",
    },
    "CyberRealistic XL v5.8 (Photoreal)": {
        "repo_id": "John6666/cyberrealistic-xl-v58-sdxl",
        "tag": "cyber58",
    },
    "Animagine XL 4.0 (Anime)": {
        "repo_id": "cagliostrolab/animagine-xl-4.0",
        "tag": "animaxl4",
    },
    "Juggernaut XL (stablediffusionapi)": {
        "repo_id": "stablediffusionapi/juggernautxl",
        "tag": "juggerxl",
    },
}

SCHEDULER_CLASSES = {
    "Default": None,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "UniPC": UniPCMultistepScheduler,
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
    # New “fun” styles
    "R Rated": {
        "prompt_suffix": ", dark gritty atmosphere, intense, cinematic violence",
        "negative_suffix": ", censored, child-friendly, cartoonish, goofy",
    },
    "Pencil Sketch": {
        "prompt_suffix": ", pencil sketch, hand-drawn, monochrome, cross-hatching, sketchbook style",
        "negative_suffix": ", full color, digital painting, glossy, photorealistic skin",
    },
    "B&W": {
        "prompt_suffix": ", black and white, high contrast, monochrome photograph",
        "negative_suffix": ", color, oversaturated, neon, rainbow",
    },
    "35mm Film": {
        "prompt_suffix": ", shot on 35mm film, subtle film grain, cinematic composition, slight blur",
        "negative_suffix": ", CGI, digital render, overly sharp CG edges",
    },
    "Rotoscoping": {
        "prompt_suffix": ", rotoscoped style, semi realistic cel shading, outlined shapes, animation frame",
        "negative_suffix": ", flat cartoon, 3d render, low detail, stick figure",
    },
}

ASPECT_PRESETS = {
    "Keep sliders": None,
    "Widescreen 16:9 – 1152x648": (1152, 648),
    "Low-res 16:9 – 960x540": (960, 540),
    "Square 1:1 – 1024x1024": (1024, 1024),
    "Portrait 3:4 – 832x1216": (832, 1216),
    "Landscape 4:3 – 1216x832": (1216, 832),
    "Vertical 9:16 – 648x1152": (648, 1152),
    "Match reference image": "match_ref",
}

# Global pipelines (single GPU)
pipe_txt2img: Optional[DiffusionPipeline] = None
pipe_img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None
current_model_name: Optional[str] = None
current_model_id: Optional[str] = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log_job(record: dict) -> None:
    """Append one JSON line to jobs.log."""
    record = dict(record)
    record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write log: {e}")


def slugify(text: str, max_len: int = 40) -> str:
    text = text.strip().lower()
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        elif ch in (" ", "-", "_"):
            safe.append("-")
    slug = "".join(safe)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "image"


def set_scheduler(pipe, scheduler_name: str):
    if pipe is None:
        return
    scheduler_class = SCHEDULER_CLASSES.get(scheduler_name)
    if scheduler_class is None:
        return
    try:
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    except Exception as e:
        print(f"[WARN] Failed to set scheduler {scheduler_name}: {e}")


def _enable_memory_saving(pipe):
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
        pipe.enable_attention_slicing("auto")
    except Exception:
        pass
    # xFormers if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def apply_style(prompt: str, negative_prompt: Optional[str], style_name: str):
    """Apply style suffixes. Negative default only used if user left box empty."""
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None (raw prompt)"])
    ps = style.get("prompt_suffix", "")
    ns = style.get("negative_suffix", "")

    full_prompt = prompt + ps if ps else prompt

    if negative_prompt and negative_prompt.strip():
        full_negative = negative_prompt.strip()
    else:
        full_negative = ns.lstrip(" ,") if ns else None

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


def ensure_img2img_loaded():
    """Lazy-load Img2Img pipeline for the current model if/when needed."""
    global pipe_img2img, current_model_id
    if pipe_img2img is not None:
        return
    if current_model_id is None:
        return
    print(f"[lazy-load] Loading Img2Img pipeline for {current_model_id} ...")
    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            current_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=HF_OFFLINE,
        )
        _enable_memory_saving(pipe)
        pipe.to(DEVICE)
        pipe_img2img = pipe
        print("[lazy-load] Img2Img ready.")
    except Exception as e:
        print(f"[lazy-load] Failed to load Img2Img: {e}")
        pipe_img2img = None

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_pipelines(model_name: str):
    """Load / switch the txt2img pipeline for a given model."""
    global pipe_txt2img, pipe_img2img, current_model_name, current_model_id

    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["repo_id"]

    if current_model_name == model_name and pipe_txt2img is not None:
        msg = f"<span style='color: limegreen;'>✅ Model already loaded: <b>{model_name}</b></span>"
        print(msg)
        return msg

    print(f"\n=== Loading model: {model_name} ({model_id}), offline={HF_OFFLINE} ===")
    t0 = time.time()

    try:
        new_txt2img = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=HF_OFFLINE,
        )
        _enable_memory_saving(new_txt2img)
        new_txt2img.to(DEVICE)
    except Exception as e:
        offline_note = " (HF_HUB_OFFLINE=1, local cache only)" if HF_OFFLINE else ""
        err = f"<span style='color: red;'>❌ Failed to load <b>{model_name}</b>{offline_note}:<br>{e}</span>"
        print(f"[ERROR] {err}")
        return err

    # Replace old with new
    try:
        del pipe_txt2img
    except Exception:
        pass
    try:
        del pipe_img2img
    except Exception:
        pass
    torch.cuda.empty_cache()

    pipe_txt2img = new_txt2img
    pipe_img2img = None
    current_model_name = model_name
    current_model_id = model_id

    elapsed = time.time() - t0
    msg = (
        f"<span style='color: limegreen;'>✅ Loaded model <b>{model_name}</b> "
        f"in {elapsed:.1f} seconds.</span>"
    )
    print(msg)
    return msg

# -----------------------------------------------------------------------------
# Generation
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

    if pipe_txt2img is None:
        print("[WARN] No model loaded yet. Please load a model first.")
        return []

    prompt = (prompt or "").strip()
    if not prompt:
        return []

    neg_prompt = (negative_prompt or "").strip() or None

    # Style
    styled_prompt, styled_negative = apply_style(prompt, neg_prompt, style_name)

    # Size
    effective_width, effective_height = get_effective_size(
        aspect_preset, width, height, ref_image
    )

    # Batch
    batch_size = max(1, int(batch_size))

    # Seed
    if seed and seed > 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    else:
        base_seed = random.randint(1, 2**31 - 1)
        generator = torch.Generator(device=DEVICE).manual_seed(base_seed)
        seed = base_seed

    mode = "img2img" if ref_image is not None else "text2img"

    print("\n=== New generation request ===")
    print(f"Model: {current_model_name}")
    print(f"Mode: {mode}")
    print(f"Style: {style_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Prompt: {styled_prompt!r}")
    print(f"Negative prompt: {styled_negative!r}")
    print(f"Seed: {seed}, Batch size: {batch_size}")
    print(f"Size: {effective_width}x{effective_height}")
    print(f"Guidance: {guidance}, Steps: {steps}")

    # Pipeline selection
    if mode == "text2img":
        pipe = pipe_txt2img
    else:
        ensure_img2img_loaded()
        if pipe_img2img is None:
            print("[ERROR] Img2Img pipeline is not available.")
            return []
        pipe = pipe_img2img

    set_scheduler(pipe, scheduler_name)

    with torch.no_grad():
        if mode == "text2img":
            result = pipe(
                prompt=[styled_prompt] * batch_size,
                negative_prompt=[styled_negative] * batch_size
                if styled_negative
                else None,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(effective_width),
                height=int(effective_height),
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
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(effective_width),
                height=int(effective_height),
                generator=generator,
                num_images_per_prompt=1,
            )

    images: List[PilImage.Image] = result.images

    # Auto-save each image in the batch
    saved_files: List[str] = []
    cfg = MODEL_CONFIGS.get(current_model_name, {})
    model_tag = cfg.get("tag", "model")
    prompt_slug = slugify(prompt)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    for idx, img in enumerate(images):
        filename = f"{timestamp}_{model_tag}_b{idx}_s{seed}_{prompt_slug}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        try:
            img.save(path)
            saved_files.append(path)
        except Exception as e:
            print(f"[WARN] Failed to save image {idx}: {e}")

    # Log job
    log_job(
        {
            "model": current_model_name,
            "mode": mode,
            "style": style_name,
            "scheduler": scheduler_name,
            "prompt": prompt,
            "styled_prompt": styled_prompt,
            "negative_prompt_input": negative_prompt,
            "negative_prompt_effective": styled_negative,
            "seed": seed,
            "batch_size": batch_size,
            "width": effective_width,
            "height": effective_height,
            "guidance": guidance,
            "steps": steps,
            "saved_files": saved_files,
        }
    )

    if saved_files:
        print(f"Saved {len(saved_files)} image(s) to {OUTPUT_DIR}")
    return images

# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------

def on_style_change(style_name: str, current_negative: str):
    """When style changes, put its negative text in the box if it's empty."""
    style = STYLE_PRESETS.get(style_name, {})
    ns = style.get("negative_suffix", "")
    if not ns:
        return current_negative
    if current_negative and current_negative.strip():
        return current_negative
    return ns.lstrip(" ,")


def on_aspect_change(aspect_name: str, current_w: int, current_h: int):
    preset = ASPECT_PRESETS.get(aspect_name)
    if isinstance(preset, tuple):
        return preset[0], preset[1]
    return current_w, current_h


def on_model_button_click(model_name: str):
    return load_pipelines(model_name)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

print("Starting DGX SDXL Image Lab v11 (single GPU).")
initial_model_name = list(MODEL_CONFIGS.keys())[0]
initial_status = (
    "⚠️ <b>No model loaded yet.</b><br>"
    "Select a model from the dropdown and click <b>Load / switch model</b>."
)

with gr.Blocks(title="DGX SDXL Image Lab v11") as demo:
    gr.Markdown(
        """
        # DGX SDXL Image Lab v11 (single GPU)

        - Multiple SDXL models (base, turbo, photoreal, anime, Juggernaut)
        - Text-to-image and optional Img2Img (reference image)
        - Style presets (Photoreal, Cinematic, Pencil Sketch, B&W, Rotoscoping, etc.)
        - Aspect presets (16:9 widescreen by default, plus others)
        - All batch images auto-saved with timestamped filenames
        - A `jobs.log` file is written in `./output_images` with prompts & settings
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=initial_model_name,
                label="Model",
            )
            scheduler_dropdown = gr.Dropdown(
                choices=list(SCHEDULER_CLASSES.keys()),
                value="DPM++ 2M",
                label="Sampler / Scheduler",
                info="Default = pipeline's own scheduler; others tweak quality/speed.",
            )
            style_dropdown = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value="Cinematic",
                label="Style preset",
            )
            model_load_button = gr.Button("Load / switch model", variant="primary")
            model_status = gr.Markdown(initial_status)

        with gr.Column(scale=1):
            negative_prompt_in = gr.Textbox(
                lines=3,
                label="Negative prompt",
                placeholder="Things you DON'T want (e.g. blurry, extra limbs, text, etc.)",
            )
            seed_in = gr.Number(
                value=0,
                label="Seed (0 = random)",
                precision=0,
            )

        with gr.Column(scale=1):
            aspect_dropdown = gr.Dropdown(
                choices=list(ASPECT_PRESETS.keys()),
                value="Widescreen 16:9 – 1152x648",
                label="Aspect ratio preset",
            )
            width_in = gr.Slider(
                minimum=512,
                maximum=1408,
                value=1152,
                step=64,
                label="Width",
            )
            height_in = gr.Slider(
                minimum=512,
                maximum=1408,
                value=648,
                step=64,
                label="Height",
            )
            batch_size_in = gr.Slider(
                minimum=1,
                maximum=8,
                value=2,
                step=1,
                label="Batch size (number of variations)",
            )

    # Wiring
    model_load_button.click(
        fn=on_model_button_click,
        inputs=[model_dropdown],
        outputs=[model_status],
    )

    style_dropdown.change(
        fn=on_style_change,
        inputs=[style_dropdown, negative_prompt_in],
        outputs=[negative_prompt_in],
    )

    aspect_dropdown.change(
        fn=on_aspect_change,
        inputs=[aspect_dropdown, width_in, height_in],
        outputs=[width_in, height_in],
    )

    with gr.Tab("Text / Img2Img"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(
                    lines=4,
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
                    value=9.0,
                    step=0.5,
                    label="Guidance scale (higher = more literal prompt following)",
                )
                steps_in = gr.Slider(
                    minimum=10,
                    maximum=80,
                    value=36,
                    step=2,
                    label="Number of inference steps",
                )
                generate_btn = gr.Button("Generate", variant="primary")
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

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7867)

