import os
import time
import torch
import numpy as np

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import gradio as gr
from PIL import Image as PilImage

# Disable Gradio analytics to avoid pandas/NDFrame issues in some environments
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "RealVis XL 5.0 (Photoreal)": {
        "repo_id": "SG161222/RealVisXL_V5.0",
        "kind": "sdxl",
        "is_turbo": False,
    },
    "DreamShaper XL Lightning": {
        "repo_id": "Lykon/dreamshaper-xl-lightning",
        "kind": "sdxl",
        "is_turbo": True,
    },
    "ZavyChroma XL v1.0": {
        "repo_id": "misri/zavychromaxl_v100",
        "kind": "sdxl",
        "is_turbo": False,
    },
}

# Models that tend to be heavier on VRAM
HEAVY_MODELS = {
    "CyberRealistic XL v5.8 (Photoreal)",
    "Juggernaut XL (Photoreal)",
    "RealVis XL 5.0 (Photoreal)",
    "ZavyChroma XL v1.0",
}

# Available schedulers
SCHEDULER_CLASSES = {
    "Default": None,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M (DPMSolverMultistep)": DPMSolverMultistepScheduler,
}

# Style presets for prompt / negative prompt
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
}

# Aspect ratio presets (default = Widescreen)
ASPECT_PRESETS = {
    "Keep sliders": None,
    "Square 1:1 - 1024x1024": (1024, 1024),
    "Portrait 3:4 - 832x1216": (832, 1216),
    "Landscape 4:3 - 1216x832": (1216, 832),
    "Widescreen 16:9 - 1152x648 (default)": (1152, 648),
    "Low-res 16:9 - 768x432": (768, 432),
    "Vertical 9:16 - 648x1152": (648, 1152),
    "Match reference image": "match_ref",
}

DEFAULT_ASPECT_KEY = "Widescreen 16:9 - 1152x648 (default)"

# Globals for pipelines
pipe_txt2img = None
pipe_img2img = None
current_model_name = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_scheduler(pipe, scheduler_name: str):
    """Replace the scheduler on a pipeline based on the dropdown choice."""
    if pipe is None:
        return
    scheduler_class = SCHEDULER_CLASSES.get(scheduler_name)
    if scheduler_class is None:
        return
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)


def _enable_memory_saving_features(pipe):
    """Enable common VRAM-saving options on a pipeline."""
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


def apply_style(prompt: str, negative_prompt: str, style_name: str):
    """Apply style preset to prompt and negative prompt."""
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["None (raw prompt)"])
    ps = style.get("prompt_suffix", "")
    ns = style.get("negative_suffix", "")

    full_prompt = prompt + ps if ps else prompt
    if negative_prompt:
        full_negative = negative_prompt + ns
    else:
        full_negative = ns if ns else None

    return full_prompt, full_negative


def round_to_multiple(x, base=64):
    return int(base * round(float(x) / base))


def compute_size_from_ref(ref_image, max_dim=1024, min_dim=512):
    """
    Compute width/height from a reference image:
    - preserve aspect ratio
    - scale so the larger dim is ~max_dim
    - clamp to [min_dim, max_dim]
    - round both dims to multiples of 64
    """
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
    """
    Resolve final (width, height) from:
    - aspect preset dropdown
    - sliders
    - optional reference image
    """
    if aspect_preset == "Match reference image" and ref_image is not None:
        ref_size = compute_size_from_ref(ref_image)
        if ref_size is not None:
            return ref_size

    preset_val = ASPECT_PRESETS.get(aspect_preset, None)
    if isinstance(preset_val, tuple):
        return preset_val

    return int(slider_w), int(slider_h)


def clamp_for_heavy_models(width, height):
    """
    For very heavy SDXL finetunes, clamp resolution a bit
    to reduce chances of CUDA OOM, but keep the requested batch_size.
    """
    max_pixels = 1024 * 640  # safe-ish upper bound
    pixels = width * height

    if pixels > max_pixels:
        scale = (max_pixels / float(pixels)) ** 0.5
        width = round_to_multiple(width * scale, 64)
        height = round_to_multiple(height * scale, 64)
        print(f"Heavy model: clamped resolution to {width}x{height}")

    return width, height


def make_filename_base(prompt: str) -> str:
    """
    Build a readable filename base from timestamp + prompt slug.
    Example: 20241224_213045_dark_room_ultra_realistic
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt = (prompt or "").strip().lower()

    cleaned_chars = []
    for ch in prompt:
        if ch.isalnum() or ch.isspace():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = "".join(cleaned_chars)

    words = [w for w in cleaned.split() if w]
    slug = "_".join(words[:6]) if words else "image"

    if len(slug) > 60:
        slug = slug[:60]

    return f"{timestamp}_{slug}"

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
    """
    Text-to-image or Img2Img, depending on whether ref_image is provided.
    Returns a list of images (for Gradio Gallery).

    All generated images are also saved to OUTPUT_DIR with filenames based on
    timestamp + prompt slug, for example:
    20241224_213045_dark_room_ultra_realistic_01.png
    """
    global pipe_txt2img, pipe_img2img, current_model_name

    try:
        print("\n>>> generate_image() called")

        if current_model_name is None:
            print("No model loaded yet.")
            return None

        if pipe_txt2img is None and ref_image is None:
            print("No Text2Img pipeline available.")
            return None
        if pipe_img2img is None and ref_image is not None:
            print("No Img2Img pipeline available for this model.")
            return None

        prompt = (prompt or "").strip()
        if not prompt:
            print("Empty prompt, aborting.")
            return None

        neg_prompt = (negative_prompt or "").strip() or None

        # Apply style preset
        styled_prompt, styled_negative = apply_style(prompt, neg_prompt, style_name)

        # Resolve final width/height (including ref-image-aware mode)
        effective_width, effective_height = get_effective_size(
            aspect_preset, width, height, ref_image
        )

        # Clamp resolution for heavy models
        if current_model_name in HEAVY_MODELS:
            effective_width, effective_height = clamp_for_heavy_models(
                effective_width, effective_height
            )

        # Ensure batch size is valid
        batch_size = max(1, int(batch_size))

        print(f"Model: {current_model_name}")
        print(f"Style: {style_name}")
        print(f"Scheduler: {scheduler_name}")
        print(f"Prompt: {styled_prompt!r}")
        print(f"Negative prompt: {styled_negative!r}")
        print(f"Seed: {seed}, Batch size: {batch_size}")
        print(f"Size: {effective_width}x{effective_height}")
        print(f"Guidance: {guidance}, Steps: {steps}")
        print("Mode:", "img2img" if ref_image is not None else "text2img")

        # Seed
        generator = None
        if seed and seed > 0:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Set scheduler
        if ref_image is None:
            set_scheduler(pipe_txt2img, scheduler_name)
            pipe = pipe_txt2img
        else:
            set_scheduler(pipe_img2img, scheduler_name)
            pipe = pipe_img2img

        with torch.no_grad():
            if ref_image is None:
                # Text-to-image
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
                    num_images_per_prompt=1,  # because we replicate prompts
                )
            else:
                # Image-to-image
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

        images = result.images

        if not images:
            print("Pipeline returned no images.")
            return None

        # Save ALL images with timestamp+prompt-based names
        base_name = make_filename_base(prompt)
        for idx, img in enumerate(images, start=1):
            filename = f"{base_name}_{idx:02d}.png"
            path = os.path.join(OUTPUT_DIR, filename)
            img.save(path)
            print(f"Saved image {idx} to {path}")

        return images

    except Exception as e:
        import traceback
        print("!!! ERROR in generate_image !!!")
        print(repr(e))
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# Model loading with progress (optimized: load once & reuse components)
# -----------------------------------------------------------------------------

def on_model_change(model_name):
    """
    Generator function so we can:
    1) Immediately show a loading message + disable button
    2) Then run loading steps with progress messages
    3) Finally enable button if load succeeded
    """
    global pipe_txt2img, pipe_img2img, current_model_name

    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["repo_id"]

    start_time = time.time()

    # Step 1: show "starting" state
    status = (
        f"Loading model {model_name}...\n"
        f"- Repo: {model_id}\n"
        f"- Offline mode: {HF_OFFLINE}\n\n"
        "Step 1/3: starting..."
    )
    yield (
        status,
        gr.update(interactive=False),  # Generate button
    )

    try:
        # Step 2: load base pipeline
        status = (
            f"Loading model {model_name}...\n"
            f"- Repo: {model_id}\n"
            f"- Offline mode: {HF_OFFLINE}\n\n"
            "Step 2/3: loading base SDXL pipeline from local cache..."
        )
        yield (
            status,
            gr.update(interactive=False),
        )

        base_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=HF_OFFLINE,
        )
        _enable_memory_saving_features(base_pipe)
        base_pipe.to(DEVICE)

        # Step 3: build Img2Img from components
        status = (
            f"Loading model {model_name}...\n"
            f"- Repo: {model_id}\n"
            f"- Offline mode: {HF_OFFLINE}\n\n"
            "Step 3/3: building Img2Img pipeline from components..."
        )
        yield (
            status,
            gr.update(interactive=False),
        )

        try:
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**base_pipe.components)
            _enable_memory_saving_features(img2img_pipe)
            img2img_pipe.to(DEVICE)
        except Exception as e:
            print(f"WARNING: Could not build Img2Img pipeline for {model_id} from components: {e}")
            img2img_pipe = None

        # Loading succeeded: swap in new pipelines
        for p in [pipe_txt2img, pipe_img2img]:
            try:
                del p
            except Exception:
                pass
        torch.cuda.empty_cache()

        pipe_txt2img = base_pipe
        pipe_img2img = img2img_pipe
        current_model_name = model_name

        elapsed = time.time() - start_time
        final_status = (
            f"Loaded model {model_name} in {elapsed:.1f} seconds.\n\n"
            f"- Repo: {model_id}\n"
            f"- Offline mode: {HF_OFFLINE}\n"
            f"- Pipelines: txt2img OK, img2img {'OK' if img2img_pipe is not None else 'NOT AVAILABLE'}\n\n"
            "All generated images will be saved under ./output_images "
            "with timestamp + prompt-based filenames."
        )

        yield (
            final_status,
            gr.update(interactive=True),
        )

    except Exception as e:
        import traceback
        print(f"ERROR loading model {model_name}: {e}")
        traceback.print_exc()
        offline_note = " (HF_HUB_OFFLINE=1, so no network access)" if HF_OFFLINE else ""
        msg = f"Failed to load model {model_name}{offline_note}: {e}"

        any_loaded = pipe_txt2img is not None or pipe_img2img is not None
        if current_model_name:
            msg += f"\nStill using previously loaded model: {current_model_name}."
        else:
            msg += "\nNo model is currently loaded."

        yield (
            msg,
            gr.update(interactive=any_loaded),
        )

# -----------------------------------------------------------------------------
# Aspect preset -> update width / height sliders
# -----------------------------------------------------------------------------

def on_aspect_change(aspect_name, current_width, current_height):
    """
    When user changes aspect preset, update width/height sliders to match,
    except for "Keep sliders" and "Match reference image".
    """
    preset_val = ASPECT_PRESETS.get(aspect_name)

    if isinstance(preset_val, tuple):
        w, h = preset_val
        return gr.update(value=w), gr.update(value=h)

    # For "Keep sliders" or "Match reference image" just keep current values
    return gr.update(value=current_width), gr.update(value=current_height)

# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

print("Starting DGX SDXL Image Lab (no model loaded at startup).")
initial_model_name = list(MODEL_CONFIGS.keys())[0]
initial_status = (
    "No model loaded yet.\n\n"
    "Select a model from the dropdown above to start. "
    "The Generate & Save button will enable once the model is ready.\n\n"
    "All generated images are automatically saved in ./output_images."
)

with gr.Blocks(title="DGX SDXL Image Lab v8") as demo:
    gr.Markdown(
        """
        # DGX SDXL Image Lab v8

        - Multiple SDXL models: Base, Turbo, CyberRealistic, Juggernaut, RealVis, DreamShaper, ZavyChroma
        - Features:
          - Text-to-image
          - Img2Img with strength
          - Style presets (Photoreal, Cinematic, Anime...)
          - Negative prompts
          - Seed (0 = random)
          - Aspect ratio presets (Widescreen default, low-res 16:9, etc.)
          - Width / Height sliders, synced with aspect presets
          - Batch size (variations)
        - All generated images are saved automatically to ./output_images
          with filenames based on timestamp + prompt (for example: 20241224_213045_dark_room_01.png).
        """
    )

    # Global controls (apply to all generations)
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=initial_model_name,
                label="Model",
            )
            scheduler_dropdown = gr.Dropdown(
                choices=list(SCHEDULER_CLASSES.keys()),
                value="Default",
                label="Scheduler / Sampler",
            )
            style_dropdown = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value="Photoreal",
                label="Style preset",
            )
            model_status = gr.Markdown(initial_status)
        with gr.Column(scale=1):
            negative_prompt_in = gr.Textbox(
                lines=2,
                label="Negative prompt",
                placeholder="Things you DON'T want (for example: ugly, blurry, extra fingers, cartoon)...",
            )
            seed_in = gr.Number(
                value=0,
                label="Seed (0 = random)",
                precision=0,
            )
        with gr.Column(scale=1):
            aspect_dropdown = gr.Dropdown(
                choices=list(ASPECT_PRESETS.keys()),
                value=DEFAULT_ASPECT_KEY,
                label="Aspect ratio preset",
            )
            width_in = gr.Slider(
                minimum=512,
                maximum=1280,
                value=ASPECT_PRESETS[DEFAULT_ASPECT_KEY][0],
                step=64,
                label="Width",
            )
            height_in = gr.Slider(
                minimum=384,
                maximum=1280,
                value=ASPECT_PRESETS[DEFAULT_ASPECT_KEY][1],
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

    # Aspect preset -> update sliders
    aspect_dropdown.change(
        fn=on_aspect_change,
        inputs=[aspect_dropdown, width_in, height_in],
        outputs=[width_in, height_in],
    )

    # ------------------------
    # Text / Img2Img
    # ------------------------
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
                "Generate & Save",
                variant="primary",
                interactive=False,  # disabled until a model is loaded
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

    # Bind model change AFTER generate_btn exists so we can enable/disable it
    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[model_status, generate_btn],
    )

# Queue is needed for generator-based UI updates (loading state)
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7867)

