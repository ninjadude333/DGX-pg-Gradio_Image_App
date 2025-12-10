import os
import time
import torch
import numpy as np

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import gradio as gr
from PIL import Image as PilImage

# Disable Gradio analytics to avoid pandas NDFrame error spam
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda"

# If you want to force local-only (no network), run container with:
#   -e HF_HUB_OFFLINE=1
HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

# Human-friendly model names -> HF repo IDs (or local paths if you prefer)
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
        "is_turbo": True,  # designed for few-step lightning use
    },
    "ZavyChroma XL v1.0": {
        "repo_id": "misri/zavychromaxl_v100",
        "kind": "sdxl",
        "is_turbo": False,
    },
}

# Models that tend to be heavier on VRAM – we’ll clamp size/batch a bit
HEAVY_MODELS = {
    "CyberRealistic XL v5.8 (Photoreal)",
    "Juggernaut XL (Photoreal)",
    "RealVis XL 5.0 (Photoreal)",
    "ZavyChroma XL v1.0",
}

# Available schedulers
SCHEDULER_CLASSES = {
    "Default": None,  # use whatever the pipeline comes with
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

# Aspect ratio presets
ASPECT_PRESETS = {
    "Keep sliders": None,
    "Square 1:1 – 1024x1024": (1024, 1024),
    "Portrait 3:4 – 832x1216": (832, 1216),
    "Landscape 4:3 – 1216x832": (1216, 832),
    "Widescreen 16:9 – 1152x648": (1152, 648),
    "Vertical 9:16 – 648x1152": (648, 1152),
    "Match reference image": "match_ref",
}

# Globals for pipelines
pipe_txt2img = None
pipe_img2img = None
pipe_inpaint = None
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
        # "Default" or unknown => keep current scheduler
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


def load_pipelines(model_name: str):
    """
    Load (or reload) SDXL pipelines for a given model.
    Returns (ok: bool, msg: str).
    Respects HF_OFFLINE: if True, uses local_files_only=True.
    """
    global pipe_txt2img, pipe_img2img, pipe_inpaint, current_model_name

    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["repo_id"]

    print(f"\n=== Loading model: {model_name} ({model_id}), offline={HF_OFFLINE} ===")

    try:
        # Try loading into temporary vars first
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
            print(f"WARNING: Could not load Img2Img pipeline for {model_id}: {e}")
            new_img2img = None

        try:
            new_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=HF_OFFLINE,
            )
            _enable_memory_saving_features(new_inpaint)
            new_inpaint.to(DEVICE)
        except Exception as e:
            print(f"WARNING: Could not load Inpaint pipeline for {model_id}: {e}")
            new_inpaint = None

    except Exception as e:
        print(f"ERROR loading model {model_id}: {e}")
        offline_note = " (HF_HUB_OFFLINE=1, so no network access)" if HF_OFFLINE else ""
        # Do NOT wipe old pipelines here; just report error.
        return False, f"❌ Failed to load model **{model_name}**{offline_note}: {e}"

    # If we got here, loading succeeded → free old, assign new
    for p in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
        try:
            del p
        except Exception:
            pass
    torch.cuda.empty_cache()

    pipe_txt2img = new_txt2img
    pipe_img2img = new_img2img
    pipe_inpaint = new_inpaint
    current_model_name = model_name

    print("Pipelines loaded.")
    return True, f"✅ Loaded model: **{model_name}**"


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

    # Scale so that the larger dimension becomes ~max_dim
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

    # Fallback to sliders
    return int(slider_w), int(slider_h)


def Image_fromarray(arr, mode="L"):
    return PilImage.fromarray(arr, mode=mode)


def extract_inpaint_image_and_mask(editor_value):
    """
    Convert a Gradio ImageEditor EditorValue (dict with background, layers, composite)
    into (base_image, mask_image) suitable for SDXL inpainting.

    Strategy:
    - base_image: use 'background'
    - mask_image: pixels where composite != background (user drew on them)
    """
    if editor_value is None:
        return None, None

    # EditorValue is a dict with keys: background, layers, composite
    bg = editor_value.get("background", None)
    comp = editor_value.get("composite", None)

    if bg is None or comp is None:
        return None, None

    # Convert to RGB
    bg = bg.convert("RGB")
    comp = comp.convert("RGB")

    bg_np = np.array(bg).astype("int16")
    comp_np = np.array(comp).astype("int16")

    # Difference per pixel
    diff = np.abs(comp_np - bg_np)
    diff_sum = diff.sum(axis=2)

    # Threshold: pixels that changed become white in the mask
    threshold = 10  # tweakable
    mask_np = (diff_sum > threshold).astype("uint8") * 255

    mask = Image_fromarray(mask_np, mode="L")
    return bg, mask


def clamp_for_heavy_models(width, height, batch_size):
    """
    For very heavy SDXL finetunes, clamp resolution a bit
    to reduce chances of CUDA OOM, but keep the requested batch_size.
    """
    max_pixels = 1024 * 640  # safe-ish upper bound
    pixels = width * height

    if pixels > max_pixels:
        # scale down while keeping aspect roughly similar
        scale = (max_pixels / float(pixels)) ** 0.5
        width = round_to_multiple(width * scale, 64)
        height = round_to_multiple(height * scale, 64)
        print(f"Heavy model: clamped resolution to {width}x{height}")

    if batch_size > 4:
        print(f"Heavy model: requested batch_size={batch_size}, clamping to 4")
        batch_size = 4

    # Do NOT touch batch_size here, so slider works as expected
    return width, height, batch_size



# -----------------------------------------------------------------------------
# Generation functions
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
    """
    global pipe_txt2img, pipe_img2img

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

    # Apply style preset
    styled_prompt, styled_negative = apply_style(prompt, neg_prompt, style_name)

    # Resolve final width/height (including ref-image-aware mode)
    effective_width, effective_height = get_effective_size(
        aspect_preset, width, height, ref_image
    )

    # Clamp a bit for heavy models to avoid OOM
    batch_size = max(1, int(batch_size))
    if current_model_name in HEAVY_MODELS:
        effective_width, effective_height, batch_size = clamp_for_heavy_models(
            effective_width, effective_height, batch_size
        )

    print(f"\n=== New generation request ===")
    print(f"Model: {current_model_name}")
    print(f"Style: {style_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Prompt: {styled_prompt!r}")
    print(f"Negative prompt: {styled_negative!r}")
    print(f"Seed: {seed}, Batch size: {batch_size}")
    print(f"Size: {effective_width}x{effective_height}")
    print(f"Guidance: {guidance}, Steps: {steps}")
    if ref_image is not None:
        print("Mode: img2img (using reference image)")
    else:
        print("Mode: text2img")

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

    # Save first image to disk (for logging / debugging)
    if images:
        filename = f"image_{int(time.time())}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        images[0].save(path)
        print(f"Saved first image to {path}")

    return images


def inpaint_image(
    prompt: str,
    inpaint_input,
    inpaint_strength: float,
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
    Inpainting using a Gradio ImageEditor value (image + mask painted on top).
    Returns list of images for Gradio Gallery.
    """
    global pipe_inpaint

    if pipe_inpaint is None:
        print("Inpaint pipeline not available for this model.")
        return None

    prompt = (prompt or "").strip()
    if not prompt:
        return None

    neg_prompt = (negative_prompt or "").strip() or None

    # Extract base image and mask from editor
    base_image, mask_image = extract_inpaint_image_and_mask(inpaint_input)
    if base_image is None or mask_image is None:
        print("Could not extract image and mask from ImageEditor input.")
        return None

    # Apply style preset
    styled_prompt, styled_negative = apply_style(prompt, neg_prompt, style_name)

    # Resolve final width/height using base_image for aspect match
    effective_width, effective_height = get_effective_size(
        aspect_preset, width, height, base_image
    )

    # Clamp for heavy models
    batch_size = max(1, int(batch_size))
    if current_model_name in HEAVY_MODELS:
        effective_width, effective_height, batch_size = clamp_for_heavy_models(
            effective_width, effective_height, batch_size
        )

    print(f"\n=== New inpaint request ===")
    print(f"Model: {current_model_name}")
    print(f"Style: {style_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Prompt: {styled_prompt!r}")
    print(f"Negative prompt: {styled_negative!r}")
    print(f"Seed: {seed}, Batch size: {batch_size}")
    print(f"Size: {effective_width}x{effective_height}")
    print(f"Guidance: {guidance}, Steps: {steps}")
    print(f"Inpaint strength: {inpaint_strength}")

    # Seed
    generator = None
    if seed and seed > 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Make sure mask is single-channel
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")

    # Set scheduler
    set_scheduler(pipe_inpaint, scheduler_name)

    # Replicate inputs for batch
    images_in = [base_image] * batch_size
    masks_in = [mask_image] * batch_size
    prompts = [styled_prompt] * batch_size
    neg_prompts = [styled_negative] * batch_size if styled_negative else None

    with torch.no_grad():
        result = pipe_inpaint(
            prompt=prompts,
            negative_prompt=neg_prompts,
            image=images_in,
            mask_image=masks_in,
            strength=inpaint_strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=effective_width,
            height=effective_height,
            generator=generator,
            num_images_per_prompt=1,
        )

    images = result.images

    if images:
        filename = f"inpaint_{int(time.time())}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        images[0].save(path)
        print(f"Saved first inpainted image to {path}")

    return images


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

# Model load handler with "loading" state and button disabling
def on_model_change(model_name):
    """
    This is a generator function so we can:
    1) Immediately show a loading message + disable buttons
    2) Then load the pipelines
    3) Finally enable buttons if load succeeded
    """
    global pipe_txt2img, pipe_img2img, pipe_inpaint, current_model_name

    # Step 1: show loading state, disable Generate/Inpaint
    loading_md = f"⏳ Loading model **{model_name}**. Please wait..."
    yield (
        loading_md,
        gr.update(interactive=False),  # Generate button
        gr.update(interactive=False),  # Inpaint button
    )

    # Step 2: actually load pipelines
    ok, msg = load_pipelines(model_name)

    # If load failed, decide whether buttons should be usable
    if not ok:
        any_loaded = pipe_txt2img is not None or pipe_img2img is not None or pipe_inpaint is not None
        # Keep using previous model if we had one, but tell the user
        if current_model_name:
            msg = msg + f"\n\nStill using previously loaded model: **{current_model_name}**."
        else:
            msg = msg + "\n\nNo model is currently loaded."
        yield (
            msg,
            gr.update(interactive=any_loaded),
            gr.update(interactive=any_loaded),
        )
    else:
        # Step 3: load succeeded -> enable buttons
        yield (
            msg,
            gr.update(interactive=True),
            gr.update(interactive=True),
        )


print("Starting DGX SDXL Image Lab (no model loaded at startup).")
initial_model_name = list(MODEL_CONFIGS.keys())[0]
initial_status = (
    "⚠️ No model loaded yet.\n\n"
    "Select a model from the dropdown above to start. "
    "Buttons will enable once the model is ready."
)

with gr.Blocks(title="DGX SDXL Image Lab") as demo:
    gr.Markdown(
        """
        # DGX SDXL Image Lab

        - **Tab 1:** Text / Img2Img
        - **Tab 2:** Inpainting (draw directly on the image)

        Global controls at the top apply to **all** modes:
        - Model selection (SDXL Base, Turbo, CyberRealistic, Juggernaut, RealVis, DreamShaper, ZavyChroma...)
        - Scheduler / Sampler (Default, Euler, DPM++)
        - Style preset (Photoreal, Cinematic, Anime, etc.)
        - Negative prompt
        - Seed (0 = random)
        - Aspect ratio preset (or match reference image)
        - Width / Height (used when preset is not overriding)
        - Batch size (multiple variations)
        """
    )

    # Global controls (apply to all modes)
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
                placeholder="Things you DON'T want (e.g. ugly, blurry, extra fingers, cartoon)...",
            )
            seed_in = gr.Number(
                value=0,
                label="Seed (0 = random)",
                precision=0,
            )
        with gr.Column(scale=1):
            aspect_dropdown = gr.Dropdown(
                choices=list(ASPECT_PRESETS.keys()),
                value="Keep sliders",
                label="Aspect ratio preset",
            )
            width_in = gr.Slider(
                minimum=512,
                maximum=1024,
                value=1024,
                step=64,
                label="Width",
            )
            height_in = gr.Slider(
                minimum=512,
                maximum=1024,
                value=1024,
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

    # ------------------------
    # Tab 1: Text / Img2Img
    # ------------------------
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

    # ------------------------
    # Tab 2: Inpainting
    # ------------------------
    with gr.Tab("Inpainting"):
        with gr.Row():
            with gr.Column(scale=2):
                inpaint_prompt_in = gr.Textbox(
                    lines=3,
                    label="Prompt (Inpainting)",
                    placeholder="Describe what you want to appear in the masked area...",
                )
                inpaint_image_in = gr.ImageEditor(
                    label="Image + mask (upload image, then paint where you want changes)",
                    type="pil",
                    image_mode="RGBA",
                    canvas_size=(768, 768),
                    height=512,
                )
                inpaint_strength_in = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.75,
                    step=0.05,
                    label="Inpaint strength (how strongly to apply changes)",
                )
                inpaint_guidance_in = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance scale",
                )
                inpaint_steps_in = gr.Slider(
                    minimum=10,
                    maximum=80,
                    value=30,
                    step=2,
                    label="Number of inference steps",
                )
                inpaint_btn = gr.Button(
                    "Inpaint",
                    variant="primary",
                    interactive=False,  # disabled until a model is loaded
                )
            with gr.Column(scale=3):
                inpaint_gallery = gr.Gallery(
                    label="Inpainted Images",
                    show_label=True,
                   columns=4,
                    height="auto",
                )

        inpaint_btn.click(
            fn=inpaint_image,
            inputs=[
                inpaint_prompt_in,
                inpaint_image_in,
                inpaint_strength_in,
                inpaint_guidance_in,
                inpaint_steps_in,
                negative_prompt_in,
                seed_in,
                width_in,
                height_in,
                batch_size_in,
                scheduler_dropdown,
                style_dropdown,
                aspect_dropdown,
            ],
            outputs=[inpaint_gallery],
        )

    # Bind model change AFTER buttons exist so we can enable/disable them
    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[model_status, generate_btn, inpaint_btn],
    )

# Queue is needed for generator-based UI updates (loading state)
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7867)

