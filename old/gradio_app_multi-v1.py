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

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda"

# Models you want available in the dropdown
MODEL_CHOICES = {
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    # Add more SDXL variants here:
    # "Photoreal SDXL": "author/photoreal-sdxl",
    # "Anime SDXL": "author/anime-sdxl",
}

# Available schedulers
SCHEDULER_CLASSES = {
    "Default": None,  # use whatever the pipeline comes with
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M (DPMSolverMultistep)": DPMSolverMultistepScheduler,
}

# Globals for pipelines
pipe_txt2img = None
pipe_img2img = None
pipe_inpaint = None
current_model_name = None


# -----------------------------------------------------------------------------
# Pipeline loading and scheduler helpers
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


def load_pipelines(model_name: str):
    """
    Load (or reload) SDXL pipelines for a given model.
    We keep txt2img, img2img, and inpaint variants.
    """
    global pipe_txt2img, pipe_img2img, pipe_inpaint, current_model_name

    model_id = MODEL_CHOICES[model_name]
    current_model_name = model_name

    print(f"\n=== Loading model: {model_name} ({model_id}) ===")

    # Free any existing pipelines
    for p in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
        try:
            del p
        except Exception:
            pass
    torch.cuda.empty_cache()

    # Text-to-image pipeline
    pipe_txt2img = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe_txt2img.to(DEVICE)

    # Image-to-image pipeline
    try:
        pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        pipe_img2img.to(DEVICE)
    except Exception as e:
        print(f"WARNING: Could not load Img2Img pipeline for {model_id}: {e}")
        pipe_img2img = None

    # Inpainting pipeline
    try:
        pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        pipe_inpaint.to(DEVICE)
    except Exception as e:
        print(f"WARNING: Could not load Inpaint pipeline for {model_id}: {e}")
        pipe_inpaint = None

    print("Pipelines loaded.")


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
):
    """
    Text-to-image or Img2Img, depending on whether ref_image is provided.
    Returns a list of images (for Gradio Gallery).
    """
    global pipe_txt2img, pipe_img2img

    prompt = (prompt or "").strip()
    if not prompt:
        return None

    neg_prompt = (negative_prompt or "").strip() or None

    print(f"\n=== New generation request ===")
    print(f"Model: {current_model_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Prompt: {prompt!r}")
    print(f"Negative prompt: {neg_prompt!r}")
    print(f"Seed: {seed}, Batch size: {batch_size}")
    print(f"Size: {width}x{height}")
    print(f"Guidance: {guidance}, Steps: {steps}")
    if ref_image is not None:
        print("Mode: img2img (using reference image)")
    else:
        print("Mode: text2img")

    # Seed
    generator = None
    if seed and seed > 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Ensure batch_size is at least 1
    batch_size = max(1, int(batch_size))

    # Set scheduler
    if ref_image is None:
        if pipe_txt2img is None:
            print("Text2Img pipeline not available.")
            return None
        set_scheduler(pipe_txt2img, scheduler_name)
        pipe = pipe_txt2img
    else:
        if pipe_img2img is None:
            print("Img2Img pipeline not available for this model.")
            return None
        set_scheduler(pipe_img2img, scheduler_name)
        pipe = pipe_img2img

    with torch.no_grad():
        if ref_image is None:
            # Text-to-image
            result = pipe(
                prompt=[prompt] * batch_size,
                negative_prompt=[neg_prompt] * batch_size if neg_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=1,  # because we replicate prompts
            )
        else:
            # Image-to-image
            result = pipe(
                prompt=[prompt] * batch_size,
                negative_prompt=[neg_prompt] * batch_size if neg_prompt else None,
                image=[ref_image] * batch_size,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
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
    threshold = 10  # you can tweak this
    mask_np = (diff_sum > threshold).astype("uint8") * 255

    mask = Image_fromarray(mask_np, mode="L")
    return bg, mask


# Small helper because PIL might not be imported explicitly
from PIL import Image as PilImage


def Image_fromarray(arr, mode="L"):
    return PilImage.fromarray(arr, mode=mode)


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

    base_image, mask_image = extract_inpaint_image_and_mask(inpaint_input)
    if base_image is None or mask_image is None:
        print("Could not extract image and mask from ImageEditor input.")
        return None

    print(f"\n=== New inpaint request ===")
    print(f"Model: {current_model_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Prompt: {prompt!r}")
    print(f"Negative prompt: {neg_prompt!r}")
    print(f"Seed: {seed}, Batch size: {batch_size}")
    print(f"Size: {width}x{height}")
    print(f"Guidance: {guidance}, Steps: {steps}")
    print(f"Inpaint strength: {inpaint_strength}")

    # Seed
    generator = None
    if seed and seed > 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Ensure batch_size is at least 1
    batch_size = max(1, int(batch_size))

    # Make sure mask is single-channel
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")

    # Set scheduler
    set_scheduler(pipe_inpaint, scheduler_name)

    # Replicate inputs for batch
    images_in = [base_image] * batch_size
    masks_in = [mask_image] * batch_size
    prompts = [prompt] * batch_size
    neg_prompts = [neg_prompt] * batch_size if neg_prompt else None

    with torch.no_grad():
        result = pipe_inpaint(
            prompt=prompts,
            negative_prompt=neg_prompts,
            image=images_in,
            mask_image=masks_in,
            strength=inpaint_strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
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

def on_model_change(model_name):
    load_pipelines(model_name)
    return f"Loaded model: {model_name}"


print("Loading initial SDXL model...")
initial_model_name = list(MODEL_CHOICES.keys())[0]
load_pipelines(initial_model_name)

with gr.Blocks(title="DGX SDXL Image Lab") as demo:
    gr.Markdown(
        """
        # DGX SDXL Image Lab

        - **Tab 1:** Text / Img2Img  
        - **Tab 2:** Inpainting (draw directly on the image)  

        Global controls at the top apply to **all** modes:
        - Model selection (SDXL Base, SDXL Turbo, etc.)
        - Scheduler / Sampler (Default, Euler, DPM++)
        - Negative prompt
        - Seed (0 = random)
        - Width / Height
        - Batch size (multiple variations)
        """
    )

    # Global controls (apply to all modes)
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value=initial_model_name,
                label="Model",
            )
            scheduler_dropdown = gr.Dropdown(
                choices=list(SCHEDULER_CLASSES.keys()),
                value="Default",
                label="Scheduler / Sampler",
            )
            model_status = gr.Markdown(f"Loaded model: {initial_model_name}")
        with gr.Column(scale=1):
            negative_prompt_in = gr.Textbox(
                lines=2,
                label="Negative prompt",
                placeholder="Things you DON'T want (e.g. ugly, blurry, extra fingers)...",
            )
            seed_in = gr.Number(
                value=0,
                label="Seed (0 = random)",
                precision=0,
            )
        with gr.Column(scale=1):
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

    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[model_status],
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
                # Gradio 6: use ImageEditor instead of Image(tool='sketch')
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
                inpaint_btn = gr.Button("Inpaint", variant="primary")
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
            ],
            outputs=[inpaint_gallery],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7867)

