import os
import time
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import gradio as gr

OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading SDXL text-to-image pipeline (this can take a bit)...")
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Text-to-image pipeline
pipe_txt2img = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe_txt2img.to("cuda")

print("Loading SDXL img2img pipeline...")
pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe_img2img.to("cuda")

def generate_image(prompt: str, ref_image, strength: float, guidance: float, steps: int):
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    print(f"\n=== New request ===")
    print(f"Prompt: {prompt!r}")
    if ref_image is not None:
        print("Mode: img2img (using reference image)")
    else:
        print("Mode: text2img")

    filename = f"image_{int(time.time())}.png"
    path = os.path.join(OUTPUT_DIR, filename)

    with torch.no_grad():
        if ref_image is None:
            # Pure text-to-image
            result = pipe_txt2img(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            image = result.images[0]
        else:
            # Image-to-image with reference
            result = pipe_img2img(
                prompt=prompt,
                image=ref_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            image = result.images[0]

    image.save(path)
    print(f"Saved image to {path}")
    return image

with gr.Blocks(title="DGX SDXL Image Generator") as demo:
    gr.Markdown(
        """
        # DGX SDXL Image Generator  
        Enter a prompt, optionally upload a reference image.  
        If a reference image is provided, img2img mode is used; otherwise text2img.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_in = gr.Textbox(
                lines=3,
                label="Prompt",
                placeholder="Describe the image you want...",
            )
            ref_image_in = gr.Image(
                label="Optional reference image (img2img)",
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
                maximum=60,
                value=30,
                step=2,
                label="Number of inference steps",
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_in, ref_image_in, strength_in, guidance_in, steps_in],
        outputs=[output_image],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7867)
