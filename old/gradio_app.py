import os
import time
import torch
from diffusers import DiffusionPipeline
import gradio as gr

# Where to save images
OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model (this can take a bit)...")

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipeline.to("cuda")

def generate_image(prompt: str):
    prompt = prompt.strip()
    if not prompt:
        return None

    print(f"Generating image for prompt: {prompt!r}")
    with torch.no_grad():
        image = pipeline(prompt=prompt).images[0]

    filename = f"image_{int(time.time())}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)
    print(f"Saved image to {path}")

    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, label="Prompt", placeholder="Describe the image you want..."),
    outputs=gr.Image(label="Generated Image"),
    title="DGX Image Generator",
    description="Enter a text prompt and generate an image using Stable Diffusion XL.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
