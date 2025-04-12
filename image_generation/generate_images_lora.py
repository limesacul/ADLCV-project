# generate_images.py

from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, model_path, lora_weights):
    # Load the fine-tuned model
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.unet.load_attn_procs(lora_weights)
    pipeline = pipeline.to("cuda")

    # Generate image
    image = pipeline(prompt).images[0]
    image.show()

if __name__ == "__main__":
    prompt = "A futuristic city with flying cars"
    generate_image(prompt, "runwayml/stable-diffusion-v1-5", "lora_weights")
