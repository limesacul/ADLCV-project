from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model (consider fine-tuned models for medical imaging)
model_id = "CompVis/stable-diffusion-v1-4"  # Replace with a medical-specific model if available

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU if available

# Generate an image from a medical text prompt
prompt = "A relaistic medical grade MRI scan of a human brain"
image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_medical_image2.png")
image.show()
