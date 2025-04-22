# generate_images_lora.py

from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
import torch
import csv
import os

def generate_image(prompt, model_path, lora_weights, output_dir="image_generation/results", output_csv="image_generation/results/generated_images.csv"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None  # <== disables the NSFW filter
    ).to("cuda")

    # Load LoRA weights into the UNet
    peft_config = PeftConfig.from_pretrained(lora_weights)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights, adapter_name="default")

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image with a unique name in the results folder
    image_file_name = f"generated_image_{len(os.listdir(output_dir)) + 1}.png"
    image_path = os.path.join(output_dir, image_file_name)
    image.save(image_path)

    # Append the image file name and prompt to the CSV file
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["Image File Name", "Prompt"])  # Write header if file doesn't exist
        writer.writerow([image_file_name, prompt])

    print(f"Image saved as {image_path} and logged in {output_csv}")

if __name__ == "__main__":
    # Define the prompt and paths
    prompt = "The image shows a close-up view of a lesion on the anterior torso of a 30-year-old female patient. The lesion appears as a dark brown, irregularly shaped area with a fuzzy texture, surrounded by a lighter background with some hair and small bubbles visible."
    model_path = "runwayml/stable-diffusion-v1-5"  # Base model path
    lora_weights = "/dtu/blackhole/07/203495/ADLCV-project/image_generation/lora_weights_isic_5000"  # Path to LoRA weights

    # Generate the image and log it in the CSV
    generate_image(prompt, model_path, lora_weights)